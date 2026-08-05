"""The training loop, in four phases.

    setup -> warmup -> timed -> anatomy

Phase separation is the whole methodology:

* **warmup** absorbs everything that happens once: cuBLAS and cuDNN handle creation,
  kernel autotuning, the caching allocator growing its pools, and — under fp16 — the
  initial loss-scale backoff. Timing across warmup measures start-up, not steady state.
* **timed** touches nothing but the step. No `loss.item()`, no memory queries, no hooks.
  Losses land in a preallocated device buffer and come back to the host once, at the end;
  a per-step `.item()` would synchronise every step and hide exactly the launch bubbles
  the busy-fraction diagnostic exists to expose.
* **anatomy** runs afterwards with fine-grained memory marks and the saved-tensor
  ledger, and its timings are discarded.

Peak memory is reset at the start of the timed phase, so the reported peak is a
steady-state peak rather than the first-step allocation ramp.

The memory marks are taken by `_step` itself rather than by a parallel instrumented
copy of it. Two code paths that are supposed to compute the same thing eventually will
not, and the loss curve is the one thing here that must be exact.
"""

from __future__ import annotations

import contextlib
from typing import Any

import torch
from torch import nn

from .config import RunConfig
from .data import SyntheticCorpus
from .instrument import (
    EventTimer,
    MemoryMarks,
    SavedTensorLedger,
    allocated,
    allocator_census,
    peak_allocated,
    reset_peak,
    timing_summary,
)
from .model import TinyGPT
from .predict import predict


def _apply_backend_flags(cfg: RunConfig) -> None:
    # "highest" is IEEE fp32; "high" permits TF32 tensor cores for fp32 matmuls. The
    # fp32 baseline has to pin this, because the default has moved across torch versions
    # and a silently-TF32 baseline would understate AMP's speedup.
    torch.set_float32_matmul_precision("high" if cfg.spec["tf32"] else "highest")


def _sdpa_backend(cfg: RunConfig) -> str:
    """Which attention kernel this dtype will actually get.

    fp32 cannot use flash attention, so the fp32 baseline silently runs the mem-efficient
    kernel. Both are tiled and neither materialises the S x S score matrix, so the
    comparison stays fair — but that has to be recorded rather than assumed: a fallback
    to the `math` backend would add a B*H*S*S term to the baseline alone and flatter AMP
    enormously.
    """
    m, d = cfg.model, cfg.data
    q = torch.zeros(
        d.batch_size, m.n_head, d.seq_len, m.head_dim, device="cuda", dtype=cfg.act_dtype
    )
    try:
        params = torch.backends.cuda.SDPAParams(q, q, q, None, 0.0, True, False)
        if torch.backends.cuda.can_use_flash_attention(params, False):
            return "flash"
        if torch.backends.cuda.can_use_efficient_attention(params, False):
            return "mem_efficient"
        return "math"
    except Exception:  # API drift: record ignorance rather than a guess
        return "unknown"
    finally:
        del q


def _build(
    cfg: RunConfig,
) -> tuple[TinyGPT, torch.optim.Optimizer, SyntheticCorpus, dict[str, int]]:
    device = torch.device(f"cuda:{cfg.device}")
    torch.cuda.set_device(device)
    torch.manual_seed(cfg.seed)

    marks: dict[str, int] = {"empty": allocated()}
    corpus = SyntheticCorpus(cfg.data, cfg.model.vocab_size, device)
    marks["corpus"] = allocated()

    model = TinyGPT(cfg.model)
    model.init_weights(cfg.seed)
    model.to(device)
    marks["model"] = allocated()

    if cfg.optim.impl == "foreach":
        kwargs: dict[str, Any] = {"foreach": True}
    elif cfg.optim.impl == "fused":
        kwargs = {"fused": True}
    else:
        kwargs = {"foreach": False, "fused": False}
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        betas=cfg.optim.betas,
        weight_decay=cfg.optim.weight_decay,
        **kwargs,
    )
    return model, opt, corpus, marks


def _autocast(cfg: RunConfig):
    return torch.autocast(
        "cuda", dtype=cfg.autocast_dtype, enabled=cfg.autocast_dtype is not None
    )


def _step(
    cfg: RunConfig,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    x: torch.Tensor,
    y: torch.Tensor,
    marks: MemoryMarks | None = None,
) -> torch.Tensor:
    """One optimiser step. Returns the loss, still on the device and still async.

    With `marks`, records allocator readings and per-phase peaks. With `marks=None` — the
    timed path — the branches collapse to nothing measurable.
    """
    peak = marks.peak_of if marks is not None else (lambda _label: contextlib.nullcontext())
    mark = marks.mark if marks is not None else (lambda _label: None)

    opt.zero_grad(set_to_none=True)
    mark("before_forward")
    with peak("forward"), _autocast(cfg):
        loss = model(x, y)
    mark("after_forward")

    with peak("backward"):
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
    mark("after_backward")

    with peak("optimizer"):
        if scaler.is_enabled():
            if cfg.optim.grad_clip > 0:
                # Gradients are still multiplied by the loss scale at this point.
                # Clipping before unscaling would compare the norm against a threshold
                # scaled by a number that changes during training — the classic
                # AMP-plus-clipping bug.
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            if cfg.optim.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            opt.step()
    mark("after_optimizer")
    return loss.detach()


def _tensor_bytes(model: nn.Module, opt: torch.optim.Optimizer) -> dict[str, int]:
    """Requested bytes for the persistent tensors, read straight off their storages.

    This is the quantity the analytic model predicts. `memory_allocated()` is a different
    quantity — it counts allocator blocks — so comparing theory against it directly would
    charge the theory for the allocator's split policy. Call while gradients are alive,
    which means after a step and before the next `zero_grad`.
    """
    params = sum(p.untyped_storage().nbytes() for p in model.parameters())
    grads = sum(
        p.grad.untyped_storage().nbytes() for p in model.parameters() if p.grad is not None
    )
    state = 0
    for entry in opt.state.values():
        for v in entry.values():
            if torch.is_tensor(v) and v.is_cuda:
                state += v.untyped_storage().nbytes()
    return {"parameters": params, "gradients": grads, "optimizer_state": state}


def _ledger_pass(
    cfg: RunConfig,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    corpus: SyntheticCorpus,
) -> dict[str, Any]:
    """Forward once under `saved_tensors_hooks` to enumerate what backward holds.

    Its own pass: a Python callback per saved tensor would distort both the timings and
    the marks. Backward runs afterwards only to release the graph.
    """
    x, y = corpus.batch(0)
    opt.zero_grad(set_to_none=True)
    ledger = SavedTensorLedger(model)
    with ledger.capture():
        before = allocated()
        with _autocast(cfg):
            loss = model(x, y)
        after = allocated()
    loss.backward()
    opt.zero_grad(set_to_none=True)
    return {
        **ledger.to_dict(),
        "allocated_delta_forward": after - before,
    }


def run(cfg: RunConfig) -> dict[str, Any]:
    _apply_backend_flags(cfg)
    model, opt, corpus, build_marks = _build(cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.spec["scaler"])
    sdpa = _sdpa_backend(cfg)

    total = cfg.warmup + cfg.steps
    losses = torch.zeros(total, device="cuda", dtype=torch.float32)
    scales = torch.zeros(total, device="cuda", dtype=torch.float32)

    # -- warmup ------------------------------------------------------------------------
    # Step 0 is instrumented: it is the only step at which no gradients and no optimiser
    # state exist yet, so its marks separate parameters, gradients and optimiser state
    # from one another by subtraction.
    cold = MemoryMarks()
    for step in range(cfg.warmup):
        x, y = corpus.batch(step)
        losses[step] = _step(cfg, model, opt, scaler, x, y, marks=cold if step == 0 else None)
        if scaler.is_enabled():
            # Copied device-side; get_scale() would synchronise on every step.
            scales[step] = scaler._scale

    # -- timed -------------------------------------------------------------------------
    reset_peak()
    timer = EventTimer(cfg.steps)
    timer.begin_region()
    for step in range(cfg.warmup, total):
        x, y = corpus.batch(step)
        timer.start()
        losses[step] = _step(cfg, model, opt, scaler, x, y)
        timer.stop()
        if scaler.is_enabled():
            scales[step] = scaler._scale
    timer.end_region()

    result: dict[str, Any] = {
        "config": cfg.to_dict(),
        "env": {
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(cfg.device),
            "capability": ".".join(map(str, torch.cuda.get_device_capability(cfg.device))),
            "sdpa_backend": sdpa,
            "fp32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "param_count": model.param_count(),
        "matmul_param_count": model.matmul_param_count(),
        "loss": losses.tolist(),  # a single host transfer, after the timed region
        "loss_scale": scales.tolist() if scaler.is_enabled() else None,
        "timing": timing_summary(timer.results_ms(), timer.wall_s, cfg.tokens_per_step),
        "memory": {
            "build_marks": build_marks,
            "cold_step": cold.to_dict(),
            "peak_allocated": peak_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(),
            "corpus_bytes": corpus.nbytes(),
        },
        "prediction": predict(cfg, sdpa_backend=sdpa),
    }

    if cfg.anatomy:
        # A steady-state instrumented step: unlike the cold step, gradients and optimiser
        # state already exist, which is the state the reported peak was measured in.
        steady = MemoryMarks()
        x, y = corpus.batch(0)
        _step(cfg, model, opt, scaler, x, y, marks=steady)
        result["memory"]["steady_step"] = steady.to_dict()
        # Gradients are still alive here and the graph is not, which is exactly the
        # condition both of these need.
        result["memory"]["tensor_bytes"] = _tensor_bytes(model, opt)
        result["memory"]["census"] = allocator_census()
        result["ledger"] = _ledger_pass(cfg, model, opt, corpus)
    if cfg.snapshot:
        _dump_snapshot(cfg, model, opt, scaler, corpus)
        result["snapshot"] = cfg.snapshot
    return result


def _dump_snapshot(cfg, model, opt, scaler, corpus) -> None:
    """Allocator trace for pytorch.org/memory_viz.

    The aggregate numbers say how much; this says which tensors, allocated from which
    line of Python, and how long they lived. Kept out of the timed region because
    recording every allocation is not free.
    """
    torch.cuda.memory._record_memory_history(max_entries=200_000)
    for step in range(2):
        x, y = corpus.batch(step)
        _step(cfg, model, opt, scaler, x, y)
    torch.cuda.synchronize()
    torch.cuda.memory._dump_snapshot(cfg.snapshot)
    torch.cuda.memory._record_memory_history(enabled=None)
