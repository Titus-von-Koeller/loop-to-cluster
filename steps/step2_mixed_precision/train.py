"""Step 2 — mixed precision. One arm per process.

`diff steps/step1_training_loop/train.py steps/step2_mixed_precision/train.py` is the
lesson. Everything added is one of four things:

1. a `--precision` flag, and `precision.apply()` replacing the hardcoded
   `set_float32_matmul_precision("highest")`
2. the forward wrapped in `torch.autocast`
3. a `GradScaler`, and the ordering it forces: scale, backward, **unscale**, clip, step
4. the anatomy pass wrapped in the same autocast, so the ledger sees the arm's real dtypes

Note what is *not* here. No parameter is converted, no gradient is converted, and the
optimizer is untouched. `torch.autocast` casts op *inputs*, and the master weights stay
fp32 — which is why model states do not shrink, and why they in fact grow by the size of
the bf16 copies autocast caches.

Run one arm directly, or all of them through `compare.py`:

    python steps/step2_mixed_precision/train.py --precision bf16
    python steps/step2_mixed_precision/compare.py

In accelerate this whole file collapses to `Accelerator(mixed_precision="bf16")` plus
`accelerator.backward(loss)`. What that hides, and what running the arms by hand shows:

- the autocast context is installed by wrapping `model.forward`, not the call site
  (accelerate/accelerator.py near line 1820)
- the scaler is created for you from the distributed type (`get_grad_scaler`, near
  line 583) and only when the dtype needs one
- `accelerator.clip_grad_norm_` unscales before clipping (line 2944), which is the
  ordering constraint below that is easy to get wrong by hand
- `AcceleratedOptimizer.step` (accelerate/optimizer.py:162-175) calls `scaler.step` then
  `scaler.update`, and infers from whether the inner step ran that gradients overflowed,
  exposing it as `optimizer.step_was_skipped()`
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from l2c.common import data
from l2c.common import model as model_lib
from l2c.harness import ledger, measure, predict, report
from l2c.harness.precision import Precision

STEP = "step2_mixed_precision"
STEP_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", type=Precision, choices=list(Precision), required=True)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = measure.require_cuda()
    # Sampled before anything is allocated. Includes this process's CUDA context, so the
    # signal is the excess over an idle run: a busy GPU invalidates every timing below.
    memory_at_start = measure.memory_in_use(device)
    precision: Precision = args.precision
    precision.apply()

    preset = model_lib.SMOLLM2_135M
    if args.num_layers is not None:
        preset = model_lib.replace_preset(preset, num_hidden_layers=args.num_layers)

    net = model_lib.build_model(preset, seed=args.seed).to(device)
    net.train()
    num_params = model_lib.count_parameters(net)

    dataset = data.build_packed_dataset(model_lib.TOKENIZER, seq_len=args.seq_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
        # Same seed as every other arm, so all four see identical batches in identical
        # order. Without that, "the loss curve is unchanged" would be untestable.
        generator=data.batch_generator(args.seed),
    )
    batches = data.endless(loader)

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate)

    # A disabled scaler is a passthrough, so one code path serves all four arms. It is
    # enabled only for fp16, whose 5-bit exponent puts the smallest normal value at
    # 6.1e-5 — inside the range real gradients occupy. bf16 keeps fp32's exponent and
    # needs no scaling at all.
    scaler = torch.amp.GradScaler("cuda", enabled=precision.needs_scaler)

    def loss_from(module: torch.nn.Module, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        input_ids = batch[0].to(device, non_blocking=True)
        with torch.autocast(
            "cuda", dtype=precision.autocast_dtype, enabled=precision.uses_autocast
        ):
            return module(input_ids=input_ids, labels=input_ids).loss

    def training_step(batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        loss = loss_from(net, batch)
        scaler.scale(loss).backward()
        # Unscale before clipping. Clipping scaled gradients would compare their norm
        # against a threshold inflated by whatever scale factor the scaler chose, so the
        # effective max_grad_norm would drift with the scaler.
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
        # Skips the update entirely if any gradient is inf or nan, then lowers the scale.
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        return loss

    # ---- phases: the memory staircase, and step 0 ---------------------------------
    measure.reset_peak(device)
    phases = measure.measure_phases(net, optimizer, next(batches), device, loss_from)
    inventory_states = measure.state_inventory(net, optimizer)
    optimizer.zero_grad(set_to_none=True)

    # ---- warmup -------------------------------------------------------------------
    # Also where fp16's initial loss scale settles: GradScaler starts at 65536 and halves
    # on every overflow, so the first few steps are usually skipped updates.
    for _ in range(args.warmup_steps):
        training_step(next(batches))

    # ---- timed --------------------------------------------------------------------
    measure.reset_peak(device)
    timer = measure.StepTimer(device, capacity=args.num_steps)
    loss_history = torch.empty(args.num_steps, device=device)

    for index in range(args.num_steps):
        batch = next(batches)
        with timer.step():
            loss = training_step(batch)
        loss_history[index] = loss.detach()

    timings = timer.finish()
    losses = loss_history.tolist()
    peak = measure.snapshot(device)

    # Updates that actually landed. AdamW increments a per-parameter step counter, so
    # comparing it with the number of attempts counts the steps the scaler skipped
    # without a synchronize inside the loop. accelerate surfaces the same fact as
    # `AcceleratedOptimizer.step_was_skipped()`.
    attempted = 1 + args.warmup_steps + args.num_steps
    any_state = next(iter(optimizer.state.values()))
    applied = int(any_state["step"].item())

    # ---- anatomy ------------------------------------------------------------------
    with ledger.record(net, vocab_size=preset.vocab_size) as inventory:
        held = loss_from(net, next(batches))
    del held
    optimizer.zero_grad(set_to_none=True)

    # ---- predicted versus measured ------------------------------------------------
    states = predict.model_states(num_params, optimizer="adamw")
    cache = (
        predict.autocast_weight_cache(net, precision.autocast_dtype)
        if precision.uses_autocast
        else predict.WeightCache(0, 0, 0)
    )
    prediction = report.load_prediction(STEP_DIR)
    tokens_per_step = args.batch_size * args.seq_len

    actual = {
        "precision": str(precision),
        "num_params": num_params,
        "initial_loss": phases.initial_loss,
        "final_loss": losses[-1],
        "model_states_mib": measure.mib(inventory_states.total_bytes),
        "weight_cache_mib": measure.mib(inventory.bytes_in(ledger.WEIGHT_CASTS)),
        "weight_cache_tensors": inventory.count_in(ledger.WEIGHT_CASTS),
        # The raw forward-pass delta bundles the weight cache, because autocast allocates
        # those copies during the forward. Subtracting it separates "what the graph holds
        # to differentiate" from "what mixed precision added to hold the weights twice".
        "activations_mib": measure.mib(phases.activations),
        "activations_excl_cache_mib": measure.mib(
            phases.activations - inventory.bytes_in(ledger.WEIGHT_CASTS)
        ),
        "saved_activations_mib": measure.mib(inventory.bytes_in(ledger.ACTIVATIONS)),
        "saved_logits_mib": measure.mib(inventory.bytes_in(ledger.LOGITS)),
        "saved_still_fp32_mib": measure.mib(inventory.by_dtype().get("float32", 0)),
        "peak_mib": measure.mib(peak.peak_allocated),
        "peak_reserved_mib": measure.mib(peak.peak_reserved),
        "median_step_ms": timings.median_ms,
        "p10_step_ms": timings.p10_ms,
        "p90_step_ms": timings.p90_ms,
        "steps_per_second": timings.steps_per_second,
        "tokens_per_second": timings.tokens_per_second(tokens_per_step),
        "final_grad_scale": scaler.get_scale(),
        "steps_attempted": attempted,
        "steps_applied": applied,
        "steps_skipped": attempted - applied,
        "saved_tensors": inventory.summary(),
        "loss_curve": losses,
    }

    rows = [
        report.Row(
            "model states (MiB)",
            measure.mib(states.total_bytes),
            actual["model_states_mib"],
            ",.1f",
            0.1,
        ),
        report.Row(
            "weight cache (MiB)",
            measure.mib(cache.total_bytes),
            actual["weight_cache_mib"],
            ",.1f",
            0.1,
        ),
        report.Row(
            "weight cache tensors",
            float(cache.num_tensors),
            float(actual["weight_cache_tensors"]),
            ",.0f",
            0.0,
        ),
        report.Row(
            "initial loss",
            predict.expected_initial_loss(preset.vocab_size),
            phases.initial_loss,
            ",.4f",
            1.0,
        ),
        report.Row(
            "activations (MiB)",
            prediction.get(f"activations_mib_{precision}"),
            actual["activations_excl_cache_mib"],
            ",.1f",
            15.0,
        ),
        report.Row(
            "peak allocated (MiB)",
            prediction.get(f"peak_mib_{precision}"),
            actual["peak_mib"],
            ",.1f",
            10.0,
        ),
        report.Row(
            "median step (ms)",
            prediction.get(f"median_step_ms_{precision}"),
            actual["median_step_ms"],
            ",.1f",
            20.0,
        ),
    ]

    print(f"\n{STEP}   precision={precision}   {tokens_per_step:,} tokens/step\n")
    print(report.table(rows))
    print(
        f"\nstep time   median {timings.median_ms:.1f} ms   "
        f"p10 {timings.p10_ms:.1f}   p90 {timings.p90_ms:.1f}"
    )
    print(f"throughput  {actual['tokens_per_second']:,.0f} tokens/sec")
    print(f"loss        {phases.initial_loss:.4f} -> {losses[-1]:.4f}")
    print(
        f"scaler      scale {actual['final_grad_scale']:,.0f}   "
        f"{actual['steps_skipped']} of {attempted} updates skipped"
    )
    print("\nsaved for backward:")
    for category, byte_count in inventory.by_category().items():
        print(f"  {category:<14}{measure.mib(byte_count):>10,.1f} MiB")
    print("  by dtype:")
    for dtype, byte_count in inventory.by_dtype().items():
        print(f"    {dtype:<12}{measure.mib(byte_count):>10,.1f} MiB")

    environment = report.environment(device, memory_in_use_bytes=memory_at_start)
    run = {
        k: str(v) if k == "precision" else v for k, v in vars(args).items() if k != "json_out"
    }
    payload = {
        "step": STEP,
        "preset": model_lib.preset_dict(preset),
        "run": run,
        "environment": environment,
        "predicted": prediction,
        "actual": actual,
    }
    path = report.record(
        STEP,
        preset=payload["preset"],
        run=run,
        environment=environment,
        predicted=prediction,
        actual=actual,
    )
    print(f"\nappended to {path}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
