"""Measurement primitives.

Three rules this module exists to enforce:

1. **Time on the device, not the host.** CUDA launches are asynchronous, so
   `perf_counter()` around a step measures launch time, not execution time. Pairs of
   CUDA events are recorded in-stream and read back *after* the measured region, so
   nothing in the loop ever blocks on a synchronise.
2. **Never time and introspect in the same pass.** The saved-tensor ledger adds a Python
   call per saved tensor; the allocator snapshot adds far more. Both run in their own
   pass, after timing is done.
3. **Report allocated and reserved separately.** `max_memory_allocated` is what the
   theory predicts. `max_memory_reserved` is what the process holds from the driver; the
   gap is caching-allocator fragmentation and belongs in its own column.
"""

from __future__ import annotations

import contextlib
import gc
import statistics
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch
from torch import nn

MIB = 1024 * 1024


# --------------------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------------------


class EventTimer:
    """Per-step device timing with deferred readback.

    `start()`/`stop()` only enqueue events. `results_ms()` synchronises once, at the end,
    and returns one elapsed time per step.
    """

    def __init__(self, capacity: int) -> None:
        self._starts = [torch.cuda.Event(enable_timing=True) for _ in range(capacity)]
        self._stops = [torch.cuda.Event(enable_timing=True) for _ in range(capacity)]
        self._n = 0
        self._wall_start: float | None = None
        self._wall_elapsed: float | None = None

    def begin_region(self) -> None:
        torch.cuda.synchronize()
        self._wall_start = time.perf_counter()

    def end_region(self) -> None:
        torch.cuda.synchronize()
        assert self._wall_start is not None
        self._wall_elapsed = time.perf_counter() - self._wall_start

    def start(self) -> None:
        self._starts[self._n].record()

    def stop(self) -> None:
        self._stops[self._n].record()
        self._n += 1

    def results_ms(self) -> list[float]:
        torch.cuda.synchronize()
        return [self._starts[i].elapsed_time(self._stops[i]) for i in range(self._n)]

    @property
    def wall_s(self) -> float:
        assert self._wall_elapsed is not None
        return self._wall_elapsed


def timing_summary(step_ms: list[float], wall_s: float, tokens_per_step: int) -> dict[str, Any]:
    """Robust statistics plus the device-versus-wall gap.

    Median rather than mean: consumer GPUs clock down under sustained load and one stall
    moves a mean. `busy_fraction` is device time over wall time — well under 1.0 means
    the loop is launch-bound, and until that is fixed no kernel-level speedup will show
    up in end-to-end throughput.
    """
    s = sorted(step_ms)
    n = len(s)
    med = statistics.median(s)
    return {
        "steps_timed": n,
        "median_ms": med,
        "mean_ms": statistics.fmean(s),
        "p10_ms": s[int(0.10 * (n - 1))],
        "p90_ms": s[int(0.90 * (n - 1))],
        "iqr_ms": s[int(0.75 * (n - 1))] - s[int(0.25 * (n - 1))],
        "device_sum_s": sum(s) / 1e3,
        "wall_s": wall_s,
        "busy_fraction": (sum(s) / 1e3) / wall_s,
        "tokens_per_s": tokens_per_step / (med / 1e3),
        "steps_per_s": 1e3 / med,
    }


# --------------------------------------------------------------------------------------
# memory
# --------------------------------------------------------------------------------------


def allocated() -> int:
    return torch.cuda.memory_allocated()


def peak_allocated() -> int:
    return torch.cuda.max_memory_allocated()


def reset_peak() -> None:
    torch.cuda.reset_peak_memory_stats()


@dataclass
class MemoryMarks:
    """Allocator readings at labelled points within one step.

    The caching allocator maintains its counters on the host at allocation time, so
    reading them costs nothing and needs no synchronise.
    """

    marks: dict[str, int] = field(default_factory=dict)
    peaks: dict[str, int] = field(default_factory=dict)

    def mark(self, label: str) -> None:
        self.marks[label] = allocated()

    @contextlib.contextmanager
    def peak_of(self, label: str) -> Iterator[None]:
        """Peak allocated inside the block, isolated from earlier phases."""
        reset_peak()
        yield
        self.peaks[label] = peak_allocated()

    def to_dict(self) -> dict[str, Any]:
        return {"marks": dict(self.marks), "peaks": dict(self.peaks)}


# --------------------------------------------------------------------------------------
# saved-tensor ledger
# --------------------------------------------------------------------------------------


@dataclass
class LedgerEntry:
    module: str
    kind: str  # activation | weight_cast | param
    shape: tuple[int, ...]
    dtype: str
    storage_bytes: int


class SavedTensorLedger:
    """Exactly what backward is holding, measured rather than reasoned about.

    `saved_tensors_hooks` sees every tensor stashed for backward. Three details make the
    numbers trustworthy:

    * Accounting is per *storage*, deduplicated by pointer. Views share memory and a
      tensor saved twice costs once, so counting tensors would double-count both.
    * The pack hook returns the tensor unchanged, so the ledger holds no extra reference
      and cannot extend any tensor's lifetime.
    * Autocast's cached weight casts are identified by autograd shape — a cast of a leaf
      parameter is a `ToCopyBackward` whose input is an `AccumulateGrad` — rather than by
      matching sizes, which would confuse an activation that happens to be
      parameter-shaped.

    Module attribution comes from a forward-hook stack, which is why an entry can be
    blamed on `blocks.3.mlp.fc1` rather than on "some matmul".
    """

    # Autograd nodes that only rearrange metadata. A chain made solely of these,
    # terminating at AccumulateGrad, means the tensor is a reinterpretation of a
    # parameter rather than a computed activation.
    _PASSTHROUGH = frozenset(
        {
            "ToCopyBackward0",
            "TBackward0",
            "ViewBackward0",
            "UnsafeViewBackward0",
            "ReshapeAliasBackward0",
            "ExpandBackward0",
            "PermuteBackward0",
            "TransposeBackward0",
        }
    )

    def __init__(self, model: nn.Module) -> None:
        self._param_storages = {p.untyped_storage().data_ptr() for p in model.parameters()}
        self._model = model
        self._stack: list[str] = []
        self._seen: set[int] = set()
        self.entries: list[LedgerEntry] = []
        self._handles: list[Any] = []

    @classmethod
    def _is_param_derived(cls, t: torch.Tensor, depth: int = 8) -> bool:
        """Walk the autograd graph from `t` towards its leaf.

        A matmul does not save autocast's cast of the weight directly; it saves the
        *transpose* of that cast, so the chain is TBackward -> ToCopyBackward ->
        AccumulateGrad. Matching on shape instead would be wrong in both directions: the
        saved tensor's shape is the transposed shape, which for a square weight
        coincidentally matches and for a rectangular one does not.
        """
        gf = t.grad_fn
        while gf is not None and depth > 0:
            name = type(gf).__name__
            if name == "AccumulateGrad":
                return True
            if name not in cls._PASSTHROUGH:
                return False
            nxt = [f for f, _ in gf.next_functions if f is not None]
            if len(nxt) != 1:
                return False
            gf = nxt[0]
            depth -= 1
        return False

    def _classify(self, t: torch.Tensor) -> str:
        if t.untyped_storage().data_ptr() in self._param_storages:
            return "param"
        if self._is_param_derived(t):
            return "weight_cast"
        return "activation"

    def _pack(self, t: torch.Tensor) -> torch.Tensor:
        if isinstance(t, torch.Tensor) and t.is_cuda:
            storage = t.untyped_storage()
            key = storage.data_ptr()
            if key not in self._seen:
                self._seen.add(key)
                self.entries.append(
                    LedgerEntry(
                        module=self._stack[-1] if self._stack else "<root>",
                        kind=self._classify(t),
                        shape=tuple(t.shape),
                        dtype=str(t.dtype).removeprefix("torch."),
                        storage_bytes=storage.nbytes(),
                    )
                )
        return t

    def _push(self, name: str) -> None:
        self._stack.append(name)

    def _pop(self) -> None:
        self._stack.pop()

    @contextlib.contextmanager
    def capture(self) -> Iterator["SavedTensorLedger"]:
        # Both hooks must return None. A forward_pre_hook's return value replaces the
        # module's inputs and a forward_hook's replaces its output, so returning the
        # result of list.append/list.pop would rewrite the model's activations.
        for name, module in self._model.named_modules():
            if not name:
                continue
            self._handles.append(
                module.register_forward_pre_hook(lambda _m, _i, n=name: self._push(n))
            )
            self._handles.append(module.register_forward_hook(lambda _m, _i, _o: self._pop()))
        try:
            with torch.autograd.graph.saved_tensors_hooks(self._pack, lambda t: t):
                yield self
        finally:
            for h in self._handles:
                h.remove()
            self._handles.clear()
            self._stack.clear()

    # -- aggregation ------------------------------------------------------------------

    def totals_by_kind(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for e in self.entries:
            out[e.kind] += e.storage_bytes
        return dict(out)

    @staticmethod
    def bucket_of(module: str) -> str:
        """Collapse per-layer module paths into one algebraic term.

        `blocks.3.mlp.fc1` and `blocks.0.mlp.fc1` are the same term multiplied by
        `n_layer`; keeping them apart would make the table `n_layer` times longer without
        adding information.
        """
        parts = module.split(".")
        if parts[0] == "blocks" and len(parts) > 2:
            return "blocks.*." + ".".join(parts[2:])
        if parts[0] == "blocks":
            return "blocks.*"
        return module

    def activation_buckets(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for e in self.entries:
            if e.kind == "activation":
                out[self.bucket_of(e.module)] += e.storage_bytes
        return dict(out)

    def to_dict(self) -> dict[str, Any]:
        return {
            "totals_by_kind": self.totals_by_kind(),
            "activation_buckets": self.activation_buckets(),
            "entries": [e.__dict__ for e in self.entries],
        }


# --------------------------------------------------------------------------------------
# allocator census
# --------------------------------------------------------------------------------------


def live_cuda_storages() -> dict[int, int]:
    """Pointer -> requested bytes for every Python-visible CUDA tensor.

    Only complete when no autograd graph is alive: tensors saved for backward are held by
    C++ nodes and may have no Python object, so calling this mid-graph would misreport
    activations as untracked allocations.
    """
    out: dict[int, int] = {}
    with warnings.catch_warnings():
        # Touching every object trips deprecation shims on module attributes.
        warnings.simplefilter("ignore")
        for obj in gc.get_objects():
            try:
                if isinstance(obj, torch.Tensor) and obj.is_cuda:
                    storage = obj.untyped_storage()
                    out[storage.data_ptr()] = storage.nbytes()
            except Exception:
                continue
    return out


def allocator_census() -> dict[str, Any]:
    """Explain `memory_allocated()` down to the byte.

    Three things separate the number the theory predicts from the number the allocator
    reports, and both of the latter two are invisible unless looked for:

    * **Block padding.** `memory_allocated` accumulates *block* sizes, not requested
      sizes. A large-pool block is only split when the remainder would exceed 1 MiB, so a
      tensor can be charged for a block larger than it asked for.
    * **Non-tensor blocks.** cuBLAS and cuBLASLt take their workspaces from the same
      caching allocator, so they count towards `memory_allocated` while belonging to no
      tensor at all.
    * **Reserved but free.** Blocks the allocator holds from the driver and has not handed
      out. This is the fragmentation column.

    Must be called at rest, with no autograd graph alive. See `live_cuda_storages`.
    """
    known = live_cuda_storages()
    requested = blocks = nontensor = 0
    nontensor_sizes: list[int] = []
    for seg in torch.cuda.memory_snapshot():
        addr = seg["address"]
        for blk in seg["blocks"]:
            if blk["state"] == "active_allocated":
                if addr in known:
                    requested += known[addr]
                    blocks += blk["size"]
                else:
                    nontensor += blk["size"]
                    nontensor_sizes.append(blk["size"])
            addr += blk["size"]
    allocated_now = allocated()
    return {
        "tensor_requested": requested,
        "tensor_blocks": blocks,
        "block_padding": blocks - requested,
        "nontensor_blocks": nontensor,
        "nontensor_sizes": sorted(nontensor_sizes, reverse=True)[:8],
        "allocated": allocated_now,
        "reserved": torch.cuda.memory_reserved(),
        "unexplained": allocated_now - blocks - nontensor,
    }
