"""Measurement primitives. Shared by every script so that numbers stay comparable.

Three rules this module exists to enforce.

**1. Time on the device, not the host.** CUDA launches are asynchronous, so
`perf_counter()` around a step measures how fast Python *enqueues* kernels, not how
long they run. Pairs of CUDA events are recorded in-stream and read back once, after
the loop, so no step ever blocks on a synchronize. Wall time is also reported, from a
single synchronize at each end: device time answers "how efficient are the kernels",
wall time answers "how many steps per second do I actually get", and they are
different questions.

**2. Never time and introspect in the same pass.** The memory staircase and the
saved-tensor ledger both perturb what they observe — the ledger adds a Python call per
saved tensor. They run in their own passes, before and after the timed one.

**3. Distinguish the four answers to "how much memory does this use".** Conflating them
is what makes memory arithmetic feel unpredictable:

    requested        sum of tensor storage bytes       <- what theory predicts
    memory_allocated sum of allocator *block* bytes    <- requested + block padding
    memory_reserved  what the allocator holds          <- allocated + free cached blocks
    nvidia-smi       reserved + CUDA context (~0.3-0.6 GB) + driver overhead

The first gap is the one that surprises people. `memory_allocated` is *not* the sum of
tensor sizes: the caching allocator rounds each request up, and when splitting a segment
would leave a remainder too small to reuse it hands that remainder to the block instead.
So an identical request can occupy different amounts depending on the state of the pool —
for a 135M-parameter model the parameters alone come to 513.13 MiB requested against
520.88 MiB of blocks, 1.5% of padding that no arithmetic can predict.

Arithmetic is therefore checked against `requested` (see `state_inventory`), and the
allocator's view is reported next to it so the padding is quantified rather than
mistaken for a broken prediction.

In accelerate: there is no counterpart, because accelerate does not measure. The
closest relatives live in accelerate/utils/memory.py and *react* to memory pressure
rather than predict it: `find_executable_batch_size` (line 119) retries a step with a
halved batch after catching an OOM, `should_reduce_batch_size` (100) classifies the
exception, and `release_memory` (70) drops references and empties the cache. For
getting numbers *out*, accelerate offers `accelerate.tracking` and `Accelerator.print`.
"""

import statistics
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn

type LossFn = Callable[[nn.Module, object], torch.Tensor]


def mib(num_bytes: float) -> float:
    return num_bytes / 1024**2


def gib(num_bytes: float) -> float:
    return num_bytes / 1024**3


def require_cuda() -> torch.device:
    """Fail loudly rather than silently measuring nothing.

    Every memory number here comes from the CUDA caching allocator. On CPU torch
    reports no peak at all, so a CPU fallback would emit zeros that look like
    measurements — worse than an error. The lab targets one GPU; `CUDA_VISIBLE_DEVICES`
    selects which.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("This lab measures GPU memory and needs CUDA. torch.cuda.is_available() is False.")
    return torch.device("cuda")


def describe_device(device: torch.device) -> dict[str, object]:
    """Identity of the GPU, recorded with every result.

    Static facts only. If a number looks wrong, the first question is which card produced
    it — see `foreign_memory_bytes` for whether that card was busy.
    """
    properties = torch.cuda.get_device_properties(device)
    return {
        "name": properties.name,
        "capability": f"{properties.major}.{properties.minor}",
        "total_gib": round(gib(properties.total_memory), 2),
    }


def memory_in_use(device: torch.device) -> int:
    """Total memory resident on the device, sampled before this run allocates anything.

    Not attributable to other processes, and deliberately not named as though it were:
    querying it creates this process's CUDA primary context, which is itself a few hundred
    MiB and is included in the reading. There is no way to separate the two through torch,
    since the allocator does not account for the context.

    It is still the reading worth recording, because the useful signal is the *excess* over
    what an idle run reports. A card driving a display shows several GiB more, and its
    clocks move with whatever the compositor is doing — which invalidates the timings.
    Sample it first regardless: once a model is resident this is dominated by our own
    allocations and says nothing at all.
    """
    free, total = torch.cuda.mem_get_info(device)
    return total - free


# --------------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Both answers at one instant, so the gap between them is never lost."""

    allocated: int
    reserved: int
    peak_allocated: int
    peak_reserved: int

    @property
    def allocator_overhead(self) -> int:
        """Reserved but not allocated: block padding and non-tensor workspaces."""
        return self.peak_reserved - self.peak_allocated


def snapshot(device: torch.device) -> Snapshot:
    return Snapshot(
        allocated=torch.cuda.memory_allocated(device),
        reserved=torch.cuda.memory_reserved(device),
        peak_allocated=torch.cuda.max_memory_allocated(device),
        peak_reserved=torch.cuda.max_memory_reserved(device),
    )


def reset_peak(device: torch.device) -> None:
    torch.cuda.reset_peak_memory_stats(device)


def requested_bytes(tensors: Iterable[torch.Tensor]) -> int:
    """Sum of distinct CUDA storages, which is what the arithmetic predicts.

    Deduplicated by storage address, so a tied weight counts once and a view counts with
    its base. CPU tensors are skipped: AdamW keeps its `step` counter on the host, and it
    is not GPU memory.
    """
    seen: dict[int, int] = {}
    for tensor in tensors:
        if tensor is None or not tensor.is_cuda:
            continue
        storage = tensor.untyped_storage()
        seen[storage.data_ptr()] = storage.nbytes()
    return sum(seen.values())


@dataclass(frozen=True, slots=True)
class StateInventory:
    """The three model-state buckets, measured as requested bytes.

    Independent of allocator behavior, so these can be checked against 4/4/8 bytes per
    parameter exactly rather than approximately.
    """

    param_bytes: int
    grad_bytes: int
    optimizer_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.param_bytes + self.grad_bytes + self.optimizer_bytes


def state_inventory(model: nn.Module, optimizer: torch.optim.Optimizer) -> StateInventory:
    """Measure model states directly, as requested bytes.

    Must be called after a first `optimizer.step()` and before `zero_grad()`: AdamW creates
    its moments lazily on that first step, and `set_to_none=True` discards the gradients.
    Both preconditions are checked, because outside that window this returns a plausible
    number with a bucket silently missing — and a budget that is short by exactly
    4 or 8 bytes per parameter is the hardest kind of wrong to notice.
    """
    parameters = list(model.parameters())
    if not optimizer.state:
        raise RuntimeError(
            "optimizer state is empty, so its moments would count as zero. Call "
            "state_inventory() after the first optimizer.step()."
        )
    if all(p.grad is None for p in parameters):
        raise RuntimeError(
            "no parameter has a gradient, so gradients would count as zero. Call "
            "state_inventory() before optimizer.zero_grad(set_to_none=True)."
        )
    return StateInventory(
        param_bytes=requested_bytes(parameters),
        grad_bytes=requested_bytes(p.grad for p in parameters),
        optimizer_bytes=requested_bytes(
            value for state in optimizer.state.values() for value in state.values() if isinstance(value, torch.Tensor)
        ),
    )


@dataclass(frozen=True, slots=True)
class Phases:
    """Memory at each point where a new bucket first appears.

    Tensors do not all exist from the start. The staircase:

        after_model      parameters
        after_forward    parameters + activations (the graph is alive)
        after_backward   parameters + gradients (activations have been freed)
        after_step       parameters + gradients + optimizer state

    That last transition catches people: AdamW allocates `exp_avg` and `exp_avg_sq`
    lazily on the **first** `step()`, not in `__init__`. Measure before it and the
    budget is short by exactly 8 bytes per parameter.

    `initial_loss` comes from this pass's forward, which is the only forward that ever
    runs on pristine weights. Reading it from the training loop instead would report
    the loss *after* an optimizer step, and the `ln(vocab_size)` check would be quietly
    off by an amount small enough to accept.
    """

    after_model: int
    after_forward: int
    after_backward: int
    after_step: int
    initial_loss: float

    @property
    def activations(self) -> int:
        """Everything the forward pass added: saved tensors, logits and the loss."""
        return self.after_forward - self.after_model

    @property
    def gradients(self) -> int:
        return self.after_backward - self.after_model

    @property
    def optimizer_states(self) -> int:
        return self.after_step - self.after_backward


def measure_phases(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: object,
    device: torch.device,
    loss_from: LossFn,
) -> Phases:
    """Run one step, reading allocated memory between phases.

    This *is* training step 0 — it updates the weights. It is kept out of the timed
    loop because a synchronize between every phase would distort the timing, and
    because interleaving measurement into the loop obscures the loop, which is the
    thing being learned.
    """
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    after_model = torch.cuda.memory_allocated(device)

    loss = loss_from(model, batch)
    initial_loss = loss.item()
    torch.cuda.synchronize(device)
    after_forward = torch.cuda.memory_allocated(device)

    loss.backward()
    del loss  # drop the last reference so the graph can be released
    torch.cuda.synchronize(device)
    after_backward = torch.cuda.memory_allocated(device)

    optimizer.step()
    torch.cuda.synchronize(device)
    after_step = torch.cuda.memory_allocated(device)

    return Phases(after_model, after_forward, after_backward, after_step, initial_loss)


# --------------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Timings:
    """Per-step device times plus total wall time.

    Percentiles rather than a mean: step-time distributions have a tail from allocator
    growth and clock drift, so the mean reports the tail while the median reports the
    machine.
    """

    device_ms: list[float]
    wall_seconds: float

    @property
    def median_ms(self) -> float:
        return statistics.median(self.device_ms)

    @property
    def p10_ms(self) -> float:
        return self._decile(0)

    @property
    def p90_ms(self) -> float:
        return self._decile(8)

    def _decile(self, index: int) -> float:
        if len(self.device_ms) < 2:
            return self.device_ms[0]
        return statistics.quantiles(self.device_ms, n=10)[index]

    @property
    def steps_per_second(self) -> float:
        """Wall-clock throughput: what the run actually delivers, launch gaps included."""
        return len(self.device_ms) / self.wall_seconds

    @property
    def device_steps_per_second(self) -> float:
        """Device-time throughput: the ceiling if the host never became the bottleneck."""
        return 1000.0 / self.median_ms

    def tokens_per_second(self, tokens_per_step: int) -> float:
        return self.steps_per_second * tokens_per_step


class StepTimer:
    """Times each step with a CUDA event pair, never synchronizing inside the loop.

    Events are cheap to record and are read back only in `finish()`. The loop
    therefore stays free of the two things that would corrupt it: a `synchronize()`
    that serializes the pipeline, and a `loss.item()` that does the same implicitly.
    """

    def __init__(self, device: torch.device, capacity: int) -> None:
        self._device = device
        self._capacity = capacity
        self._starts = [torch.cuda.Event(enable_timing=True) for _ in range(capacity)]
        self._ends = [torch.cuda.Event(enable_timing=True) for _ in range(capacity)]
        self._recorded = 0
        self._wall_start: float | None = None

    @contextmanager
    def step(self) -> Iterator[None]:
        if self._recorded >= self._capacity:
            raise RuntimeError(f"StepTimer was built for {self._capacity} steps and has recorded {self._recorded}.")
        if self._wall_start is None:
            # Begin the wall clock from a quiet GPU, so queued warmup work is not
            # billed to the first timed step.
            torch.cuda.synchronize(self._device)
            self._wall_start = time.perf_counter()

        index = self._recorded
        self._starts[index].record()
        try:
            yield
        finally:
            self._ends[index].record()
            self._recorded += 1

    def finish(self) -> Timings:
        if self._wall_start is None:
            raise RuntimeError("StepTimer.finish() called before any step.")
        torch.cuda.synchronize(self._device)
        wall_seconds = time.perf_counter() - self._wall_start
        device_ms = [
            start.elapsed_time(end)
            for start, end in zip(self._starts[: self._recorded], self._ends[: self._recorded], strict=True)
        ]
        return Timings(device_ms=device_ms, wall_seconds=wall_seconds)
