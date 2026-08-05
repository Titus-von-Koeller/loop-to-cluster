"""Measurement primitives. Shared by every step so that numbers stay comparable.

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

**3. Report `allocated` and `reserved` separately.** There are three different true
answers to "how much memory does this use", and conflating them is what makes memory
arithmetic feel unpredictable:

    memory_allocated   bytes in live tensors            <- what theory predicts
    memory_reserved    what the caching allocator holds <- allocated + free blocks
    nvidia-smi         reserved + CUDA context (~0.3-0.6 GB) + driver overhead

Predictions are checked against `allocated`. `reserved` is recorded alongside so the
gap — allocator block padding and cuBLAS workspaces — is visible rather than mysterious.

In accelerate: there is no counterpart, because accelerate does not measure. The
closest relatives live in accelerate/utils/memory.py and *react* to memory pressure
rather than predict it: `find_executable_batch_size` (line 119) retries a step with a
halved batch after catching an OOM, `should_reduce_batch_size` (100) classifies the
exception, and `release_memory` (70) drops references and empties the cache. For
getting numbers *out*, accelerate offers `accelerate.tracking` and `Accelerator.print`.
"""

import statistics
import time
from collections.abc import Callable, Iterator
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
        raise RuntimeError(
            "This lab measures GPU memory and needs CUDA. torch.cuda.is_available() is False."
        )
    return torch.device("cuda")


def describe_device(device: torch.device) -> dict[str, object]:
    """Identity of the GPU, recorded with every result.

    Worth capturing because a second GPU driving a display starts several hundred MiB
    down and its clocks move with whatever the compositor is doing. If a number looks
    wrong, the first question is which card produced it.
    """
    properties = torch.cuda.get_device_properties(device)
    free, total = torch.cuda.mem_get_info(device)
    return {
        "name": properties.name,
        "capability": f"{properties.major}.{properties.minor}",
        "total_gib": round(gib(properties.total_memory), 2),
        "free_gib_at_start": round(gib(free), 2),
        "used_by_others_gib": round(gib(total - free), 2),
    }


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
            raise RuntimeError(
                f"StepTimer was built for {self._capacity} steps and has recorded "
                f"{self._recorded}."
            )
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
            for start, end in zip(
                self._starts[: self._recorded], self._ends[: self._recorded], strict=True
            )
        ]
        return Timings(device_ms=device_ms, wall_seconds=wall_seconds)
