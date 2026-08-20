"""Printing a comparison, and capturing what a number depends on.

Two things only: a table that puts a prediction next to a measurement with a verdict,
and the environment record that goes into every result file. Nothing here touches the
GPU — presentation must never be able to move a measurement.

A predicted value that missed is the most useful line in the table, so `table` reports
the delta rather than hiding a miss behind a pass/fail.
"""

import platform
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from l2c.harness.measure import describe_device, gib


@dataclass(frozen=True, slots=True)
class Row:
    """One line of the comparison table.

    `tolerance_pct` encodes how exact the claim is. Model states are arithmetic and
    should match to a rounding error, so they get a tight tolerance; an activation
    estimate is a first guess and gets a loose one. Writing the tolerance down forces
    the question "how close would count as confirmation?" to be answered before the
    number appears.
    """

    label: str
    predicted: float | None
    actual: float | None
    fmt: str = ",.1f"
    tolerance_pct: float = 1.0


def table(rows: Sequence[Row]) -> str:
    header = f"{'quantity':<28}{'predicted':>16}{'measured':>16}{'delta':>10}  verdict"
    lines = [header, "-" * len(header)]
    for row in rows:
        measured = "--" if row.actual is None else format(row.actual, row.fmt)
        if row.predicted is None or row.actual is None:
            lines.append(f"{row.label:<28}{'--':>16}{measured:>16}{'':>10}")
            continue
        predicted = format(row.predicted, row.fmt)
        if row.predicted == 0:
            delta, verdict = "n/a", ""
        else:
            delta_pct = (row.actual - row.predicted) / row.predicted * 100
            delta = f"{delta_pct:+.2f}%"
            verdict = "ok" if abs(delta_pct) <= row.tolerance_pct else "OFF"
        lines.append(f"{row.label:<28}{predicted:>16}{measured:>16}{delta:>10}  {verdict}")
    return "\n".join(lines)


def environment(
    device: torch.device, *, memory_in_use_bytes: int | None = None
) -> dict[str, object]:
    """Everything needed to reproduce a number, recorded with the number.

    The precision knobs are here for a specific reason. `float32_matmul_precision`
    decides whether an fp32 matmul runs at full 24-bit significand on the CUDA cores or
    is truncated to TF32's 11 bits on the tensor cores. Same dtypes, same memory,
    different kernels, and roughly a 2x difference in step time. It is inherited from a
    global default that has changed across torch releases, so a run that does not record
    it cannot be compared with a run from six months later.
    """
    captured = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": describe_device(device),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }
    if memory_in_use_bytes is not None:
        captured["device_used_gib_at_start"] = round(gib(memory_in_use_bytes), 2)
    return captured
