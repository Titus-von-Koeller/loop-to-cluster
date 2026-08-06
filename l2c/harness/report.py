"""Predicted versus measured, and the record of both.

The contract: the prediction is written in `prediction.toml` **before** the step runs.
The harness reads it, runs, and prints the delta. Committing the prediction first is
what makes it a prediction rather than a rationalization.

TOML rather than YAML so that `tomllib` from the standard library can read it — one
fewer dependency, and the "omit a value and the harness just reports the measurement"
convention falls out naturally, since TOML has no null.

Nothing here touches the GPU: everything reads a result dict and returns text.
Measurement must never depend on presentation. Where results are stored is
`l2c.harness.runs`.

In accelerate: reporting lives in `accelerate.tracking`, which adapts a common
`GeneralTracker` interface onto TensorBoard, W&B, MLflow and others, driven by
`Accelerator(log_with=...)` and `Accelerator.log`. Its trackers are wired so that only
the main process writes — `@on_main_process` — which is the concern that replaces plain
`print` once step 4 introduces more than one rank.
"""

import platform
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from l2c.harness import runs
from l2c.harness.measure import describe_device, gib, mib
from l2c.harness.rows import RowSpec


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


def load_prediction(step_dir: str | Path) -> dict[str, float]:
    """Read `prediction.toml` from a step directory. Empty dict if absent."""
    path = Path(step_dir) / "prediction.toml"
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def build_rows(
    specs: Sequence[RowSpec],
    predicted: dict[str, float],
    actual: dict[str, object],
    tolerances: dict[str, float] | None = None,
) -> list[Row]:
    """Pair each spec's key with both sides. An absent prediction leaves the row open."""
    overrides = tolerances or {}
    built = []
    for spec in specs:
        measured = actual.get(spec.measured_key())
        built.append(
            Row(
                label=spec.label,
                predicted=predicted.get(spec.key),
                actual=None if measured is None else float(measured),
                fmt=spec.fmt,
                tolerance_pct=overrides.get(spec.key, spec.tolerance_pct),
            )
        )
    return built


def publish(
    step: str,
    specs: Sequence[RowSpec],
    *,
    step_dir: str | Path,
    preset: dict[str, object],
    run: dict[str, object],
    environment: dict[str, object],
    actual: dict[str, object],
    headline: str,
    derived: dict[str, float] | None = None,
    notes: Sequence[str] = (),
) -> Path:
    """Print the comparison and record the run. The only reporting a step performs.

    `derived` carries predictions the harness computes rather than reads; a
    `prediction.toml` entry of the same name wins, since a hand derivation is the thing
    being checked.
    """
    prediction = load_prediction(step_dir)
    tolerances = prediction.get("tolerance", {})
    claims = {key: value for key, value in prediction.items() if key != "tolerance"}
    predicted = (derived or {}) | claims
    tokens_per_step = run["batch_size"] * run["seq_len"]

    print(f"\n{step}   {headline}   {tokens_per_step:,} tokens/step\n")
    print(table(build_rows(specs, predicted, actual, tolerances)))
    print(
        f"\nstep time   median {actual['median_step_ms']:.1f} ms   "
        f"p10 {actual['p10_step_ms']:.1f}   p90 {actual['p90_step_ms']:.1f}"
    )
    print(f"throughput  {actual['tokens_per_second']:,.0f} tokens/sec")
    print(f"loss        {actual['initial_loss']:.4f} -> {actual['final_loss']:.4f}")
    print(
        f"memory      peak allocated {actual['peak_mib']:,.1f} MiB   "
        f"reserved {actual['peak_reserved_mib']:,.1f} MiB   "
        f"allocator overhead {actual['allocator_overhead_mib']:,.1f} MiB"
    )
    for note in notes:
        print(note)
    print(_saved_for_backward(actual["saved_tensors"]))

    # The prediction as written, not as compared: the derived half is arithmetic that
    # can be recomputed, while `prediction.toml` is the claim that was committed first.
    path = runs.save(
        step,
        preset=preset,
        run=run,
        environment=environment,
        predicted=prediction,
        actual=actual,
    )
    print(f"\nrecorded to {path}")
    return path


def _saved_for_backward(summary: dict[str, object]) -> str:
    lines = ["\nsaved for backward:"]
    for category, byte_count in summary["bytes_by_category"].items():
        lines.append(f"  {category:<14}{mib(byte_count):>10,.1f} MiB")
    lines.append("  by dtype:")
    for dtype, byte_count in summary["bytes_by_dtype"].items():
        lines.append(f"    {dtype:<12}{mib(byte_count):>10,.1f} MiB")
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


