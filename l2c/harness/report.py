"""Predicted versus measured, and the record of both.

The contract: the prediction is written in `prediction.toml` **before** the step runs.
The harness reads it, runs, and prints the delta. Committing the prediction first is
what makes it a prediction rather than a rationalization.

TOML rather than YAML so that `tomllib` from the standard library can read it — one
fewer dependency, and the "omit a value and the harness just reports the measurement"
convention falls out naturally, since TOML has no null.

Nothing here touches the GPU: everything reads a result dict and returns text or writes
JSON. Measurement must never depend on presentation.

In accelerate: reporting lives in `accelerate.tracking`, which adapts a common
`GeneralTracker` interface onto TensorBoard, W&B, MLflow and others, driven by
`Accelerator(log_with=...)` and `Accelerator.log`. Its trackers are wired so that only
the main process writes — `@on_main_process` — which is the concern that replaces plain
`print` once step 4 introduces more than one rank.
"""

import json
import platform
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch

from l2c.harness.measure import describe_device
from l2c.paths import results_file


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


def environment(device: torch.device) -> dict[str, object]:
    """Everything needed to reproduce a number, recorded with the number.

    The precision knobs are here for a specific reason. `float32_matmul_precision`
    decides whether an fp32 matmul runs at full 24-bit significand on the CUDA cores or
    is truncated to TF32's 11 bits on the tensor cores. Same dtypes, same memory,
    different kernels, and roughly a 2x difference in step time. It is inherited from a
    global default that has changed across torch releases, so a run that does not record
    it cannot be compared with a run from six months later.
    """
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": describe_device(device),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }


def record(
    step: str,
    *,
    preset: dict[str, object],
    run: dict[str, object],
    environment: dict[str, object],
    predicted: dict[str, float],
    actual: dict[str, object],
) -> Path:
    """Append one run to `results.jsonl` for cross-step plotting.

    `preset` and `run` stay in separate sub-objects. Flattening them into one dict makes
    an unset CLI flag, which arrives as None, overwrite the real preset value it shadows;
    the sweep's x axis then reads null for exactly the runs that used a default.
    """
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "step": step,
        "preset": preset,
        "run": run,
        "environment": environment,
        "predicted": predicted,
        "actual": actual,
    }
    path = results_file()
    with path.open("a") as handle:
        handle.write(json.dumps(entry) + "\n")
    return path


def load_results(step: str | None = None) -> list[dict]:
    """Every recorded run, optionally filtered to one step."""
    path = results_file()
    if not path.exists():
        return []
    entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [e for e in entries if step is None or e["step"] == step]
