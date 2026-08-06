"""One file per run, named by the configuration that produced it.

A run is identified by what would change its numbers: the step, the resolved model
preset, and the arguments that steer measurement. Two invocations that describe the
same configuration therefore land on the same file, whether an argument was passed
explicitly or left at its default. That makes the cache and the permanent record the
same artifact — re-running a comparison to reformat a table costs nothing, and a
result cannot be recorded twice under two names.

Flags that shape the preset are excluded from the key, because their effect is already
in `preset`. Keeping both would give `--num-layers 30` and an unmodified default
separate files despite describing one model.

`commit` is recorded with every run. Measurements are only comparable across runs that
the same code produced, so a sweep spanning a change to the training loop has to be
detectable after the fact rather than fitted straight through.

In accelerate the equivalent concern is `ProjectConfiguration(project_dir=...)`
(accelerate/utils/dataclasses.py:916), which takes the output directory explicitly and
tracks checkpoint iterations under it.
"""

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from l2c.paths import ROOT, runs_dir

#: Arguments whose only effect is to build the preset. `preset` already records the
#: outcome, so including them would key one configuration under two names.
PRESET_SHAPING = frozenset({"num_layers"})

_UNSAFE = re.compile(r"[^A-Za-z0-9]+")


def _material(step: str, preset: dict, run: dict) -> dict:
    """The subset of a run's description that its numbers depend on."""
    return {
        "step": step,
        "preset": preset,
        "run": {k: v for k, v in sorted(run.items()) if k not in PRESET_SHAPING},
    }


def key(step: str, preset: dict, run: dict) -> str:
    material = json.dumps(_material(step, preset, run), sort_keys=True, default=str)
    return hashlib.sha256(material.encode()).hexdigest()[:10]


def _slug(preset: dict, run: dict) -> str:
    parts = [f"L{preset.get('num_hidden_layers')}"]
    parts += [
        f"{name}{run[name]}"
        for name in ("batch_size", "seq_len", "num_steps")
        if name in run
    ]
    if "precision" in run:
        parts.append(str(run["precision"]))
    return _UNSAFE.sub("-", "-".join(parts)).strip("-").lower()


def path_for(step: str, preset: dict, run: dict) -> Path:
    """Where this configuration's result lives. Readable, with the key as a suffix."""
    return runs_dir() / f"{step}-{_slug(preset, run)}-{key(step, preset, run)}.json"


def _commit() -> str | None:
    """The revision that produced a measurement, or None outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


@dataclass(frozen=True, slots=True)
class Run:
    """A recorded run, as it is stored and as `fit` reads it back."""

    step: str
    preset: dict
    run: dict
    environment: dict
    predicted: dict
    actual: dict
    timestamp: str
    commit: str | None

    def as_dict(self) -> dict:
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "commit": self.commit,
            "preset": self.preset,
            "run": self.run,
            "environment": self.environment,
            "predicted": self.predicted,
            "actual": self.actual,
        }


def save(
    step: str,
    *,
    preset: dict,
    run: dict,
    environment: dict,
    predicted: dict,
    actual: dict,
    timestamp: str | None = None,
    commit: str | None = None,
) -> Path:
    """Write one run, replacing any earlier measurement of the same configuration."""
    entry = Run(
        step=step,
        preset=preset,
        run=run,
        environment=environment,
        predicted=predicted,
        actual=actual,
        timestamp=timestamp or datetime.now(UTC).isoformat(),
        commit=commit if commit is not None else _commit(),
    )
    path = path_for(step, preset, run)
    path.write_text(json.dumps(entry.as_dict(), indent=2) + "\n")
    return path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def load_all(step: str | None = None) -> list[dict]:
    """Every recorded run, optionally filtered to one step, ordered by configuration."""
    directory = runs_dir()
    entries = [load(path) for path in sorted(directory.glob("*.json"))]
    return [e for e in entries if step is None or e["step"] == step]
