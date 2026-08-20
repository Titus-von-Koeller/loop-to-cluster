"""One JSON file per run, named by the script and the configuration that produced it.

Every profiled script writes the same schema, because a comparison between two topics
is only meaningful if both sides were recorded the same way:

    script       which script produced this
    timestamp    when
    commit       the revision it ran at
    config       the knobs that would change the numbers
    environment  torch, CUDA and device identity
    predicted    what was expected, where a prediction was made
    measured     what was observed

Two runs describing the same configuration land on the same file, so re-running a
comparison to redraw a plot costs nothing and one result cannot be recorded twice
under two names.

`commit` is recorded because measurements only compare across runs the same code
produced. A change to the training loop moves the numbers for reasons no plot can see.
"""

import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from l2c.paths import ROOT, results_dir

_UNSAFE = re.compile(r"[^A-Za-z0-9]+")


def key(script: str, config: dict[str, Any]) -> str:
    material = json.dumps({"script": script, "config": config}, sort_keys=True, default=str)
    return hashlib.sha256(material.encode()).hexdigest()[:10]


def _slug(config: dict[str, Any]) -> str:
    parts = [f"{name}{config[name]}" for name in sorted(config) if _is_scalar(config[name])]
    return _UNSAFE.sub("-", "-".join(parts)).strip("-").lower()[:70] or "default"


def _is_scalar(value: object) -> bool:
    return isinstance(value, str | int | float | bool)


def path_for(script: str, config: dict[str, Any]) -> Path:
    """Where this configuration's result lives. Readable, with the key as a suffix."""
    return results_dir() / f"{script}-{_slug(config)}-{key(script, config)}.json"


def commit() -> str | None:
    """The revision that produced a measurement, or None outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except OSError, subprocess.CalledProcessError:
        return None
    return result.stdout.strip() or None


def save(
    script: str,
    *,
    config: dict[str, Any],
    environment: dict[str, Any],
    measured: dict[str, Any],
    predicted: dict[str, Any] | None = None,
) -> Path:
    """Write one run, replacing any earlier measurement of the same configuration."""
    entry = {
        "script": script,
        "timestamp": datetime.now(UTC).isoformat(),
        "commit": commit(),
        "config": config,
        "environment": environment,
        "predicted": predicted or {},
        "measured": measured,
    }
    path = path_for(script, config)
    path.write_text(json.dumps(entry, indent=2, default=str) + "\n")
    return path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def load_all(script: str | None = None) -> list[dict]:
    """Every recorded run, optionally filtered to one script."""
    entries = [load(path) for path in sorted(results_dir().glob("*.json"))]
    return [e for e in entries if script is None or e["script"] == script]
