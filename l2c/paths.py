"""Where benchmark output lands.

The root is the directory containing `pyproject.toml`, found by walking up from this
file. That is independent of the working directory, which matters because a profiled
script may be launched from the repo root or from its own directory.

Resolving against `Path.cwd()` instead would silently write a second `bench/` under
whichever directory the process started in, splitting a comparison across two places
with no error. Failing to find the root is loud; writing to the wrong root is not.
"""

import os
from pathlib import Path


def _find_root() -> Path:
    if override := os.environ.get("L2C_ROOT"):
        return Path(override).resolve()
    for directory in Path(__file__).resolve().parents:
        if (directory / "pyproject.toml").is_file():
            return directory
    return Path.cwd()


ROOT = _find_root()

#: One JSON file per run, plus the figures generated from them.
BENCH_DIR = Path(os.environ.get("L2C_BENCH") or ROOT / "bench")


def results_dir() -> Path:
    path = BENCH_DIR / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir() -> Path:
    path = BENCH_DIR / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path
