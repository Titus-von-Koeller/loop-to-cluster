"""Where the lab writes its cache and its results.

The root is the directory containing `pyproject.toml`, found by walking up from this
file. That is independent of both the package layout and the working directory, which
matters because three different things launch this code from three different places:
`python steps/step1_training_loop/train.py` from the repo root, a notebook whose
kernel starts in `notebooks/`, and a subprocess spawned by a comparison driver.

Resolving against `Path.cwd()` instead silently writes a *second* `results.jsonl` under
whichever directory the process was launched from, splitting a sweep across two files
with no error and leaving the fitted line missing half its points. Failing to find the
root is loud; writing to the wrong root is not.

In accelerate the same concern is handled by asking rather than inferring:
`ProjectConfiguration(project_dir=...)` (accelerate/utils/dataclasses.py:916) takes
the directory explicitly, and `Accelerator` places checkpoints and tracker logs
under it. Inference is only safe here because this repo is the project.
"""

import os
from pathlib import Path


def _find_root() -> Path:
    if override := os.environ.get("L2C_ROOT"):
        return Path(override).resolve()
    for directory in Path(__file__).resolve().parents:
        if (directory / "pyproject.toml").is_file():
            return directory
    # Installed non-editable: there is no repo to find. The cwd is a guess, but an
    # explicit L2C_ROOT is the documented answer for that case.
    return Path.cwd()


ROOT = _find_root()

#: Tokenized corpus. Written once, then every run is offline and instant.
CACHE_DIR = Path(os.environ.get("L2C_CACHE") or ROOT / ".cache")

#: One JSON object per run, appended. The substrate for cross-step plots.
RESULTS_FILE = Path(os.environ.get("L2C_RESULTS") or ROOT / "results.jsonl")

#: Per-run JSON written by worker processes, keyed by their flags.
RUNS_DIR = Path(os.environ.get("L2C_RUNS") or ROOT / "runs")


def cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def results_file() -> Path:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    return RESULTS_FILE


def runs_dir() -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return RUNS_DIR
