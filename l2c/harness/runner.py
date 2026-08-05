"""Run one configuration per subprocess, and cache the result on disk.

A comparison between precision modes is only valid if the arms cannot contaminate each
other, and inside one process they can. A fresh process guarantees a fresh caching
allocator, fresh cuBLAS workspaces and handles, and no autotuning state carried over
from a previous dtype. It also guarantees an untouched `max_memory_allocated`, which is
a high-water mark for the life of the process: measure two arms in one process and the
second inherits the first's peak.

Results are keyed by the flags that produced them, so re-running a comparison to reformat
a table costs nothing and the numbers in a write-up cannot drift from the numbers that
were measured. Delete one JSON file to re-run one arm; pass `force=True` for all of them.

This is also the shape that survives the move to multiple GPUs. `accelerate launch` and
`accelerate.launchers.notebook_launcher` both work by starting one process per rank,
because accelerate's state is per-process: `PartialState` and `AcceleratorState`
(accelerate/state.py) are singletons, so a single process cannot hold two distributed
configurations at once. Step 4 replaces this module's `subprocess.run` with `torchrun`
and the structure is unchanged.
"""

import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from l2c.paths import runs_dir

_UNSAFE = re.compile(r"[^A-Za-z0-9]+")


def _cache_path(script: Path, flags: Sequence[str]) -> Path:
    """A readable filename with a hash suffix, so cache files can be recognized."""
    digest = hashlib.sha256("\0".join([str(script), *flags]).encode()).hexdigest()[:10]
    slug = _UNSAFE.sub("-", "-".join(flags)).strip("-").lower() or "default"
    return runs_dir() / f"{script.parent.name}-{slug[:60]}-{digest}.json"


def run_worker(
    script: Path,
    flags: Sequence[str] = (),
    *,
    force: bool = False,
    quiet: bool = False,
) -> dict:
    """Run `script` in a fresh interpreter and return the JSON it wrote.

    The worker is expected to accept `--json-out PATH` and write its result there. That
    contract is what lets the same file be both a standalone step you can read top to
    bottom and an arm of a comparison.
    """
    flags = list(flags)
    destination = _cache_path(script, flags)

    if destination.exists() and not force:
        if not quiet:
            print(f"[runner] cached  {' '.join(flags) or '(defaults)'}")
        return json.loads(destination.read_text())

    if not quiet:
        print(f"[runner] running {' '.join(flags) or '(defaults)'}")
    subprocess.run(
        [sys.executable, str(script), *flags, "--json-out", str(destination)],
        check=True,
    )
    return json.loads(destination.read_text())
