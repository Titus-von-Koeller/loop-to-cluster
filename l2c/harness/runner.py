"""Run one configuration per subprocess, and cache the result on disk.

A comparison between precision modes is only valid if the arms cannot contaminate each
other, and inside one process they can. A fresh process guarantees a fresh caching
allocator, fresh cuBLAS workspaces and handles, and no autotuning state carried over
from a previous dtype. It also guarantees an untouched `max_memory_allocated`, which is
a high-water mark for the life of the process: measure two arms in one process and the
second inherits the first's peak.

Results are keyed by the configuration that produced them, so re-running a comparison to
reformat a table costs nothing and the numbers in a write-up cannot drift from the numbers
that were measured. Delete one JSON file to re-run one arm; pass `force=True` for all of
them.

This is also the shape that survives the move to multiple GPUs. `accelerate launch` and
`accelerate.launchers.notebook_launcher` both work by starting one process per rank,
because accelerate's state is per-process: `PartialState` and `AcceleratorState`
(accelerate/state.py) are singletons, so a single process cannot hold two distributed
configurations at once. Step 4 replaces this module's `subprocess.run` with `torchrun`
and the structure is unchanged.
"""

import importlib.util
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

from l2c.common import model as model_lib
from l2c.harness import cli, runs


def _load_step(script: Path) -> ModuleType:
    """Import a step module without running it, to read its parser and its name.

    A step guards `main()` behind `__name__ == "__main__"`, so importing it costs its
    imports and nothing else.
    """
    spec = importlib.util.spec_from_file_location(f"_step_{script.parent.name}", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import a step from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def destination(script: Path, flags: Sequence[str] = ()) -> Path:
    """Where `script` will write the result of running with `flags`.

    Computed by parsing the flags with the step's own parser, so defaults are filled in
    exactly as the worker will fill them. The driver and the worker therefore derive the
    same path from the same configuration, and naming a run one way here and another way
    there is not possible.
    """
    module = _load_step(script)
    args = module.build_parser().parse_args(list(flags))
    preset = model_lib.preset_for(args.num_layers)
    return runs.path_for(module.STEP, model_lib.preset_dict(preset), cli.run_args(args))


def run_worker(
    script: Path,
    flags: Sequence[str] = (),
    *,
    force: bool = False,
    quiet: bool = False,
) -> dict:
    """Run `script` in a fresh interpreter and return the result it recorded.

    The worker writes its own result, keyed by its configuration. That is what lets the
    same file be both a standalone step you can read top to bottom and an arm of a
    comparison, without a flag that exists only to serve the driver.
    """
    flags = list(flags)
    target = destination(script, flags)
    label = " ".join(flags) or "(defaults)"

    if target.exists() and not force:
        if not quiet:
            print(f"[runner] cached  {label}")
        return runs.load(target)

    if not quiet:
        print(f"[runner] running {label}")
    subprocess.run([sys.executable, str(script), *flags], check=True)
    if not target.exists():
        raise RuntimeError(
            f"{script.name} ran but wrote no result at {target.name}. The worker and the "
            "driver disagree about the configuration key."
        )
    return runs.load(target)
