"""Record where GPU memory goes during a run, for https://pytorch.org/memory_viz

    CUDA_VISIBLE_DEVICES=0 pixi run python snapshot.py scripts/00-basic-loop.py

The study script runs unmodified. Recording allocation history is studying the training
rather than training, so it lives here rather than in the script, and one runner works on
every script present and future.

Two things worth knowing before reading the picture. AdamW allocates `exp_avg` and
`exp_avg_sq` lazily on the *first* `step()`, so the optimizer's bands appear one step in
rather than at the start. And the flat bands are the states that never die -- parameters,
gradients until `zero_grad`, optimizer moments -- while the sawtooth is activations, born
in forward and consumed by backward.
"""

import runpy
import sys
from pathlib import Path

import torch

MAX_ENTRIES = 100_000


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python snapshot.py <script>")

    script = Path(sys.argv[1]).resolve()
    if not script.is_file():
        sys.exit(f"no such script: {script}")
    if not torch.cuda.is_available():
        sys.exit("no CUDA device -- a snapshot of zero allocations looks like a measurement")

    destination = Path("bench/snapshots")
    destination.mkdir(parents=True, exist_ok=True)
    pickle = destination / f"{script.stem}.pickle"

    torch.cuda.memory._record_memory_history(max_entries=MAX_ENTRIES)
    sys.argv = [str(script)]
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        torch.cuda.memory._dump_snapshot(str(pickle))
        torch.cuda.memory._record_memory_history(enabled=None)

    print(f"\nwrote {pickle}  ->  open it at https://pytorch.org/memory_viz")


if __name__ == "__main__":
    main()
