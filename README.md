# loop-to-cluster

Learning distributed training one concept at a time, from a single-GPU training loop to a sharded
multi-GPU one — predicting the numbers before measuring them.

This is a set of study scripts, not a codebase. Each one is a complete, runnable record of one
concept, written to be read top to bottom.

```text
scripts/               study scripts. self-contained, hand-written
  00-basic-loop.py       trains and prints the loss
snapshot.py            records allocation history for any script, for pytorch.org/memory_viz
notebooks/             interactive work; where off-path experiments live
l2c/                   measurement helpers. only tooling imports them, never a study script
  harness/               timing, the memory staircase, the autograd ledger, predictions
  common/model.py        model construction, used by figures/
figures/               generators and verifiers for the figures posted to Notion
  out/                   rendered PNGs (tracked)
bench/                 generated: run output and memory snapshots
tests/                 harness arithmetic, and the script-boundary rule
CLAUDE.md              how to work in this repo
DECISIONS.md           choices and their reasons; what not to bring back
```

The notes live in Notion. Nothing here mirrors them.

## The one rule

> **A study script imports nothing from this repo.**

Duplication between scripts is correct. Shared helpers make experiments interfere: change a
`build_loader` for the mixed-precision run and you have silently changed the baseline it is being
compared against. Self-contained scripts can be modified without touching each other.

## Install

```bash
pixi install
CUDA_VISIBLE_DEVICES=0 pixi run python -c 'import torch; print(torch.cuda.is_available())'
```

Always `CUDA_VISIBLE_DEVICES=0`; GPU 1 drives a display. Always `cd` into this directory in the
same command as `pixi run` — see `CLAUDE.md`.

## Run the tests

```bash
CUDA_VISIBLE_DEVICES=0 pixi run pytest tests/ -q
```
