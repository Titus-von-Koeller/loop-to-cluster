# loop-to-cluster

Learning distributed training one concept at a time, from a single-GPU training loop to a
sharded multi-GPU one — predicting the numbers before measuring them.

This is a set of study scripts, not a codebase. Each one is a complete, runnable record of
one concept, written to be read top to bottom.

```text
scripts/               study scripts. self-contained, hand-written, ~50 lines
  NN_topic.py            trains and prints the loss
  NN_topic_profiled.py   generated twin: same training, plus measurement
l2c/                   profiling harness. only the profiled twins import it
  harness/               timing, the memory staircase, the autograd ledger,
                         predictions, predicted-vs-measured tables, run JSON
  common/model.py        model construction, used by the docs tooling
bench/                 generated: results/*.json and figures/*.png (tracked)
tests/                 harness arithmetic, and the script-boundary rule
CLAUDE.md              how to work in this repo
PROFILING.md           the measurement contract every profiled twin follows
docs/CONVENTIONS.md    the standard for the Notion learning pages
docs/START_HERE.md     current state and decisions already taken
docs/_wiki_build/      verifiers and figure generators for those pages
```

## The one rule

> **A study script imports nothing from this repo.**

Duplication between scripts is correct. Shared helpers make experiments interfere: change
`build_loader` for the mixed-precision run and you have silently changed the baseline it is
being compared against. Self-contained scripts can be modified without touching each other.

The generated `*_profiled.py` twins are the exception and share the harness on purpose —
two topics are only comparable if both were measured identically.

One modification per script. Don't stack features.

## Install

```bash
pixi install
CUDA_VISIBLE_DEVICES=0 pixi run python -c 'import torch; print(torch.cuda.is_available())'
```

Always `CUDA_VISIBLE_DEVICES=0`; GPU 1 drives a display. Always `cd` into this directory in
the same command as `pixi run` — see `CLAUDE.md`.

## Run the tests

```bash
CUDA_VISIBLE_DEVICES=0 pixi run pytest tests/ -q
```
