# CLAUDE.md — loop-to-cluster

Titus is learning distributed training to onboard onto HuggingFace **accelerate**. The
method is *predict the numbers, then measure them, then explain the gap.* One concept per
script, from a single-GPU loop to a sharded multi-GPU one.

**The scripts are for reading, not for shipping.** Abstraction that would be a virtue in
library code is a defect here: it turns one linear read into a jump-and-return, and it
couples experiments that must be independently modifiable.

## Who writes what

| Titus | Claude |
| --- | --- |
| `scripts/NN_topic.py` — the training loop, by hand | `scripts/NN_topic_profiled.py` — the measured twin |
| each topic's modification to the loop | `l2c/` harness, plots, JSON, README, repo hygiene |

**If asked to write or complete a training loop, don't.** Hand-writing it is the exercise,
not an inefficiency to remove. Offer the documentation-lookup form instead: name the torch
APIs, link the docs, describe the shape. Never fill in a skeleton left deliberately blank.

Never one-shot a whole script. Work incrementally and explain each change.

## The structural rule

A script in `scripts/` imports nothing from this repo — not `l2c`, not another script.
Duplication between scripts is correct: it is what lets one change without disturbing the
baseline it is compared against. `tests/test_boundary.py` enforces this.

Generated `*_profiled.py` twins are exempt, and share `l2c/` deliberately — identical
measurement is the only thing that makes two topics comparable. See `PROFILING.md`.

## Depth ceiling

Explain at the level of the public torch API and its documented behaviour: what a call
does and what it costs, not how it is implemented.

**accelerate source is always in scope, and is the exception that matters.** Read it
before claiming anything about it — file and line, or do not say it. The same goes for
FSDP, DeepSpeed and NCCL semantics: cite the source or flag it as unverified.

## Environment

Two GPUs. **Always `CUDA_VISIBLE_DEVICES=0`** — GPU 1 drives a display, so its clocks move
with the compositor and its memory starts several GiB down.

**Always name the project directory in the same command:**

```bash
cd /home/titus/src/loop-to-cluster && pixi run python ...
```

Never a bare `pixi run` from a parent directory: manifest lookup walks *upward* and will
silently bind a different workspace rather than erroring. Do not rely on ambient direnv
activation either — it applies at shell-init, so an agent's cwd can reset between commands.
Both fail silently with a working-but-wrong environment. Use `direnv exec <abs-dir> <cmd>`
if you also need `nvcc` or nix's `libstdc++`.

Python 3.14, torch 2.13 (CUDA 13 wheels from PyPI), transformers 5.x, accelerate 1.14.
`accelerate` is installed but imported by nothing — it is there so claims about it can be
checked against source.

US spelling. Ruff-clean (`E,F,I,UP,B,SIM,RUF`, line length 95). Code documents itself and
never references the conversation that produced it.

`docs/_wiki_build/` is owned by a separate effort and imports `l2c.common.model`. Leave
both alone unless asked. Jupyter setup lives in `docs/JUPYTER.md`.
