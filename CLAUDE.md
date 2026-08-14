# CLAUDE.md — loop-to-cluster

Titus is learning distributed training to onboard onto HuggingFace **accelerate**. The method
is *predict the numbers, then measure them, then explain the gap.* One concept per script,
from a single-GPU loop to a sharded multi-GPU one.

**The scripts are for reading, not for shipping.** Abstraction that would be a virtue in
library code is a defect here: it turns one linear read into a jump-and-return, and it couples
experiments that must be independently modifiable.

## Who writes what

| Titus | Claude |
| --- | --- |
| the training loops, by hand | the measured twins (`PROFILING.md`) |
| each topic's modification to the loop | `l2c/`, the Notion pages (`docs/CONVENTIONS.md`), repo hygiene |

**If asked to write or complete a training loop, don't.** Hand-writing it is the exercise.
Name the torch APIs, describe the shape, give the number to expect — never fill in a skeleton
left deliberately blank. Never one-shot a whole script; work incrementally and explain each
change.

**Absorb the lookups, though.** Most of what slows a newcomer is retrieval rather than
concepts. Read config values rather than describing where they live, give exact signatures and
defaults, check the installed source when behaviour is in question, and fetch a documentation
page rather than recalling it. Naming an API and its cost is assistance; writing the statement
is the exercise.

## The structural rule

A script in `scripts/` imports nothing from this repo — not `l2c`, not another script.
Duplication between scripts is correct: it lets one change without disturbing the baseline it
is compared against. `tests/test_boundary.py` enforces this. One modification per script; do
not stack features.

Generated `*_profiled.py` twins are exempt and share `l2c/` deliberately, because identical
measurement is the only thing that makes two topics comparable.

## Depth ceiling

Explain at the level of the public torch API and its documented behaviour: what a call does
and what it costs, not how it is implemented.

**accelerate source is always in scope, and is the exception that matters.** Read it before
claiming anything about it — file and line, or do not say it. The same for FSDP, DeepSpeed and
NCCL semantics: cite the source or flag the claim as unverified.

## Environment

Two GPUs. **Always `CUDA_VISIBLE_DEVICES=0`** — GPU 1 drives a display, so its clocks move
with the compositor and its memory starts several GiB down.

**Always name the project directory in the same command:**

```bash
cd /home/titus/src/loop-to-cluster && pixi run python ...
```

A bare `pixi run` from a parent directory walks *upward* for a manifest and can bind a
different workspace. Ambient direnv activation is no safer, since an agent's cwd resets between
commands. Use `direnv exec <abs-dir> <cmd>` when you also need `nvcc` or nix's `libstdc++`.

Python 3.14, torch 2.13, transformers 5.x, accelerate 1.14. `accelerate` is imported by
nothing — it is installed so claims about it can be checked against source.

United States spelling. Ruff-clean (`E,F,I,UP,B,SIM,RUF`, line length 95). Expand every acronym
on first use. Code and prose document themselves and never reference the conversation that
produced them.

**Commit straight to `main`, never open a branch.** More than one agent works here at once, so
a branch only diverges from what the other is committing. Stage your own paths explicitly —
`git add <paths>`, never `git add -A` — or a whole-worktree commit sweeps another session's
in-flight work into yours.

## The docs

The learning pages are in Notion; this repo is their exercises. `docs/CONVENTIONS.md` is the
standard for both, and Notion is the source of truth — nothing here mirrors page text.
`docs/_wiki_build/` holds the verifiers and figure generators and imports `l2c.common.model`,
so that module must keep working.

Authoring uses `write-chapter` and `refine-chapter`; measurement uses `profile-script`
(`PROFILING.md`). Their units of success differ, and rules that are right for one are harmful
in the others.
