# Start here

Read `CLAUDE.md` and `PROFILING.md`. Those two, and nothing else in `docs/`. This file
tells you what state the repo is in and what to do next.

## What this is

Titus is learning distributed training to onboard onto HuggingFace accelerate. He writes
small self-contained study scripts by hand; you write the profiling that measures them.
`CLAUDE.md` has the division of labour and it is not the usual one — **you do not write
training loops here.**

## State

The repo was restructured on 2026-08-07 after a code review by Marc Sun. It is clean and
green, and there is no work in progress.

- `scripts/` — empty except its README. The baseline does not exist yet.
- `l2c/` — profiling harness, pruned to measurement only. 21 tests pass.
- `bench/` — empty. Results and figures land here.
- `docs/_wiki_build/` — **owned by a separate effort. Do not touch.** It imports
  `l2c.common.model`, so that module must keep working.

## Next, in order

1. **Titus hand-writes `scripts/01_baseline.py`.** A plain fp32 single-GPU training loop,
   under ~50 lines, self-contained. If he asks for help, act as a documentation lookup —
   name the torch APIs, describe the shape, link the docs. Do not write it for him.

2. **Build the skill** at `.claude/skills/profile-script/SKILL.md` — project scope, so it
   is committed with the repo. It takes a `scripts/NN_topic.py` and produces
   `scripts/NN_topic_profiled.py` following `PROFILING.md` exactly, runs it, and writes
   JSON to `bench/results/` plus a figure to `bench/figures/`. The twin is regenerated,
   never hand-edited.

   A skill directory may hold supporting files beside `SKILL.md`. Put the profiled-twin
   template there, so generating one is filling in slots rather than following prose from
   memory — that is what keeps the measurement identical across topics.

   `.claude/skills/` does not exist yet. Claude Code watches an existing skills directory
   live, but creating the top-level one mid-session needs a restart before `/profile-script`
   is available.

3. **Run it on the baseline** and check the signals in `PROFILING.md`: parameter count,
   initial loss against `ln(V)`, loss decreasing, peak memory against the prediction.

Then topics, one per script, each a copy-and-modify of the baseline: mixed precision,
gradient accumulation, optimizer swap, dataloader variations, gradient clipping, TF32.
One modification per script — do not stack them.

## Do not resurrect

A previous design was replaced wholesale. If you find traces of it in git history or in a
stale comment, leave them there:

- `steps/stepN_*/` directories, `prediction.toml`, `NOTES.md` as a per-step lab notebook
- a shared `collect` / `publish` / row-spec reporting layer
- `build_model` / `build_loader` / `build_batches` / `training_step` helper wrappers
- a shared argparse module, or a subprocess runner that drives several arms at once
- `results.jsonl`

The reasoning is in `PROFILING.md` and `CLAUDE.md`. The short version: study scripts are
read linearly and modified independently, so shared helpers cost comprehension and let one
experiment perturb another.

## Open, for Titus to decide when he writes the baseline

- Whether the loop shows forward and score as two lines (`logits = model(...)`, then
  `loss = loss_fn(logits, labels)`) or one (`loss = model(..., labels=ids).loss`). The
  two-line form is more instructive; the one-line form is what most people write and
  avoids a live reference that inflates peak memory. `PROFILING.md` covers the trap.
- Whether the baseline uses a random-init model or a pretrained checkpoint. Both are fine;
  random init gives the free `ln(V)` check.
