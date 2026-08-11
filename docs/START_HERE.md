# Start here

Read `CLAUDE.md` and `PROFILING.md` first. Read `docs/CONVENTIONS.md` as well if the task touches
the textbook. Nothing else in `docs/`. This file tells you what state the repo is in and what
to do next.

## What this is

Titus is learning distributed training to onboard onto HuggingFace accelerate. Two products:
the **textbook**, which is the Notion wiki, and the **exercises**, which are this repo. He
writes the training loops by hand; you write the chapters, the profiling that measures the
loops, and nothing in between. `CLAUDE.md` has the division of labour and it is not the usual
one — **you do not write training loops here.**

## State

Restructured on 2026-08-07 after a code review by Marc Sun. Clean and green.

- `scripts/` — Titus is writing the baseline. Untracked work in progress; leave it alone
  unless asked, and never stage his paths.
- `l2c/` — profiling harness, pruned to measurement only. 21 tests, plus one per study
  script: `test_boundary.py` parametrizes over `scripts/*.py` to enforce that each imports
  nothing from this repo.
- `bench/` — empty. Results and figures land here.
- `docs/CONVENTIONS.md` — the textbook's design specification, added 2026-08-10. The controlling
  idea, the four-row ledger every chapter modifies, the twelve-chapter structure (0 through 11), the chapter
  contract, the numbers policy, the binding prose style.
- `docs/_wiki_build/` — verifiers and figure generators for the wiki. Imports
  `l2c.common.model`, so that module must keep working. No longer a separate effort; its
  README carries the wiki editing rules.
- `.claude/skills/` — `explain`, `write-chapter`, `refine-chapter`. A *new* skill directory is
  discovered live, but an *edit* to an existing `SKILL.md` serves from a cached payload until
  restart — read the file from disk if you need to be sure which version is in force.

## Next, in order

1. **Titus hand-writes the baseline.** A plain fp32 single-GPU training loop, under ~50
   lines, self-contained. If he asks for help, act as a documentation lookup via `explain` —
   name the torch APIs, describe the shape, link the docs, read the config values for him.
   Do not write it for him and never fill in a skeleton left blank.

2. **Build the skill** at `.claude/skills/profile-script/SKILL.md`. It takes a
   `scripts/NN_topic.py` and produces `scripts/NN_topic_profiled.py` following
   `PROFILING.md` exactly, runs it, and writes JSON to `bench/results/` plus a figure to
   `bench/figures/`. The twin is regenerated, never hand-edited.

   Put the profiled-twin template in the skill directory beside `SKILL.md`, so generating one
   is filling in slots rather than following prose from memory — that is what keeps the
   measurement identical across topics.

3. **Run it on the baseline** and check the signals in `PROFILING.md`: parameter count,
   initial loss against `ln(V)`, loss decreasing, peak memory against the prediction.

Then topics, one per script, each a copy-and-modify of the baseline: mixed precision,
gradient accumulation, optimizer swap, dataloader variations, gradient clipping, TF32.
One modification per script — do not stack them.

## The docs, in parallel

The pages predate `docs/CONVENTIONS.md` and most were written under a weaker organizing idea.
They are documentation Titus learns from, not a book: **no page names itself or the collection**,
and cross-references use topic names, never numbers. Numbered titles are being dropped as each
page is rewritten; the reading order lives on the parent page.

*The loop* has been rewritten and is the model to follow: the ledger opens it with the touched
row marked, the argument is stated once, and every cross-reference is a topic name.

**Next is *The four kinds of state*.** It carries the ledger every later page modifies, and it
takes the material the loop page deliberately defers — activation arithmetic, peak memory,
where the optimizer state appears.

Authoring is one page at a time with `write-chapter`: state the argument in a sentence, sort the
existing page against it, then write into whichever document has more surviving material. An
earlier edition of that skill forbade reading the old page first; that rule is gone, and why is
recorded in the skill.

Renumbering means the live pages and the spec disagree by one until migration catches up. When
citing a chapter, use the spec's number and say so if the live page differs.

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

## Decided

- **Random init**, via `AutoConfig.from_pretrained` then `AutoModelForCausalLM.from_config`.
  A pretrained checkpoint would forfeit the free `ln(V)` check.

## Open, for Titus to decide when he writes the baseline

- Whether the loop shows forward and score as two lines (`logits = model(...)`, then
  `loss = loss_fn(logits, labels)`) or one (`loss = model(..., labels=ids).loss`). Two lines
  is more instructive; one line is what most people write and avoids the live reference that
  inflates peak memory. A third option gets both: two lines plus `del logits` before
  `backward()`, which makes the cost explicit instead of accidental. `PROFILING.md` covers
  the trap, and note that `out = model(ids, labels=ids)` then `out.loss` leaks exactly as the
  two-line form does.
- Whether the synthetic batch is fixed or redrawn each step. This decides whether the loss
  curve carries information: on freshly random tokens the task is unlearnable and the loss
  sits flat at `ln(V)` forever, so a correct implementation and a broken one look identical.
  A fixed batch is memorized, the loss falls unmistakably, and "loss decreases" becomes a
  real signal again.
