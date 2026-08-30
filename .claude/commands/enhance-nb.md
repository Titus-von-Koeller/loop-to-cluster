---
description: Bring one notebook under notebooks/pytorch-basics/ to the standard 02-tensors.py sets, and check what the linters structurally cannot see.
argument-hint: 05 — a number, a filename, or one section within one
---

# Enhance $ARGUMENTS

## The standard is a file, not a description

`notebooks/pytorch-basics/02-tensors.py` has had this treatment in full and is what the others
are being brought to. Read it whole before a substantial pass: what is being matched is pacing
and the ratio of prose to picture, and neither is visible in a fragment. A narrow pass — one
chart, one palette — reads the matching cell of 02 and stops there.

The `show()` helper, the two ramps, `hide_code=True` on cells that only display, the
shape/dtype/stride caption: those live in that file and are deliberately not repeated here. A
second copy drifts from the first on the next edit, and then there are two standards.

## What that file cannot tell you

**The upstream prose is not yours to improve.** These are the official PyTorch tutorials as
converted, and commit `891febb` holds all eight exactly as they arrived — so the constraint is
checkable rather than remembered:

```bash
git diff 891febb -- notebooks/pytorch-basics/05-build-model.py
```

Additions read as new cells. A markdown cell outside an "Explore" heading that reads as a
*modification* is upstream text that has been edited, and it goes back. Searching for the
original text does not catch this: a paragraph with one helpful sentence appended still contains
the paragraph.

**marimo runs the dependency graph, not the page.** Position is presentation only, so reordering
for narrative is free and cannot break execution. Three consequences that ruff and
`marimo check` both pass:

- A cell's output is its last expression. Wrap it in a `for`, assign it, or add a line after it,
  and the cell renders nothing rather than erroring. One `mo.hstack` lost inside a loop looked
  like a broken notebook for an hour.
- marimo tracks reassignment, not mutation. A cell calling `t.add_(1)` on a tensor another cell
  owns is not idempotent: rerun it and the cell that drew the old value still shows it and is
  never marked stale. Keep a mutation inside the cell that creates what it mutates.
- Slices are views, so a before-and-after wants `_before = t.clone()` taken ahead of the write,
  or both panels draw the same tensor.

Every name without a leading underscore is reserved notebook-wide, hidden cells included.

## Color carries identity; lightness carries magnitude

Titus is red-green color blind, so a categorical series separated by hue alone can lose a pair.
`07-optimization-loop.py` still distinguishes three optimizer trajectories by red `#e45756`,
orange `#f2a154` and green `#54a24b` — the first and third are the pair that goes. Give a series
a second channel it does not need, a dash pattern or a direct label, and the hue becomes a
convenience rather than the only carrier.

Grepping for red is not that check, and will condemn working charts. A *continuous* diverging
scale is fine as it stands: `scheme="redblue"` in `05-build-model.py` and `06-autograd.py` ramps
lightness monotonically through its neutral midpoint, so magnitude survives with no hue
discrimination at all. What fails is hue standing alone as identity.

## The checks, and what each one cannot see

`.githooks/pre-commit` runs the first two and is the standard; run it rather than trusting a
description of it. Neither of those sees whether the notebook still executes:

```bash
cd /home/titus/src/loop-to-cluster
CUDA_VISIBLE_DEVICES=0 pixi run marimo export script notebooks/pytorch-basics/05-build-model.py > /tmp/nb-flat.py \
  && CUDA_VISIBLE_DEVICES=0 pixi run python /tmp/nb-flat.py
```

That flattens the graph into topological order and runs it, about five seconds for 02, nonzero
on any exception. Write the file and use `&&`: piping the export straight into `python -` reports
success when the export itself failed, because an empty standard input is a program that exits 0.

None of the three sees layout. A cell can run, lint, and still put its numbers outside their
squares or collide two labels — so render the thing and look at it before calling it done.

## Working conditions

Close the notebook in the editor first. A VSCode buffer open on the same file wins on its next
save and has silently reverted on-disk edits into an `app._unparsable_cell`, which is valid
Python and commits without complaint.

Read the file whole rather than splicing by line number; a line listing goes stale the moment an
edit lands above it, and index arithmetic against a stale one has corrupted this file twice.

`notebooks/` is Titus's — see **Commits** in `CLAUDE.md` before staging anything under it.
