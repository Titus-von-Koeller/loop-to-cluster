---
name: refine-chapter
description: Revise an existing learning page in a separate later pass — cut, move material to the page it belongs on, verify claims, and enforce the conventions. Use when asked to review, refine, tighten or fix a page. Not for authoring a new one (use write-chapter).
---

# Refine a page

The standard is `docs/CONVENTIONS.md`. You are enforcing it, not your preferences. This skill
is only the procedure.

A second pass, deliberately separate from authoring: an author defends structure, an editor
tests it, and one session does both badly. **Your main lever is removal and relocation, not
rewording** — most defects here are material that is correct, well written, and on the wrong
page.

**Cut freely, with one gate.** Before removing a passage, ask whether it reads as Titus's edit
rather than as earlier Claude prose — a deliberate trim, a sharpened sentence, an added aside.
If it does, name it, say why you think he wrote it, and confirm before removing that one. A
page that differs from what you expected has been edited, not damaged.

## Read first

`docs/CONVENTIONS.md`, the parent page for the topic order, the page under review in full, the
topics either side of it, and its exercise script. A page's worst defects live in its relation
to its neighbours and are invisible from inside it: the first loop page read as merely dense
until it was set against the topic order, where it turned out to contain the five topics
after it.

## The passes, in order

Run them separately — combining them lets the cheap fixes hide the structural ones.

**1. Placement.** For every paragraph: *which exercise makes the reader feel this?* An answer
several topics away means it is early; name its destination. This pass usually removes more
than all the others combined. Then check whether the page still carries what it should, or
whether removing borrowed material revealed it never had its own argument.

**2. Correctness.** Check every mechanism claim rather than skimming it. Three classes have
occurred here:

- *Imprecise mechanism.* "The scales start at 1.0, so normalization begins as an identity
  operation" — the learned affine is the identity; the normalization is fully active.
- *A fold contradicting its own prose.* One asserted that the initial-loss check catches a
  mis-shifted label beside prose calling that shift a silent failure. Both cannot be true.
  **Check every fold, table and callout against the text beside it** — they drift because they
  are written last and read first.
- *Defensive qualification.* A correct claim hedged against a configuration the exercise never
  reaches. Subtler than the others because every sentence it adds is true.

Run `docs/_wiki_build/verify_params.py` and `verify_facts.py` rather than re-deriving numbers
by hand.

**3. Numbers.** Delete the digits from each figure. If the sentence still teaches, the number
supported a general claim; if it collapses to something already known, it was performing rigor.

**4. Continuity.** Does the page hold one level throughout, or does a section drop below it and
fail to come back up? Where it uses the ledger, is the row it moves the row it claims?

**5. Prose.** Last and least, against `docs/CONVENTIONS.md`'s prose section — its diagnostic
questions, not a filter pass.

## Reporting

An ordered list of changes — cut, move, fix, verify — each with its reason and the invariant it
serves, most structural first. One misplaced section outweighs twenty prose nits. Every cut
names its destination; a cut with no destination is a loss. Where a defect admits two fixes,
recommend one. Edit the page in place only when asked.
