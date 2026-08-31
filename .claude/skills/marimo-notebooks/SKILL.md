---
name: marimo-notebooks
description: Intent and working preferences for the marimo notebooks under notebooks/ — what they are for, what makes an edit good, and how Titus wants the collaboration to run. Use when creating, reviewing, or editing notebook content. Mechanics (file format, folding, sync hazards, checks) live in CLAUDE.md under "Editing notebooks"; this skill is about judgment.
---

# Working on the marimo notebooks

## What a notebook here is

A teaching document with **one argument**, not a scratchpad and not a reference. 02's
argument is that a tensor is a shape, a stride and an offset over one flat run of memory;
every section either advances that argument or earns its place some other way it can name.
The upstream PyTorch tutorial's prose and code stay intact underneath; additions are either
self-contained Explore sections or a declared editorial layer. Before adding anything, ask
what the file's argument needs — not what the topic could include. Prune rather than
accrete; this file has had whole sections cut for not serving it.

The reader is a learner one step behind the current material — concretely, Titus a few
months from now, or someone like him. What he stumbled on, a reader will stumble on; a
question he needed answered is evidence the notebook has a gap at that spot. But the
document must never show the seam: it references itself, never the conversation that
produced it. Motivation from a question is good; question-and-answer register is a leak.

## What makes an edit good

- **Coherence over local fixes.** Read the whole file before and after. A concept must be
  defined at first contact, not two sections later. Forward promises get redeemed
  ("a later section is about nothing else"), and redemptions point back ("it settles a
  debt"). If an addition uses a concept the file hasn't taught yet, either move it, or
  flag the forward reference explicitly the way the file already does with stride.
- **Every displayed claim is executed, not transcribed.** Shapes, strides, equalities and
  error messages in prose come from running them in this repo's env, in the same session
  that writes them. When a claim later proves wrong, correct it at the source and say what
  the correction was — an overclaim discovered is a commit, not an embarrassment.
- **Concrete before abstract.** Materialize intermediate objects rather than describing
  them; one falsifying experiment beats a paragraph. Prefer self-identifying values —
  `arange`, digit-address tensors — so every slice and picture traces itself. Name traps
  precisely ("slices clamp, integers raise"), never vaguely ("be careful with indexing").
- **Visible code is narrative code.** A visible cell holds only what the reader is asked
  to read; display plumbing goes in hidden cells; the mutation demos keep rendering inline
  because snapshot timing *is* the demonstration. When display and content share a cell,
  split them.

## How Titus wants the collaboration to run

- The notebooks are his. Propose with the concrete text or code, get a go, then edit;
  ask before staging; once verified, commit without being reminded — leaving verified work
  uncommitted annoys him more than asking to commit does.
- Announce before writing to a notebook he may have open, and end with "done — ctrl+alt+m".
  The sync hazards and recovery are in the CLAUDE.md ledger.
- He pushes back with observations, not orders — "this doesn't make sense relative to what's
  above" means find the actual incoherence and fix its cause. Take his falsifying
  observations as data that outranks your model, including your reading of source code.
- Explanations to him: step back to the gist before mechanisms, be honest about what is
  and is not worth his time, and say plainly when something he suspects is wrong — with the
  check that shows it.
- Understanding belongs to him in Notion, in his own words; the repo gets artifacts and
  checkable claims. Do not write his notes.
