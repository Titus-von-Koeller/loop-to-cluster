---
name: marimo-notebooks
description: Intent and working preferences for the marimo notebooks under notebooks/ — what they are for, what makes an edit good, and how Titus wants the collaboration to run. Use when creating, reviewing, or editing notebook content. Mechanics (file format, folding, sync hazards, checks) live in CLAUDE.md under "Editing notebooks"; this skill is about judgment.
---

# Working on the marimo notebooks

An entry here that no longer matches how Titus works is wrong — fix it or delete it, and
prefer deletion: every line taxes the attention of the agent reading it. Quoted phrases
below are evidence from one file, not templates for the next.

## What a notebook here is

A teaching document, not a scratchpad and not a reference. The upstream PyTorch tutorial's
prose and code stay intact underneath; additions are self-contained Explore sections or a
declared editorial layer, and the file's header comment says which. 02 found its one
argument — a tensor is a shape, a stride and an offset over one flat run of memory — and
every section now either advances it or can name what else it earns its place by. Whether
another notebook has an argument, and what it is, is discovered from that file, not
imposed by this one.

The reader is a learner one step behind the material: Titus a few months from now, or
someone like him. A question he needed answered is evidence of a gap at that spot — the
gap is the evidence, not the answer's wording. Write for a reader who never saw any
conversation; a passage addressed to *someone* rather than to that reader is a leak,
however useful its content.

## The bar for an addition

Three questions, in order; failing any one is a no. Each has an artifact, so a review can
check the questions were asked rather than take it on faith:

1. **Does the file already promise or claim it?** (Artifact: the quoted promise.)
   Redeeming a standing promise is the strongest reason to add — 02 claimed for two
   sections that `...` binds from the right at any rank before anything showed it. An
   addition with no anchor in the file belongs in the REPL or in Notion.
2. **What does it displace?** (Artifact: the whole-file read, before and after.) Sections
   have been cut for not serving the file; an addition that makes it longer but not
   tighter fails even when correct.
3. **Where does the reader first meet each concept it uses?** (Artifact: the location.)
   Define at first contact, or flag the forward reference explicitly, as the file already
   does with stride.

## What makes the content good

- Displayed claims are executed, never transcribed: every shape, stride, equality and
  error message in prose comes from a run in this repo's env in the session that writes
  it. A claim later found wrong is corrected at the source and named as a correction.
- Concrete before abstract: materialize intermediate objects; prefer self-identifying
  values (`arange`, digit-address tensors) so every picture traces itself; name traps
  exactly ("slices clamp, integers raise"), never vaguely.
- Cell visibility and all other mechanics: CLAUDE.md, "Editing notebooks".

## The collaboration loop

- Propose with the concrete text or code and wait for the go. The go covers the whole
  arc: edit, verify, stage the named paths, commit. Verified work does not sit waiting
  for a second invitation; unverified work does not get committed to satisfy this line.
- Announce before writing to a notebook he may have open; write; end with
  "done — ctrl+alt+m". The failure this averts, and the recovery that keystroke runs,
  are in CLAUDE.md under "Editing notebooks".
- Pushback arrives as observations, not orders. A falsifying observation outranks your
  model, including your reading of source code. When he questions compliance or
  correctness, answer with an audit of the whole change, judgment calls listed for
  overrule — not a spot fix.
- Step back to the gist before mechanisms. Say what is and is not worth his time; when
  his suspicion is wrong, show the check that says so.
- Understanding is his, written by him, in Notion. The repo gets artifacts and checkable
  claims; do not write his notes.
