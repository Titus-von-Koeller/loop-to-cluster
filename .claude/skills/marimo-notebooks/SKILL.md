---
name: marimo-notebooks
description: Judgment for working on the marimo notebooks under notebooks/ — stated as owner-held intents with current, disposable expressions, evolving by incident and consent. Use when creating, reviewing, or editing notebook content. Mechanics (file format, folding, sync hazards, checks) live in CLAUDE.md under "Editing notebooks".
---

# Working on the marimo notebooks

## What this is for

A notebook's life here has two phases, in order. **First, polish**: he aims a session
at a found notebook — an upstream tutorial worth keeping — and it becomes a coherent
teaching document, by the standards below, in as many passes as that takes, each pass
checked on the rendered surface a reader actually sees (N6). **Then, the loop**: he
studies the polished notebook and asks; the session explains, in conversation; what the
exchange reveals as a gap is folded back into the notebook where its narrative needs
it, and the file stays one coherent argument in the session's care for as long as he
learns from it. The conversation is the instrument that finds gaps; the notebook is the
surface where explanations land and persist. Machinery that would stand in for the
conversation — logged predictions, telemetry, autonomous checking — is out of scope
unless Titus asks for it.

How to read this file: every entry is an **intent** — the invariant why, which changes
only with Titus's consent — and an **expression**, today's way of serving it, which any
session may re-derive when world and expression disagree, saying so in chat and in the
commit. The review question is always the intent. Satisfying an expression while missing
its intent is a failure even when every named artifact exists: artifacts make gaming
effortful, not impossible. When you notice the pull to produce an artifact instead of
the judgment it stands for, that is itself an incident — say so and record it.

The technique shelf — hypotheses about what makes these notebooks teach, with provenance
from this repo's practice, marimo-team/learn, and the explorable-explanations canon —
is `pedagogy.md` beside this file; it evolves under the same rules.

Evolution: incidents accrue as dated lines under the entry they test; they are the
evidence the next expression is derived from. Expressions may be re-derived freely
within their intent. Intents change only by proposal to Titus — and proposing one that
has outgrown its statement is part of the job, not an overstep. Prefer deleting a stale
expression to patching it; never touch an intent without his go.

This file is a floor, not a ceiling. Where you genuinely know a better move, make the
disagreement visible — name the entry it contradicts, do better at expression level,
propose at intent level. And when the work reveals something Titus would want to know
that he did not ask about, say it: teaching unasked is in scope; silence is not.

## N1 · The file teaches

**Intent**: a reader one step behind the material — Titus a few months from now, or
someone like him — meets a coherent argument in which every concept is defined at first
contact or carries a declared forward reference. What that reader stumbles on is a gap;
questions during work locate gaps, and the fix addresses the gap, never the questioner.
**Expression**: before adding, ask what the file already promises (redeeming a standing
promise is the strongest reason to add), what the addition displaces (whole-file read;
longer-but-not-tighter fails even when correct), and where each concept it uses first
appears. These are ways of looking, not gates: an addition the intent clearly wants
does not die for lacking a quotable promise, and producing a quote does not save one
the intent does not want. Upstream tutorial prose stays intact; additions are declared.
**Provenance**: 02's argument — a tensor is a shape, a stride and an offset over one
flat run of memory — was discovered in the file, not imposed; sections that did not
serve it were cut.

## N2 · Displayed truth is executed truth

**Intent**: a reader can trust every shape, stride, equality and error message in the
notebook because it ran, in this environment — nothing is transcribed from belief.
**Expression**: execute claims in this repo's env in the session that writes them;
prefer self-identifying values (`arange`, digit-address tensors) so pictures trace
themselves; name traps exactly. A claim later falsified is corrected at the source and
named as a correction in its own commit — an overclaim discovered is a commit, not an
embarrassment.
**Incidents**: 2026-08-31 — confident source-reading was falsified twice in one day by
single observations (the fold pass, float32 `0.1+0.2==0.3`); execution outranks reading.

## N3 · The document never shows the seam

**Intent**: artifacts stand alone for a reader who saw no conversation; understanding
is Titus's, written by him, in Notion.
**Expression**: prose references the document, never the exchange that produced it —
motivation from a question is good, question-answering register is a leak. Do not
write his notes.
**Incidents**: 2026-08-31 — "so here is what the flag means" shipped in document prose;
the audit found the register, not any literal reference, was the leak.

## N4 · The notebooks are his; shared state is announced

**Intent**: he owns `notebooks/`; an editor he has open is shared mutable state, and a
surprise write costs trust and can cost work.
**Expression**: propose with the concrete text or code and wait for the go; the go
covers the arc — edit, verify, stage the named paths, commit. Verified work does not
wait for a second invitation; unverified work does not ride the first. Announce before
writing to a possibly-open notebook and end with "done — ctrl+alt+m" (what that averts
and recovers: CLAUDE.md, Editing notebooks).
**Incidents**: 2026-08-31 — two live-sync crashes from unannounced batch rewrites, the
second after the announce rule was already agreed; the keystroke is the repair.

## N5 · Observations outrank models

**Intent**: his falsifying observation beats your theory, including your reading of
source code — and deference is to evidence, not to deference: when his suspicion is
wrong, show the check that says so.
**Expression**: on a challenge to correctness or compliance, audit the whole change and
present judgment calls for overrule, rather than spot-fixing the named symptom. Step
back to the gist before mechanisms; say plainly what is and is not worth his time. Your
reading of his instruction is also a model — a hypothesis only he can test: when the
reading decides how far a deletion or change reaches, ask first; report a measurement
you cannot explain as the number and the question, never the number and a story.
**Incidents**: 2026-08-31 — "ctrl+alt+m changed nothing" overturned a fold model built
from minified source; the whole-change audit pattern was minted when a single leaked
sentence was challenged and the audit found exactly one.
2026-09-01, via a sibling session's handoff — "just remove it": the removal was his
call, its boundary silently the agent's; and a 33-vs-1 commit ratio narrated as
misallocation was in fact his deliberate investment in learning agentic engineering.
2026-09-01 — a "make a breakthrough" round produced calibration machinery for an
imagined anonymous reader; his one-paragraph correction named the actual system — a
two-agent loop where conversation finds the gaps — and outranked two research agents
and a finished proposal. Depth prompts set a floor; when the honest finding is modest,
machinery fills the gap unless caught.

## N6 · The polish pass converges, and is seen

**Intent**: a notebook he aims a session at gets better in passes until another pass
would not tighten it — and "better" is judged on the rendered surface a reader actually
sees, never on green checks alone.
**Expression**: the go on "polish this" covers the whole arc — repeated passes, each
reading the file whole, editing, verifying (gate, headless run), then *looking*: the
vscode-keyhole skill screenshots the real UI and reads live notebook state, and when
live rendering is needed, ask Titus to open the notebook. Each pass states what it
changed and what it learned; a learning that generalizes moves to pedagogy.md, so the
shelf compounds. Stop when a pass yields only preference-grade changes — report then,
rather than polishing the polish.
**Provenance**: granted 2026-09-01 — "improve the existing notebooks I aim you at in
multiple passes, checking your work with the peekhole … iteratively."
