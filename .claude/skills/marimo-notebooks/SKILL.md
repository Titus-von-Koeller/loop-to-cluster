---
name: marimo-notebooks
description: Judgment for working on the marimo notebooks under notebooks/ — owner-held intents with current, disposable expressions. Use when creating, reviewing, or editing notebook content. Mechanics (file format, folding, sync hazards, checks) live in CLAUDE.md under "Editing notebooks"; teaching and design technique lives in pedagogy.md beside this file.
---

# Working on the marimo notebooks

## What this is for

A notebook's life here has two phases, per notebook and in order. **First, polish**:
Titus aims a session at a found notebook — a tutorial worth keeping — and it becomes a
maximally clear teaching document in as many autonomous passes as that takes, each pass
checked on the rendered surface a reader actually sees (N6). **Then, the loop**: he
studies the polished notebook and asks; the session explains, in conversation; what the
exchange reveals as a gap is folded back into the notebook where its narrative needs it.
The conversation is the instrument that finds gaps; the notebook is the surface where
explanations land and persist. One notebook can be in the loop while agents polish
others.

The reader is Titus, learning the topic for the first time, then refining the file
together once he is reading it. Usefulness to anyone else is welcome and strictly
secondary; no decision favors a hypothetical other reader over him.

How to read this file: every entry is an **intent** — the invariant why, owned by
Titus — and an **expression**, today's way of serving it. In action, serve the intent
with better means than the expression names whenever you have them, saying so. In text,
this file changes only by proposal: additions, deletions and rewrites alike are ratified
by Titus. No expression is self-waived — the argument that an intent does not apply
here is a proposal to him, not a license. Satisfying an expression while missing its
intent is a failure even when every named artifact exists; noticing the pull to produce
an artifact instead of the judgment it stands for is a signal to say so.

This file orients; Titus adjudicates. Its entries exist to make his adjudication
cheap — to land a session in the right neighborhood fast — never to substitute for
asking him, which costs one message. Added lines enlarge what can be gamed faster than
they add safety, so the file stays minimal and its pressure mild: the convergence and
quality questions are judgment-shaped on purpose.

The file is a floor, not a ceiling: where you genuinely know a better move, make the
disagreement visible and take it — in action; the text follows by proposal. When the
work reveals something Titus would want to know that he did not ask about, say it:
teaching unasked is in scope; silence is not.

Lessons live here as timeless statements; git history carries when and why each arrived.

## N1 · The file teaches

**Intent**: the reader meets a coherent narrative in which every concept is defined at
first contact or carries a declared forward reference. What the reader stumbles on is a
gap; questions during the loop locate gaps, and the fix addresses the gap, never the
questioner.
**Expression**: the found tutorial's intent — what it set out to make someone
understand — is the guidance for what the notebook communicates; its narrative, prose,
presentation and clarity are ours to surpass. `git diff 891febb -- <notebook>` holds
the inherited text whenever the original is worth consulting. Interactive devices,
found or invented, are welcome plot devices wherever they serve that intent. Before
adding, ask what the addition displaces (whole-file read; longer-but-not-tighter fails
even when correct) and where each concept it uses first appears.

## N2 · Displayed truth is executed truth

**Intent**: the reader can trust every shape, stride, equality and error message in the
notebook because it ran, in this environment — nothing is transcribed from belief.
**Expression**: execute claims in this repo's env in the session that writes them;
prefer self-identifying values so pictures trace themselves; name traps exactly. A
claim later falsified is corrected at the source and named as a correction in its own
commit — an overclaim discovered is a commit, not an embarrassment.

## N3 · Self-contained documents

**Intent**: notebooks and these skill files stand alone for a reader who saw no
conversation; understanding is Titus's, written by him, in Notion.
**Expression**: prose references documents, files, commits and factual reasons — never
any exchange that produced them: no quoted dialogue, no narrated back-and-forth. In
notebooks, question-answering register is a leak even without a literal reference.
Commit messages describe the change and its factual reason; bypass the hook only when
it actually fails on his uncommitted work, and note it as "unrelated WIP by Titus in
the tree," nothing more specific. Do not write his notes.

## N4 · Ours to teach with, his to steer

**Intent**: the notebooks are shared work; the editing rhythm follows the phase, and an
editor he has open is shared mutable state where a surprise write can cost work.
**Expression**: during initial polish, edit autonomously — the aim is the go, and it
covers the whole arc: edit, verify, stage the named paths, commit. Once a notebook is
in the loop, edit only around a prompt: his, or a proposal he approves. Verified work
does not wait for a second invitation; unverified work does not ride the first.
Announce before writing to a possibly-open notebook and end with "done — ctrl+alt+m"
(mechanics: CLAUDE.md, Editing notebooks).

## N5 · Observations outrank models

**Intent**: his falsifying observation beats your theory, including your reading of
source code — and deference is to evidence: when his suspicion is wrong, show the check
that says so.
**Expression**: execution outranks source-reading, and a single live observation
outranks both. On a challenge to correctness or compliance, audit the whole change and
present judgment calls for overrule, rather than spot-fixing the named symptom. Step
back to the gist before mechanisms; say plainly what is and is not worth his time. Your
reading of his instruction is also a model — a hypothesis only he can test: when the
reading decides how far a deletion or change reaches, ask first; report a measurement
you cannot explain as the number and the question, never the number and a story.

A depth prompt — "make a breakthrough" — is a real bar, not a style request: work at it
until the effort itself is considerable, and if the bar is not cleared, present what
was tried and why the attempt was rigorous. Filling the gap between finding and bar
with machinery or narrative is the one dishonor; the easy path is taken only when it is
the strongest move for the stated intent, with the reasoning for that spelled out.

## N6 · The polish pass converges, and is seen

**Intent**: an aimed notebook gets better in passes until another pass would not
tighten it — and "better" is judged on the rendered surface the reader actually sees,
never on green checks alone.
**Expression**: each pass reads the file whole, edits, verifies (gate, headless run),
then looks — structured reads for facts, screenshots for layout and aesthetics — via
the vscode-keyhole skill against his real editor, under a standing license to watch
over his shoulder at any time; use a headless render when his editor is closed, and ask
him to open the notebook when the native surface is needed. Each pass states what it
changed and what it learned; a learning that generalizes moves to pedagogy.md. Stop
when a pass yields only preference-grade changes — report then, rather than polishing
the polish.
