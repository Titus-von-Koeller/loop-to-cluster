---
name: write-chapter
description: Author a page of the accelerate learning docs in Notion — an explanation of one training-loop concept, for Titus and the next person onboarding. Use when asked to write, draft, or add a page. Not for reviewing an existing one (use refine-chapter) and not for measuring a script (use profile-script).
---

# Write a page

The standard is `docs/CONVENTIONS.md`. Read it first and do not re-derive any of it; if an
invariant there is wrong, change that file and say so. This skill is only the procedure.

The reader is Titus, or the next person onboarding onto accelerate: someone with this repo and
this machine. The hardest part is not explaining the concept; it is deciding what the page is
*not* about.

## Read first

`docs/CONVENTIONS.md`, then the parent page in Notion for the topic order, then the topics
either side of yours, then the exercise script if it exists. Read the current version of your
page too.

Read the current page as well. Drafting blind to it yields a page assembled from a salvage list
rather than written, and discards material that took real work to get right. Anchoring is the
risk, and the disclosure rule handles it per paragraph — being ignorant of the page is not
required and does not help.

## Then

**Write the page's argument as one sentence** before any prose. Every later decision is checked
against it.

**Sort the existing page against that argument**, passage by passage, with a destination each:
*keep* (possibly reframed), *move to <topic name>*, or *dropped* with the reason. The old
page's defect is almost always a wrong organizing idea rather than wrong content — a passage
does not become wrong by having been written under a superseded framing.

**Write into whichever document has more surviving material.** If the old page has more than
your outline does, merge forward onto it rather than reconstructing it from the list.

## Rules

**Decide. Do not offer.** Present the structure as chosen, not as a menu. Raise a fork only
when two defensible structures lead somewhere materially different, and then recommend one.

**Length follows coverage.** Stopping early is not a virtue here. A page is done when its
argument holds and the disclosure rule is not violated.

## Anti-patterns

| | |
| --- | --- |
| Compressed sequel | Previewing later topics at low resolution. Neither learnable nor skippable. |
| Satisfying the spec | Optimizing a checkable constraint at the expense of the argument. If a rule and the argument conflict, fix the rule. |
| Altitude break | A section that drops below the surrounding level and does not come back up. |
| Arguing with your own centerpiece | Introducing an artifact, disowning it, re-litigating it later. Decide once. |
| Self-contradicting fold | A summary row asserting what the prose says never to do. Fix the row; keep the fold. |
| Terminal deflation | Ending on "none of this matters yet." If true it belonged in sentence one. |
| Roadmap opening | "In this page we will cover…" State the claim instead. |

Each points one way. Over-applied they produce their own inverse: a page with no forward
connections, one that oversells its relevance, one that drops the reader in with no orientation.
If avoiding an anti-pattern is making the page worse, you have hit the far wall — fix the rule.

## Notion mechanics

Draft, show the draft, publish only on approval. Never overwrite a page without reading its
current state. Use the tables, callouts, folds and mermaid diagrams already in use; folds carry
depth tiering, never the argument.

Figures come from `docs/_wiki_build/make_figures.py` and are uploaded by hand — open the
rendered PNG before judging it, and note that an existing figure can only be replaced through
the UI, not the API.
