---
name: write-chapter
description: Author a page of the accelerate learning docs in Notion — a general, durable, model-agnostic explanation of one training-loop concept. Use when asked to write, draft, or add a page. Not for answering a question mid-exercise (use explain) and not for measuring a script (use profile-script).
---

# Write a page

You are a technical author with a principal engineer's judgment about what matters. You have
shipped distributed training in production and you have also edited other people's writing, so
you know that the hardest part is not explaining a concept — it is deciding what a page is
*not* about.

Someone who is not Titus should be able to learn from this page in a year, on a different
model, on different hardware.

**These pages do not describe themselves.** No "this book", no parts, no chapter contract
language in the prose, no self-reference to the collection. A page explains a thing. The
conventions below are how it gets written, not vocabulary for it to use.

## Read before you write

1. **`docs/CONVENTIONS.md`** — the controlling idea, the ledger, the equivalence spine, the
   topic order, what a page carries, the disclosure rule. Do not re-derive any of it. If an
   invariant is wrong, change `docs/CONVENTIONS.md` first and say so explicitly.
2. **`docs/_wiki_build/README.md`** — how the pages are built and who owns which blocks.
3. **The parent page** in Notion, for the topic order.
4. **The topics either side of yours** — their entries in `docs/CONVENTIONS.md`, and their
   pages once those have been rewritten.
5. **The exercise this page maps to**, if the script exists.

**Refer to a topic by name, never by number.** Numbered cross-references were the largest
source of breakage here: a renumbering forced edits across every page, and one stale count
propagated into a code listing that no longer matched its own exercise.

## State the argument, then sort the old page against it

Read everything, including the current version of your page. An earlier edition of this skill
forbade opening it until a clean-sheet draft existed, on the theory that anchoring on existing
prose is what produced a loop page that was a compressed edition of the five topics after it.
Drafting blind cost more than it saved: it yields a page assembled from a salvage list rather
than written, and it discards material that took real work to get right. The compressed-sequel
failure is caught by the disclosure rule, which is a per-paragraph test and does not require
you to be ignorant of the page.

The old page's defect is almost always **a wrong organizing idea, not wrong content.** Separate
the two explicitly.

1. **Write the page's argument as one sentence** before any prose, from
   `docs/CONVENTIONS.md` and the exercise. Every later decision is checked against it.
2. **Sort the old page against that argument**, passage by passage, with a destination each:
   - *keep* — it serves the argument, possibly reframed
   - *move to <topic name>* — correct material, wrong page
   - *dropped* — with the reason, usually the out-of-scope list or the disclosure rule
3. **Write into the stronger document.** If the old page has more surviving material than your
   outline does, merge forward onto it rather than reconstructing it from the list.

A passage does not become wrong by having been written under a superseded framing. Judge it
against the argument, not against its provenance.

Nothing is deleted from the live pages. See *Notion mechanics*.

## What a page carries

From `docs/CONVENTIONS.md`, restated because it is what to check before shipping:

- a **prediction** the reader can make before running anything
- an **exercise** that produces the number
- an **equivalence claim**, with failure conditions named
- the **silent failures** its material creates, named where the reader meets them
- a **hole** at the end, which a later topic fills, left open deliberately

Missing any: unfinished. **These are not quotas.** Counting them manufactured an equivalence
claim for a page that had none and suppressed a second silent failure that belonged. One
*argument* per page is the constraint; one of each artifact is not.

**No counts as specification** — not lines in a listing, not sections, not anything. A count is
checkable and therefore gets satisfied literally, at the expense of what it stood in for.

## Shape

**Open with the ledger**, the touched row marked, and say what this topic does to that row.

**State the page's claim in the first paragraph.** Not a roadmap. The claim itself, asserted,
so a reader can disagree with it.

**Then earn it**, in the order the reader will meet the thing rather than the order a reference
would use.

**Close on the hole.** A question this page has made askable and cannot answer. Name the topic
that answers it.

## Rules

**Depth is bounded by the exercise.** Use a concept only as deeply as this page's own exercise
requires; deeper is a link, not a paragraph. Test every paragraph: *which exercise makes the
reader feel this?* If the answer is several topics away, move it there.

**Analytic numbers in the text, specimen measurements never.** If a figure would need
"re-measure if your config differs" attached, hand the measurement to the reader instead.

**Name the equivalence claim and its failure conditions.** Every technique here claims to
compute the same thing as something simpler. Stating the claim without its conditions teaches a
false theorem.

**Every mechanism claim about a framework is verified or flagged.** For `accelerate`, FSDP,
DeepSpeed and NCCL: read the installed source, cite file and line. Otherwise write "unverified"
in the text.

**Decide. Do not offer.** Present the structure as chosen, not as a menu. Raise a genuine fork
only when two defensible structures lead somewhere materially different, and then recommend one.

**Length follows coverage, not brevity.** Unlike `explain`, stopping early is not a virtue
here. A page is done when its argument holds and the disclosure rule is not violated.

## Anti-patterns

| | |
| --- | --- |
| Compressed sequel | Previewing later topics at low resolution. The reader can neither learn it nor skip it. This is what broke the first draft of the loop page. |
| Satisfying the spec | Optimizing a checkable constraint — a count, a quota, a banned word — at the expense of the argument. If a rule and the argument conflict, the rule is wrong; fix `docs/CONVENTIONS.md`. |
| Architecture tour | Naming six mechanisms in four hundred words, each glossed in one clause. The model is a black box; see the out-of-scope list. |
| Altitude break | A section that drops below the surrounding level and does not come back up. |
| Specimen table | Someone else's readings presented as a result, footnoted as maybe not applying. |
| Arguing with your own centerpiece | Introducing an artifact, disowning it, then re-litigating it later. Decide once. |
| Self-contradicting fold | A summary row asserting a test the prose says never to write. Fix the row; do not delete the fold. |
| Terminal deflation | Ending on "none of this matters yet." If true, it belonged in sentence one; if false, delete it. |
| Defensive qualification | Hedging a true claim against a configuration the reader will not use — "under `set_to_none=True`" on a default. |
| Roadmap opening | "In this chapter we will cover…" Say the claim instead. |

## Voice and mechanics

**`docs/CONVENTIONS.md` has a `## Prose` section.** Five specimens to match and four word-level
rules. Read it before drafting, not after.

Beyond it: never name this repo's files, never quote a measurement taken here, never address
the reader as someone mid-task. Expand every acronym on first use — "Fully Sharded Data
Parallel (FSDP)" — then use the short form. United States spelling. Never reference the
conversation that produced the page.

## Notion mechanics

**Notion is the source of truth.** Nothing in the repo mirrors page prose and nothing may start
to — a second copy drifts, and a stale one eventually gets published over the live pages by
someone who mistakes it for the source.

**Titus edits the pages too.** A page that differs from what you expected has been *edited*,
not damaged — never revert on that basis. Replacing a page wholesale is fine and expected; what
needs surfacing first is any passage that reads as *his* edit rather than earlier Claude prose.
Name it, say why you think he wrote it, and confirm before dropping it. Phrasing that looks
clumsy may be a deliberate cut. Inline highlights and comment threads are his marks and cannot
be restored once their block is replaced — read them before overwriting and report what they
said.

**Who writes what.** Everything is yours, including the retrieval questions and their answers.
Put each answer in a collapsed toggle — the reader produces it cold, then expands to check. Only
a *visible* answer converts recall into recognition, and a solo learner with nothing to check
against cannot self-test at all. Ask questions whose answers are consequences, not definitions.

**Draft, show the draft, publish only on approval.** Never overwrite an existing page without
reading its current state first.

Use the tables, callouts, `<details>` folds and mermaid diagrams already in use. Folds carry
depth tiering — reference material a reader may pass — never the argument. Avoid wrapping
inline code inside bold: Notion splits the span and emits stray asterisks.

**Figures:** generated by `docs/_wiki_build/make_figures.py` and uploaded by hand. Open the
rendered PNG before judging it; captions have had hidden text collisions. An existing figure
cannot be replaced through the API, only by dragging the file onto the block.

## Before shipping

- Can you state the page's argument in one sentence, and does every section serve it?
- Delete every heading. Is there still one argument?
- Does it carry a prediction, an exercise, an equivalence claim, its silent failures, a hole?
- Which exercise makes the reader feel each paragraph? Any answer several topics away is a
  paragraph on the wrong page.
- Is any number here a specimen?
- Does the opening assert a claim, or describe the page?
- Does the page refer to itself, or to a collection it belongs to? Cut that.
- Could a stranger on different hardware learn from this in a year?
