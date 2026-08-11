# Conventions for the docs

The learning pages live in Notion. This repo holds the exercises. This file holds what both
depend on, so a page and a script cannot drift apart.

**Notion is the source of truth for the pages.** This file holds invariants, never prose. No
file in this repo mirrors page text and none may start to: a second copy drifts, and a stale
one eventually gets published over the live pages by someone who mistakes it for the source.
Verifiers and figures live in `docs/_wiki_build/`, whose README carries the editing rules.

**Titus edits the pages too.** A page that differs from what was expected has been *edited*,
not damaged. Cutting and rewriting are wanted; the gate is that a passage which reads as his
edit rather than as earlier Claude prose gets named and confirmed before removal.

**Who writes what.** Everything on a page is Claude's, including the retrieval questions and
their answers. An earlier edition left the questions for Titus to write, on the grounds that a
pre-written answer converts recall into recognition. That confused a *visible* answer with a
written one: **put the answer in a collapsed toggle.** The reader produces it cold, then expands
to check, which is the mechanism working as intended — and a solo learner with no answer to
check against cannot self-test at all.

## What these pages are, and are not

They are documentation Titus learns from on the way to maintaining `accelerate`. They are not
a book, and no page says they are — no "this book", no parts, no chapter contract language in
the prose, no self-reference to the collection at all. A page explains a thing; the reader
knows where they are.

**Refer to a topic by name, never by number.** Numbers in cross-references were the single
largest source of breakage here: renumbering forced edits across every page, and one stale
count in this file ("the six lines") propagated into a code listing that no longer matched its
own exercise. Reading order lives on the parent page, in one list. Reordering costs one edit.

**No counts of anything as a specification.** Not lines in a listing, not sections, not one of
each artifact. A count is checkable and therefore gets satisfied literally, at the expense of
the thing it was standing in for.

## Three modes, three skills

Confusing these is the most expensive mistake available. Each has a different unit of success,
and rules that are correct in one are harmful in the others.

| Mode | Skill | Unit of success | Register |
| --- | --- | --- | --- |
| Authoring | `write-chapter`, `refine-chapter` | a page a stranger could learn from in a year's time | general, durable, model-agnostic |
| Exercise assistance | `explain` | Titus unblocked without being handed the answer | immediate, narrow, stops early |
| Benchmarking | `profile-script` (see `PROFILING.md`) | a number comparable across topics | mechanical, specified, no prose |

Authoring optimizes for coverage, sequence and durability. It may be long. It never mentions
this repo's file names, never quotes a measurement taken here, and never addresses the reader
as someone mid-task.

Exercise assistance optimizes for the next twenty minutes. It is allowed to be incomplete and
should be. It never writes a training loop; see `CLAUDE.md`.

## The controlling idea

> Training holds four kinds of state. Distributed training is the art of moving some of it off
> one device without changing the answer. Every technique is a bet about which rows you can
> afford to split; every bug is that bet failing silently.

Every page is a modification to the ledger below, or a claim about equivalence, or both.

## The ledger

For a model of N parameters under fp32 AdamW:

| State | Size | Born | Dies |
| --- | --- | --- | --- |
| Parameters | 4N | before step 1 | never |
| Gradients | 4N | during backward | at `zero_grad` |
| Optimizer state | 8N | on the *first* `step()` | never |
| Activations | proportional to batch x sequence x depth | during forward | consumed by backward |

Three rows scale with the model; one scales with the batch. That split is the subject of
everything here, and it is where 16 bytes per parameter comes from.

The activations row is the one data parallelism never relieves — each rank holds its own batch
and pays in full. That is why activations set per-device batch size and why gradient
checkpointing exists as a compute-for-memory trade.

**Every page opens by showing this table with the row it touches marked**, and says what it
does to that row: changes its dtype, changes its size, delays its death, replicates it, splits
it. The recurrence is the reader's primary orientation and it does work no summary can.

## The sharding table

The multi-device topics are this table, one row at a time:

| | Parameters | Gradients | Optimizer state |
| --- | --- | --- | --- |
| DDP | replicated | all-reduced | replicated |
| ZeRO-1 | replicated | all-reduced | sharded |
| ZeRO-2 | replicated | sharded | sharded |
| ZeRO-3 / FSDP | sharded | sharded | sharded |

FSDP is not a new concept. It is the ledger with three rows sharded.

## The equivalence spine

Every technique claims to compute the same thing as something simpler, under conditions. Name
the claim and the conditions, every time.

| Technique | Claims | Breaks when |
| --- | --- | --- |
| Mixed precision | equals fp32 within tolerance | reductions run in low precision; fp16 overflows |
| Accumulation | N micro-steps equal one N-times batch | token counts vary between micro-batches |
| DDP | W ranks of B equal one rank of W times B | loss reduction inconsistent; ragged batch counts |
| FSDP / ZeRO | equals DDP | gradients reduced in low precision; clipping computed per-shard |

This table is what accelerate's test suite encodes. It is the reason any of this is being
written down.

## The topics, in order

Order is a learning path, not a numbering scheme. Grouped by what the reader can do at the end
of each group.

**The loop.** Read first; everything else assumes it. Exercise: the fp32 baseline.

*One device, everything replicated — you can predict a run's cost before you run it.*

| Topic | Exercise |
| --- | --- |
| The four kinds of state | predict peak memory, then profile the baseline |
| Numerics | mixed precision; TF32 belongs here too |
| The optimizer | optimizer swap; gradient clipping |
| The data path | dataloader variations |

*The step stops being atomic — you can decouple the batch you compute from the batch you learn
from.*

| Topic | Exercise |
| --- | --- |
| Gradient accumulation | accumulation |

Gradient accumulation is the hinge, and is taught as the **single-device rehearsal for
distributed training**: same arithmetic (sum, then divide), same failure mode (the wrong
denominator), same test (does N micro-steps equal one large batch), with no network to hide
behind. The multi-device topics are this one with a wire in the middle.

*More than one device — you can move state off a device and prove the answer did not change.*

| Topic | Exercise |
| --- | --- |
| Collectives and topology | a collectives probe |
| Data parallelism (DDP) | DDP |
| Sharding the ledger (ZeRO, FSDP) | FSDP |
| DeepSpeed | DeepSpeed |

*Making it trustworthy — you can review someone else's backend integration and know whether to
trust it.*

| Topic | Exercise |
| --- | --- |
| Checkpoint and resume | checkpoint, kill, resume, compare |
| Proving equivalence | the test suite itself |

Checkpointing comes last rather than first precisely because the multi-device topics sharded
the state it has to save.

**Reference pages.** Measurement protocol. Glossary. Question bank. accelerate source map —
concept to file and line, the bridge from these pages to the codebase.

## What a page carries

- **a prediction** the reader can make before running anything
- **an exercise** that produces the number
- **an equivalence claim**, with its failure conditions named
- **the silent failures** its material creates, named where the reader meets them
- **a hole at the end** that a later topic fills, left open on purpose

A page missing any of these is unfinished. **This is a list of what to include, not a set of
quotas** — an earlier edition demanded exactly one of each, which manufactured an equivalence
claim for a page that had no natural one and suppressed a second silent failure that belonged.

Plus a *Retrieval practice* section — questions that can be answered cold, each with its answer
in a collapsed toggle — and an *Interrogate this section* fold per major section.

A retrieval question is worth asking when getting it wrong would cost the reader something
later. Prefer "what did you measure, and what is missing from it?" over "what is X?" —
recognition questions are the ones that feel productive and do nothing.

**The one test that can fail a whole page:** state its argument in a sentence, delete every
heading, and check that the argument still holds and that every section serves it. This test
outranks everything below.

## The disclosure rule

**A page may use a concept only to the depth its own exercise requires. Anything deeper is a
forward reference, and forward references are links, not paragraphs.**

Applied per paragraph: *which exercise makes the reader feel this?* If the answer is an
exercise several topics away, move the paragraph there. This exists because the first edition
of the loop page was a compressed edition of the five topics after it, which made it
simultaneously exhausting and unlearnable.

## No defensive qualification

The disclosure rule bounds *depth*. This one bounds *conditions*.

**Do not qualify a true claim against a configuration the reader will not use.** Gradients are
released at `zero_grad`, which is correct for anyone running the page's own code because
`set_to_none=True` is the default. Adding "under `set_to_none=True`" hedges a correct sentence
and costs a branch the reader will never take. The test: would a reader following this page ever
observe the exception? If not, it belongs on the page whose subject it is.

This is the completeness reflex wearing a rigor costume, and it is harder to catch than the
plain version because every sentence it adds is true.

## Numbers

Three categories. The first two belong in the text; the third does not.

**Analytic** — derivable from shapes, dtypes and arithmetic. `ln(V)`. 16 bytes per parameter.
Backward is about twice forward, so a step is about three forward passes. A live logits
reference costs one batch-by-sequence-by-vocabulary tensor. These generalize and never expire.

**Worked instance** — the same arithmetic on one concrete configuration, so the abstraction
lands. The convention is SmolLM2-135M, declared once on the parent page. `ln(49152) = 10.8027`
is a worked instance, not a measurement: a reader with a different vocabulary substitutes their
V and the sentence still holds. Every worked number comes from the *released* config, never from
this repo's code — a field that changes no shape is invisible to a parameter count, which is how
`initializer_range` reached the pages as 0.02 when the released config says 0.041666….
`docs/_wiki_build/verify_facts.py` exists to catch exactly that.

**Specimen measurement** — a reading from one run on one machine. `11.2744`. A peak-memory
figure in mebibytes. These are evidence that something was run, not understanding, and a figure
needing "re-measure if your config differs" attached is reporting a run rather than teaching a
concept. They live in `bench/results/`. In the text, hand the measurement to the reader instead:
*"your initial loss should sit just above ln(V) — run it and see."*

Figures marked *schematic* illustrate a shape and carry no data claim.

The full predict-measure-explain demonstration is the `ln(V)` check on the loop page. Later
pages predict and measure without re-teaching the method.

## Prose

The anchor is *Designing Data-Intensive Applications*: a competent engineer reading about an
unfamiliar system, taught through mechanisms, tradeoffs and failure modes, written to still be
true in five years. Not a tutorial, not a paper, not a blog post — no hand-holding, no density
for its own sake, no personality, no first person. Never "we"; there is no we.

Match these five, which are already right.

Analogy, cashed and dropped in one clause:

> Read it as *surprise*: confident and right scores near 0, confident and wrong scores
> arbitrarily high.

Compression — one sentence carrying the whole insight:

> The feature and the footgun are the same line of code.

A claim, not a description of a claim:

> A falling loss is evidence that the plumbing runs, not that it is correct.

A rule the reader carries away:

> Averages of averages are not averages.

**The failure register** — second person, present tense, ordered events, closing on what the
reader will wrongly blame. Use this shape every time a silent failure is described:

> Nothing errors and the training is correct — you simply need a gigabyte you should not, and
> the first thing you will doubt is the memory arithmetic.

Four word-level rules, because each caught a real defect:

- **Name a thing once and never vary it.** If it is the ledger, it is never "the state table"
  four paragraphs later. The reader cannot tell whether a new name means a new thing.
- **No unquantified comparatives.** "Cheaper", "faster", "more stable" are unfinished
  sentences. Attach the quantity, name the axis, or cut the claim.
- **Hedges must carry information.** "Typically" is permitted when the next clause says when
  the exception applies. "Somewhat", "arguably", "relatively" are not.
- **Delete metadiscourse.** "In this section", "it is worth noting". If it is important, its
  presence says so.

The anti-specimen, which every rule above exists to prevent:

> Cheaper, one fewer reduction, and empirically just as stable.

Three unquantified comparatives, no actor, no failure, nothing the reader can check.

## Out of scope

- **The architecture tour.** RoPE, SwiGLU, grouped-query attention, RMSNorm versus LayerNorm.
  The model is a black box. Two facts about it matter downstream and are stated where they are
  needed: reductions resist low precision (*Numerics*), and the logits tensor scales with
  vocabulary rather than hidden size (*The four kinds of state*).
- **Initialization detail.** The loop page needs one clause — the weights are random, so the
  model is maximally unsure. `initializer_range` comparisons and depth-scaled residual
  initialization are architecture trivia.
- **Gradient checkpointing**, beyond one line in the ledger naming it as the trade.
