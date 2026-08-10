# The book — design specification

The textbook lives in Notion. This repo holds the exercises. This file holds the invariants
both depend on, so that a chapter and a script cannot drift apart.

Chapters are authored with the `write-chapter` skill and revised with `refine-chapter`.
Neither may re-derive what is written here; if an invariant is wrong, change this file first
and say so.

## Three modes, three skills

Confusing these is the most expensive mistake available. Each has a different unit of
success, and rules that are correct in one are harmful in the others.

| Mode | Skill | Unit of success | Register |
| --- | --- | --- | --- |
| Book writing | `write-chapter`, `refine-chapter` | a chapter a stranger could learn from in a year's time | general, durable, model-agnostic |
| Exercise assistance | `explain` | Titus unblocked without being handed the answer | immediate, narrow, stops early |
| Benchmarking | `profile-script` (see `PROFILING.md`) | a number that is comparable across topics | mechanical, specified, no prose |

**Book writing** optimizes for coverage, sequence and durability. It may be long. It never
mentions this repo's file names, never quotes a measurement taken here, and never addresses
the reader as someone mid-task.

**Exercise assistance** optimizes for the next twenty minutes. It is allowed to be
incomplete and should be. It never writes a training loop; see `CLAUDE.md`.

**Benchmarking** produces the profiled twin and its outputs. It is a specification to
execute, not a document to write.

## The controlling idea

> Training holds four kinds of state. Distributed training is the art of moving some of it
> off one device without changing the answer. Every technique is a bet about which rows you
> can afford to split; every bug is that bet failing silently.

Every chapter is a modification to the ledger below, or a claim about equivalence, or both.
A chapter that is neither does not belong in this book.

## The ledger

For a model of N parameters under fp32 AdamW:

| State | Size | Born | Dies |
| --- | --- | --- | --- |
| Parameters | 4N | before step 1 | never |
| Gradients | 4N | during backward | at `zero_grad` |
| Optimizer state | 8N | on the *first* `step()` | never |
| Activations | proportional to batch x sequence x depth | during forward | consumed by backward |

Three rows scale with the model; one scales with the batch. That split is the subject of the
book, and it is where 16 bytes per parameter comes from.

The activations row is the one data parallelism never relieves — each rank holds its own
batch and pays in full. That is why activations set per-device batch size and why gradient
checkpointing exists as a compute-for-memory trade.

**Every chapter opens by showing this table with the row it touches marked.** The recurrence
is the book's primary orientation device; it does work no summary section can.

## The sharding table

Part III is this table, one row at a time:

| | Parameters | Gradients | Optimizer state |
| --- | --- | --- | --- |
| DDP | replicated | all-reduced | replicated |
| ZeRO-1 | replicated | all-reduced | sharded |
| ZeRO-2 | replicated | sharded | sharded |
| ZeRO-3 / FSDP | sharded | sharded | sharded |

FSDP is not a new concept. It is the ledger with three rows sharded.

## The equivalence spine

Every technique claims to compute the same thing as something simpler, under conditions.
Name the claim and the conditions, every time.

| Technique | Claims | Breaks when |
| --- | --- | --- |
| Mixed precision | equals fp32 within tolerance | reductions run in low precision; fp16 overflows |
| Accumulation | N micro-steps equal one N-times batch | token counts vary between micro-batches |
| DDP | W ranks of B equal one rank of W times B | loss reduction inconsistent; ragged batch counts |
| FSDP / ZeRO | equals DDP | gradients reduced in low precision; clipping computed per-shard |

This table is what accelerate's test suite encodes. It is the reason this book exists.

## Structure

Four parts, twelve chapters. Each part states what the reader can do at its end.

**Part I — One device, everything replicated.** *You can predict a training run's cost
before you run it.*

| # | Chapter | Exercise |
| --- | --- | --- |
| 1 | The loop | the fp32 baseline |
| 2 | The four kinds of state | predict peak memory, then profile the baseline |
| 3 | Numerics | mixed precision (TF32 belongs here too) |
| 4 | The optimizer | optimizer swap; gradient clipping |
| 5 | The data path | dataloader variations |

**Part II — The step stops being atomic.** *You can decouple the batch you compute from the
batch you learn from.*

| # | Chapter | Exercise |
| --- | --- | --- |
| 6 | Gradient accumulation | accumulation |

Chapter 6 is the hinge of the book and should be taught as the **single-device rehearsal for
distributed training**: same arithmetic (sum, then divide), same failure mode (the wrong
denominator), same test (does N micro-steps equal one large batch), with no network to hide
behind. Part III is this chapter with a wire in the middle.

**Part III — More than one device.** *You can move state off a device and prove the answer
did not change.*

| # | Chapter | Exercise |
| --- | --- | --- |
| 7 | Collectives and topology | a collectives probe |
| 8 | Data parallelism (DDP) | DDP |
| 9 | Sharding the ledger (ZeRO, FSDP) | FSDP |
| 10 | DeepSpeed | DeepSpeed |

**Part IV — Making it trustworthy.** *You can review someone else's backend integration and
know whether to trust it.*

| # | Chapter | Exercise |
| --- | --- | --- |
| 11 | Checkpoint and resume | checkpoint, kill, resume, compare |
| 12 | Proving equivalence | the test suite itself |

Checkpointing is Part IV rather than Part I precisely because Part III sharded the state it
has to save.

**Appendices.** A · Measurement protocol. B · Glossary. C · Question bank. D · accelerate
source map — concept to file and line, the bridge from book to codebase.

Script numbering is Titus's to assign; chapters name their topic, not a filename.

## The chapter contract

Every chapter has exactly:

- **one prediction** the reader can make before running anything
- **one exercise** that produces the number
- **one equivalence claim**, with its failure conditions named
- **one silent failure** — the thing that looks right while being wrong
- **one hole at the end** that a later chapter fills, left open on purpose

A chapter missing any of these is unfinished. A chapter with two of any of them is two
chapters.

## The disclosure rule

**A chapter may use a concept only to the depth its own exercise requires. Anything deeper
is a forward reference, and forward references are links, not paragraphs.**

The test, applied per paragraph: *which exercise makes the reader feel this?* If the answer
is an exercise several chapters away, the paragraph is early — move it there.

This rule exists because the first edition of chapter 1 was a compressed edition of chapters
2 through 6, which made it simultaneously exhausting and unlearnable.

## Numbers

**Analytic quantities belong in the text.** Derivable from shapes, dtypes and arithmetic:
`ln(V)`, 16 bytes per parameter, backward is about twice forward so a step is about three
forward passes, a live logits reference costs one batch-by-sequence-by-vocabulary tensor.
These generalize and never expire.

**Specimen measurements do not belong in the text.** One model, one config, one afternoon. A
figure that needs "re-measure if your config differs" attached is reporting a run, not
teaching a concept. Hand the measurement to the reader instead: *"your initial loss should
sit just above ln(V) — run it and see."* Measured results live in `bench/results/`.

Exactly one full predict-measure-explain demonstration belongs in the book, at the `ln(V)`
check in chapter 1, as a worked instance of the method.

## Out of scope

- **The architecture tour.** RoPE, SwiGLU, grouped-query attention, RMSNorm versus
  LayerNorm. The model is a black box. Two facts about it matter downstream and are stated
  where they are needed: reductions resist low precision (chapter 3), and the logits tensor
  scales with vocabulary rather than hidden size (chapter 2).
- **Initialization detail.** Chapter 1 needs one clause — the weights are random, so the
  model is maximally unsure. `initializer_range` comparisons and depth-scaled residual
  initialization are architecture trivia.
- **Gradient checkpointing**, beyond one line in the ledger naming it as the trade. Deferred
  by the plan.
