# Conventions for the docs

The learning pages live in Notion; this repo holds their exercises. This file is what both
depend on, so a page and a script cannot drift apart.

**Notion is the source of truth.** Nothing here mirrors page prose — a second copy drifts, and
a stale one gets treated as authoritative by someone who mistakes it for the source. That has
already happened once, from a comment in `pixi.toml`.

**Titus edits the pages too.** A page that differs from what you expected has been *edited*,
not damaged. Cut and rewrite freely; the one gate is that a passage reading as his edit rather
than as earlier Claude prose gets named and confirmed first.

**Everything on a page is Claude's**, including retrieval questions and their answers — each
answer in a collapsed toggle, so the reader produces it cold and then expands to check.

**These are docs, not a book.** No page names itself or the collection. Refer to a topic by
name, never by number; renumbering used to force edits across every page.

**No count is ever a specification** — not lines in a listing, not sections, not one-of-each.
A count is checkable, so it gets satisfied literally at the expense of what it stood for.

## The controlling idea

> Training holds four kinds of state. Distributed training is the art of moving some of it off
> one device without changing the answer. Every technique is a bet about which rows you can
> afford to split; every bug is that bet failing silently.

## The ledger

For a model of N parameters under fp32 AdamW:

| State | Size | Born | Dies |
| --- | --- | --- | --- |
| Parameters | 4N | before step 1 | never |
| Gradients | 4N | during backward | at `zero_grad` |
| Optimizer state | 8N | on the *first* `step()` | never |
| Activations | proportional to batch x sequence x depth | during forward | consumed by backward |

Three rows scale with the model and one with the batch, which is where 16 bytes per parameter
comes from — and why activations set the per-device batch size, since data parallelism never
relieves them and each rank pays in full.

**Every page opens with this table, the row it touches marked**, and says what it does to that
row: changes its dtype, changes its size, delays its death, replicates it, splits it.

## Sharding

The multi-device topics are one table, a row at a time. FSDP is not a new concept; it is the
ledger with three rows sharded.

| | Parameters | Gradients | Optimizer state |
| --- | --- | --- | --- |
| DDP | replicated | all-reduced | replicated |
| ZeRO-1 | replicated | all-reduced | sharded |
| ZeRO-2 | replicated | sharded | sharded |
| ZeRO-3 / FSDP | sharded | sharded | sharded |

## Equivalence

Every technique claims to compute the same thing as something simpler, under conditions. Name
the claim and the conditions, every time. This table is what accelerate's test suite encodes,
which is the reason any of this is written down.

| Technique | Claims | Breaks when |
| --- | --- | --- |
| Mixed precision | equals fp32 within tolerance | reductions run in low precision; fp16 overflows |
| Accumulation | N micro-steps equal one N-times batch | token counts vary between micro-batches |
| DDP | W ranks of B equal one rank of W times B | loss reduction inconsistent; ragged batch counts |
| FSDP / ZeRO | equals DDP | gradients reduced in low precision; clipping computed per-shard |

## The topics, in order

A learning path. *The loop* is read first; everything after assumes it.

| Topic | Exercise |
| --- | --- |
| The loop | the fp32 baseline |
| The four kinds of state | predict peak memory, then profile the baseline |
| Numerics | mixed precision; TF32 |
| The optimizer | optimizer swap; gradient clipping |
| The data path | dataloader variations |
| Gradient accumulation | accumulation |
| Collectives and topology | a collectives probe |
| Data parallelism | DDP |
| Sharding the ledger | FSDP |
| DeepSpeed | DeepSpeed |
| Checkpoint and resume | checkpoint, kill, resume, compare |
| Proving equivalence | the test suite itself |

Gradient accumulation is the hinge, taught as the **single-device rehearsal for distributed
training**: same arithmetic, same failure mode, same test, no network to hide behind. The
multi-device topics are that one with a wire in the middle. Checkpointing comes last because
those topics sharded the state it has to save.

Reference pages: measurement protocol, glossary, question bank, and an accelerate source map
from concept to file and line.

## What a page carries

A prediction the reader can make before running anything. An exercise that produces the number.
An equivalence claim with its failure conditions named. The silent failures its own material
creates. A hole at the end that a later topic fills. Plus retrieval questions, and an
*Interrogate this section* fold per major section.

This is a list to include, not a set of quotas — demanding one of each once manufactured an
equivalence claim for a page that had none.

A retrieval question earns its place when getting it wrong would cost the reader something
later. Prefer "what did you measure, and what is missing from it?" over "what is X?"

**The one test that can fail a whole page:** state its argument in a sentence, delete every
heading, and check that the argument still holds and that every section serves it.

## Depth

**A page may use a concept only as deeply as its own exercise requires.** Deeper is a link, not
a paragraph. Applied per paragraph: *which exercise makes the reader feel this?* If the answer
is an exercise several topics away, move the paragraph there. The first loop page was a
compressed edition of the five topics after it, which made it exhausting and unlearnable at once.

The same discipline for *conditions*: **do not qualify a true claim against a configuration the
reader will not use.** "Under `set_to_none=True`" attached to a default hedges a correct
sentence for a branch nobody takes. Ask whether a reader following this page could ever observe
the exception; if not, it belongs on the page whose subject it is.

## Numbers

**Analytic** — derivable from shapes, dtypes and arithmetic. `ln(V)`, 16 bytes per parameter, a
step costing about three forward passes. These generalize and never expire.

**Worked instance** — the same arithmetic on one concrete config, so the abstraction lands. The
convention is SmolLM2-135M, declared once on the parent page. `ln(49152) = 10.8027` is this, not
a measurement. Worked numbers come from the *released* config and never from this repo's code:
`initializer_range` reached the pages as 0.02 when the config says 0.041666..., which
`docs/_wiki_build/verify_facts.py` now catches.

**Specimen measurement** — one run on one machine. These live in `bench/results/`, not in the
text. Hand the measurement to the reader instead: *"your initial loss should sit just above
ln(V) — run it and see."*

Figures marked *schematic* carry no data claim. The full predict-measure-explain demonstration
is the `ln(V)` check on the loop page; later pages predict and measure without re-teaching it.

## Prose

**These serve the argument. Where a rule and the argument conflict, the rule is wrong — change
this file and say so.** Each exists because it caught a real defect, and each can be satisfied
mechanically while making the page worse.

The anchor is *Designing Data-Intensive Applications*: a competent engineer reading about an
unfamiliar system, taught through mechanisms, tradeoffs and failure modes. No hand-holding, no
personality, no first person. Never "we".

### Match these

An analogy, cashed out in the same clause and then dropped:

> Read it as *surprise*: confident and right scores near 0, confident and wrong scores
> arbitrarily high.

One sentence carrying a whole insight:

> The feature and the footgun are the same line of code.

A claim, rather than a description of a claim:

> A falling loss is evidence that the plumbing runs, not that it is correct.

A rule the reader can carry away:

> Averages of averages are not averages.

**The failure register** — second person, present tense, ordered events, closing on what the
reader will wrongly blame. Use this shape whenever a silent failure is described:

> Nothing errors and the training is correct — you simply need a gigabyte you should not, and
> the first thing you will doubt is the memory arithmetic.

### Recognize these

**Stripped connectives.** Every sentence true, the argument gone:

> A logit is an unnormalized score. Softmax turns logits into probabilities. Cross-entropy is
> the negative log of the probability given to the correct token. At initialization the weights
> are random. The distribution is near-uniform. The loss is ln(V).

Nothing states which claim depends on which, so the reader reconstructs the reasoning that was
deleted. This is how simple sentences become unreadable. Connectives — *so, which is why, but,
the consequence is* — carry the argument and are not clutter.

**Metadiscourse**, which is a different thing and does go:

> In this section we will cover the loss function. It is worth noting that...

Talk about the document rather than the subject. When you cannot tell which one a phrase is,
it is the argument: keep it.

**Unquantified comparison:**

> Cheaper, one fewer reduction, and empirically just as stable.

Three comparatives, no axis, no quantity, nothing a reader could check.

### Ask these of a draft

- Which sentence is this paragraph's claim, and does it stand out from the ones supporting it?
- Can a reader tell how each sentence relates to the one before it?
- Read it aloud. Is it staccato? That is audible when it is not visible.
- Does this hedge say *when* the exception applies? If not, cut it.
- Is this thing called by the same name it had four paragraphs ago?

## Out of scope

- **The architecture tour** — RoPE, SwiGLU, grouped-query attention, RMSNorm. The model is a
  black box. Two facts about it matter downstream and are stated where they are needed:
  reductions resist low precision (*Numerics*), and the logits tensor scales with vocabulary
  rather than hidden size (*The four kinds of state*).
- **Initialization detail** beyond one clause: the weights are random, so the model is
  maximally unsure.
- **Gradient checkpointing**, beyond one line in the ledger naming it as the trade.
