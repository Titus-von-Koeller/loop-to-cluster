# Conventions for the docs

The learning pages live in Notion; this repo holds their exercises. This file is what both
depend on, so a page and a script cannot drift apart.

**Who reads them.** Titus, and the next person onboarding onto accelerate. They have this repo,
this model and this machine. Worked numbers on SmolLM2-135M need no ceremony, and nothing here
has to survive a change of hardware.

**Notion is the source of truth.** Nothing in this repo mirrors page prose. A second copy
drifts, and a stale copy gets read as authoritative by whoever finds it first.

**Titus edits the pages too.** A page that differs from what you expected has been *edited*,
not damaged. Cut and rewrite freely; the one gate is that a passage reading as his edit rather
than as earlier Claude prose gets named and confirmed first.

**Everything on a page is Claude's**, including retrieval questions and their answers — each
answer in a collapsed toggle, so the reader produces it cold and then expands to check.

**Refer to a topic by name, never by number.** A number in a cross-reference breaks every other
page as soon as the order changes.

**Everything below serves the argument of the page being written. Where a rule and the argument
conflict, the rule is wrong: change this file and say so.** Nothing here is a quota, and no
count of anything — lines, sections, one-of-each — is ever a specification.

## The frame

The loop is where the model, the data, the optimizer and the loss meet. That makes it the page
to read first, and the thing every other page is a digression from: you study the dataloader,
the optimizer or the model by changing one of them and coming back to see what moved.

So a page answers *why would I go there*, not only *what changes*.

## The topics

| Topic | Exercise |
| --- | --- |
| The loop | the fp32 baseline |
| Memory and compute | predict peak memory, then sweep model size and check the prediction tracks |
| Mixed precision | autocast and the loss scaler |
| The optimizer | swap the optimizer; gradient clipping |
| The dataloader | dataloader variations |
| Gradient accumulation | accumulation |
| DDP | data parallelism across two ranks |
| FSDP | sharding the model states |
| DeepSpeed and ZeRO | the same bets, a different configuration surface |

Collectives are introduced where DDP first needs them, not as their own page.
Checkpoint and resume is a section of the FSDP page, because sharded state is what makes it
hard. Equivalence is a section of each technique page rather than a page of its own.

Gradient accumulation is the hinge, and is worth teaching as the **single-device rehearsal for
distributed training**: same arithmetic, same failure mode, same test, with no network to hide
behind. The multi-device pages are that one with a wire in the middle.

Reference pages: measurement protocol, glossary, question bank, and an accelerate source map
from concept to file and line.

## The ledger

A tool, not a mandate. Where memory is the subject, this is the organizing table, and a page
that moves one of its rows should show it and say which row.

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

The multi-device pages are the same table with rows moved off the device. FSDP is not a new
concept; it is the ledger with three rows sharded.

| | Parameters | Gradients | Optimizer state |
| --- | --- | --- | --- |
| DDP | replicated | all-reduced | replicated |
| ZeRO-1 | replicated | all-reduced | sharded |
| ZeRO-2 | replicated | sharded | sharded |
| ZeRO-3 / FSDP | sharded | sharded | sharded |

## Equivalence

Where a technique claims to compute the same thing as something simpler, name the claim and the
conditions under which it fails. Not every page has one; these four do, and this table is what
accelerate's test suite encodes.

| Technique | Claims | Breaks when |
| --- | --- | --- |
| Mixed precision | equals fp32 within tolerance | reductions run in low precision; fp16 overflows |
| Accumulation | N micro-steps equal one N-times batch | token counts vary between micro-batches |
| DDP | W ranks of B equal one rank of W times B | loss reduction inconsistent; ragged batch counts |
| FSDP / ZeRO | equals DDP | gradients reduced in low precision; clipping computed per-shard |

## What a page carries

A **prediction** the reader can make before running anything, and the **exercise** that produces
the number — that is the method the whole project runs on. Plus the **silent failures** its own
material creates, named where the reader meets them, because in training the number you watch
to confirm a run is working is the same number that survives the run being broken.

Plus retrieval questions, and an *Interrogate this section* fold per major section. A retrieval
question earns its place when getting it wrong would cost the reader something later — prefer
"what did you measure, and what is missing from it?" over "what is X?"

## Depth

**A page uses a concept only as deeply as its own exercise requires.** Deeper is a link, not a
paragraph. Per paragraph: *which exercise makes the reader feel this?* If the answer is an
exercise several topics away, move it there. A page that previews the topics after it is both
exhausting and unlearnable: too shallow to learn from, too present to skip.

Likewise for conditions: **do not qualify a true claim against a configuration the reader will
not use.** "Under `set_to_none=True`" attached to a default hedges a correct sentence for a
branch nobody takes.

## Numbers

**Analytic** — derivable from shapes, dtypes and arithmetic. `ln(V)`, 16 bytes per parameter, a
step costing about three forward passes. These carry the explanation.

**Worked instance** — the same arithmetic on SmolLM2-135M, so the abstraction lands. Worked
numbers come from the *released* config, never from this repo's code: a field that changes no
shape is invisible to a parameter count, so a preset can disagree with the released config
without any shape-based check failing. `docs/_wiki_build/verify_facts.py` diffs the two.

**Specimen measurement** — one run on one machine. These live in `bench/results/`, not in the
text. Hand the measurement to the reader instead: *"your initial loss should sit just above
ln(V) — run it and see."* Figures marked *schematic* carry no data claim.

## Prose

**The test that outranks the rest:** state the page's argument in one sentence, delete every
heading, and check that the argument still holds and that every section serves it.

Register: a competent engineer reading about an unfamiliar system, taught through mechanisms,
tradeoffs and failure modes. No first person.

**The failure register.** Second person, present tense, ordered events, closing on what the
reader will wrongly blame. Use this shape whenever a silent failure is described:

> Nothing errors and the training is correct — you simply need a gigabyte you should not, and
> the first thing you will doubt is the memory arithmetic.

**Keep the connectives.** Every sentence true, the argument gone:

> A logit is an unnormalized score. Softmax turns logits into probabilities. Cross-entropy is
> the negative log of the probability given to the correct token. At initialization the weights
> are random. The distribution is near-uniform. The loss is ln(V).

Nothing states which claim depends on which, so the reader rebuilds the reasoning that was
deleted. *So, which is why, but, the consequence is* — those carry the argument. Delete
sentences about the document; keep the words that join one claim to the next.

Ask of a draft:

- Which sentence is this paragraph's claim, and does it stand out from the ones supporting it?
- Can a reader tell how each sentence relates to the one before it?
- Does this hedge say *when* the exception applies? If not, cut it.
- Is this thing called by the same name it had four paragraphs ago?

## Out of scope

- **The architecture tour** — RoPE, SwiGLU, RMSNorm as mechanisms. The model is a black box
  *except where one of its shapes enters an arithmetic the reader performs*: grouped-query
  attention, because it is why the textbook parameter formula overshoots; vocabulary, because
  the logits term scales with it; and which operations are reductions, because those are the
  ones that resist low precision.
- **Initialization detail** beyond one clause: the weights are random, so the model is
  maximally unsure.
- **Gradient checkpointing**, beyond one line in the ledger naming it as the trade.
