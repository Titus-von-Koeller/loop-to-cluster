# A failed explanation and its rewrite

Prompt: *"The wiki mentions RMSNorm without giving context. What do I need to understand
about that?"*

The failure modes in `SKILL.md` are easier to recognize than to describe. This is a real
specimen.

## What went wrong

The first attempt ran to roughly 600 words under five headings — *Why normalize at all*,
*What it computes*, *What it drops relative to LayerNorm*, *Where it sits*, *What it costs*
— plus a four-row comparison table.

| Failure | Where |
| --- | --- |
| The taxonomy | Five parallel headings weighting every fact equally, so nothing was marked as the thing that matters. |
| Buried lede | The payload — normalization is a reduction, which is why autocast keeps it in fp32 — was the fourth bullet of the fifth section. |
| No number | "Cheaper", "negligible", "not negligible", "drifts multiplicatively". Not one quantity the reader could derive or check. |
| Completeness reflex | Pre-norm versus post-norm, the 2017 architecture, the learned-shift row of the table. All true, none needed. |
| Terminal deflation | Closed with "nothing here requires action in your baseline", which retroactively marked the preceding 600 words optional. |

The reader's verdict was "really badly explained," and they were right.

## The rewrite

Nothing about root-mean-square normalization (RMSNorm) changes your baseline. It becomes
load-bearing exactly once: at the mixed-precision topic, where it explains why autocast
keeps a list of operations it refuses to run in 16-bit. Here is the version that pays off
there.

**The failure it prevents.** A transformer is thirty blocks in series, each multiplying its
input by learned matrices. The magnitude of the hidden vector drifts as it passes through,
multiplicatively, so a small per-block bias compounds with depth. The gradient at a layer
scales with the size of that layer's input — so if the scale drifts with depth, no single
learning rate is right for both block 3 and block 30. Normalization pins the magnitude at
the entrance to each block, so every block sees inputs in the same range regardless of
where it sits in the stack.

**The mechanism.** For each token's hidden vector `x` of length 576:

```
rms(x) = sqrt( (1/576) * sum(x_i^2) + 1e-5 )
out    = (x / rms(x)) * g
```

`g` is a learned vector of length 576, one scale per feature. Two operations, and the first
one is the entire point.

**`(1/576) * sum(x_i^2)` is a reduction** — it collapses 576 numbers into one by summing
them. Brain floating point 16 (bfloat16) carries 8 bits of mantissa, roughly two to three
significant decimal digits. Accumulating 576 squared values at that precision loses the
small terms into rounding error, and the result feeds a division applied to every element
downstream. So autocast runs the matrix multiplies in bfloat16 and keeps normalization and
softmax — the two reductions in the block — in 32-bit floating point. That list is not
arbitrary; it is exactly the operations that sum across a dimension.

**What it costs, derived rather than measured.** Each norm holds one scale per feature, so
`hidden_size` parameters; there are two per block plus one final. Against a model whose
parameters are dominated by matrices of `hidden_size x hidden_size` and larger, that is a
rounding error — you can do the division for your own config in your head and you will get
a fraction of a percent. Norms will not move your parameter count or your memory
prediction.

They will move your step time. Each one reads and writes the full
`(batch, sequence, hidden)` tensor while performing almost no arithmetic, so it is bound by
memory bandwidth rather than compute. That is part of why step-time predictions derived
from floating-point operation counts always come in optimistic.

**The misreading to avoid.** The wiki says the scales start at 1.0, "so normalization
begins as an identity operation." The learned *scale* starts as the identity; the division
by root mean square is running from the first forward pass. Read the whole layer as a no-op
at initialization and the 11.2744 initial-loss reading in the next wiki section stops
adding up.

## What changed

One through-line — *it is a reduction, and reductions are where reduced precision breaks* —
instead of five sections. Relevance in sentence one instead of a closing deflation. Roughly
two-thirds the length.

The quantities are derived, not measured: 576 and the mantissa width of bfloat16 are facts
about the config and the format, and the parameter fraction is arithmetic the reader can do
for their own model. An earlier draft of this rewrite quoted a measured 35,136 parameters
and 0.0261% from a probe run in-session. Those digits made the passage look rigorous while
teaching nothing the division doesn't give you, and they would be wrong for any other
model. That is the specimen-versus-analytic distinction in `SKILL.md`, failed once here on
purpose so it is recognizable.
