# Step 2 — mixed precision

## What this step adds

`torch.autocast` around the forward, and a `GradScaler` for the fp16 arm. Four arms are
measured rather than two, because "mixed precision on/off" bundles two separable effects:

| arm | tensors | matmul kernels | attention backend |
| --- | --- | --- | --- |
| fp32 | fp32 | CUDA cores, 24-bit significand | MATH (materializes B×H×T×T) |
| tf32 | fp32 | tensor cores, 11-bit significand | MATH |
| bf16 | mixed | tensor cores | FLASH |
| fp16 | mixed | tensor cores | FLASH |

fp32 → tf32 holds memory constant and changes only the kernels, isolating "tensor cores
are faster". tf32 → bf16 keeps the tensor cores and changes the dtype, isolating "smaller
tensors use less memory". Comparing fp32 with bf16 alone conflates the two.

## Prediction working

### Model states do not change

`torch.autocast` converts no parameter, no gradient, and nothing the optimizer holds. It
casts op *inputs*. So all four arms predict the identical **2052.5 MiB** from step 1.

### The autocast weight cache — the term that makes memory grow

Inside an autocast region each eligible weight is cast once and the result cached
(`cache_enabled=True` by default), then held by the graph through backward. The bf16
copies are live *simultaneously* with the fp32 masters.

Eligible means the matmul family, so every `nn.Linear` weight and no RMSNorm weight:

    per layer   q 331,776 + k 110,592 + v 110,592 + o 331,776
                + gate 884,736 + up 884,736 + down 884,736  =  3,538,944
    30 layers                                                = 106,168,320
    tied lm_head (an nn.Linear sharing the embedding table)   =  28,311,552
    total eligible                                           = 134,479,872
    × 2 bytes                                                = 256.5 MiB
    tensors: 30 × 7 + 1                                      = 211

256.5 against 2052.5 is **exactly one eighth** — two bytes added to sixteen. Mixed
precision is widely assumed to shrink model states; it grows them, by a predictable 12.5%.

### Activations shrink by more than half

Not because of the dtype alone. Flash attention is **dtype-gated**: at fp32 the only
usable SDPA backend is MATH, which materializes a `(B, heads, T, T)` attention matrix
worth 36 of the 129 MiB per layer. At 16 bits FLASH becomes available and replaces it with
a `(B, heads, T)` logsumexp. So:

    block cost per layer:  (129 − 36) / 2  =  47 MiB, not 64
    30 layers                              ≈ 1410 MiB
    logits, bf16 + the fp32 copy the loss makes ≈ 576 MiB
    total                                  ≈ 2000 MiB

### Loss must be unchanged, and "unchanged" needs a definition

Bitwise equality is impossible: bf16 carries 8 significand bits, so ~4e-3 relative error
per operation. The testable claim is that the gap between arms is smaller than the noise
already present in the baseline, so `compare.py` measures the baseline's own step-to-step
standard deviation and uses it as the floor.

## Result

```
quantity                           fp32         tf32         bf16         fp16
------------------------------------------------------------------------------
model states (MiB)              2,052.5      2,052.5      2,052.5      2,052.5
weight cache (MiB)                  0.0          0.0        256.5        256.5
weight cache tensors                  0            0          211          211
activations (MiB)               4,277.3      4,277.3      2,213.7      2,213.7
  vs fp32                         1.00x        1.00x        1.93x        1.93x
  of which still fp32           4,781.4      4,781.4        936.0        936.0
  vocab-sized logits              384.0        384.0        384.0        384.0
peak allocated (MiB)            6,621.9      6,621.9      4,808.6      4,808.6
  vs fp32                         1.00x        1.00x        1.38x        1.38x
median step (ms)                  106.3         78.0         56.2         57.3
  vs fp32                         1.00x        1.36x        1.89x        1.86x
tokens/sec                       19,250       26,254       36,418       35,714
initial loss                    10.8992      10.8991      10.8990      10.8991
final loss                       7.3267       7.3266       7.3277       7.3378
updates skipped                       0            0            0            0
grad scale                            1            1            1       65,536

weight cache bf16 (MiB)                256.5           256.5    +0.00%  ok
activations bf16 (MiB)               2,000.0         2,213.7   +10.68%  ok
peak bf16 (MiB)                      4,600.0         4,808.6    +4.54%  ok
median step bf16 (ms)                   38.0            56.2   +47.90%  OFF

baseline step-to-step stdev (the noise floor): 0.1748
arm          mean |diff|    max |diff|    vs floor
fp32              0.0000        0.0000      within
tf32              0.0001        0.0006      within
bf16              0.0006        0.0018      within
fp16              0.0278        0.0589      within
```

All four claims hold. Loss is unchanged — every arm's maximum divergence from the baseline
is below the baseline's own noise floor, with bf16 two orders of magnitude inside it.
Training is faster: 1.89×. Model states are *up* by 256.5 MiB, exactly the predicted eighth.
Peak memory is down 1.38×, so the total moves in the opposite direction from the model
states, which is only visible because the two were budgeted separately.

## What surprised me

**The weight-cache prediction was exact, and it caught two bugs in the ledger.** It first
measured 189.8 MiB over 150 tensors against a predicted 256.5 over 211. The shortfall
decomposed exactly: 54 MiB (the lm_head cast) + 12.66 MiB (`k_proj` and `v_proj` across 30
layers). Cause: autograd saves a linear's weight **transposed**, so `k_proj`'s `(192, 576)`
weight is saved as a `(576, 192)` view that matches no parameter's shape — while square
`q_proj` and `o_proj` matched by luck and `gate/up/down` matched each other's transposes.
Separately the logits test, keyed on a trailing dimension of `vocab_size`, was swallowing
the transposed lm_head cast. A prediction precise enough to decompose its own error is
worth more than a prediction that is merely close.

**38% of saved bytes are still fp32 under autocast.** 936.0 MiB of the 2432.3 MiB the graph
holds in the bf16 arm. RMSNorm is the reason: its fp32 weight multiplies the bf16 hidden
states and type promotion pushes the result back to fp32, so `last_hidden_state` is fp32
even inside the autocast region. "Activations halve" is wrong in both directions at once —
some tensors never shrink, and the attention matrix vanishes entirely.

**The vocabulary-sized tensor gets no benefit at all: 384.0 MiB in every arm.**
`ForCausalLMLoss` does `logits = logits.float()` (transformers `loss/loss_utils.py:59`), so
only the fp32 copy is ever saved for backward. At `V = 49152` that is a fixed 384 MiB the
dtype cannot touch, and it grows with vocabulary while everything else grows with hidden
size. On a model with a 150k+ vocabulary it would dominate.

**Step time was the bad prediction, at +48%.** 56 ms against a predicted 38. The measured
speedups are tf32 1.36× and bf16 1.89×, so roughly two thirds of the win comes from
reaching the tensor cores at all and the remaining third from the narrower dtype plus flash
attention. At 576 hidden the kernels are too small to approach peak throughput — the same
reason step 1's FLOP estimate was optimistic. Without the tf32 arm this would have read as
"bf16 gives 1.89×" with no way to see which half of the change earned it.

**fp16 needed the scaler and never used it.** Scale settled at 65,536 — the initial value,
never lowered — with 0 of 36 updates skipped. So this model's gradients at this learning
rate never reach fp16's 6.1e-5 floor, and fp16 is as safe as bf16 here while being very
slightly worse on final loss (7.3378 against bf16's 7.3277, and a mean curve divergence of
0.0278 against 0.0006 — 46× larger, though still inside the noise floor). fp16's 11
significand bits beat bf16's 8; the reason to prefer bf16 is that it needs no scaler
machinery, not that it is more accurate.

## Bare torch → accelerate

This file collapses to `Accelerator(mixed_precision="bf16")` plus `accelerator.backward`.
What that hides:

| here | accelerate | where |
| --- | --- | --- |
| `torch.autocast(...)` around the forward | wraps `model.forward` in the context | `accelerator.py` ~1820 |
| `GradScaler(enabled=needs_scaler)` | built from the distributed type | `accelerator.py:583`, `get_grad_scaler` |
| `scaler.scale(loss).backward()` | `accelerator.backward(loss)` | `accelerator.py:2846` |
| `scaler.unscale_(optimizer)` then clip | `accelerator.clip_grad_norm_` unscales first | `accelerator.py:2944` |
| `scaler.step()`, `scaler.update()` | `AcceleratedOptimizer.step` | `optimizer.py:162-175` |
| counting skipped updates by hand | `optimizer.step_was_skipped()` | `optimizer.py:188` |

Three things worth carrying forward:

- accelerate wraps the **model's forward**, not the call site, and wraps the output in
  `convert_outputs_to_fp32`. So model outputs come back fp32 under accelerate and bf16
  here — a real behavioral difference, not just a syntactic one.
- there is **no `tf32` option**. `float32_matmul_precision` is orthogonal to
  `mixed_precision` and left entirely to the caller, so an accelerate benchmark that does
  not pin it is comparing against an unknown baseline. Given the 1.36× measured above,
  that is the difference between reporting 1.89× and reporting 1.39×.
- `Accelerator.backward` divides the loss by `gradient_accumulation_steps`
  (`accelerator.py:2840`). At 1 that is a no-op, which is why swapping it in looks harmless
  here and stops being harmless at step 3.
