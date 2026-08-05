# Step 1 — the bare training loop

## What this step adds

Nothing. It is the baseline: one GPU, fp32, AdamW, no distribution, no mixed
precision, no accumulation. Everything later is a delta against it, so it exists to be
instrumented rather than to be interesting.

## Prediction working

### Parameters

Config: `V = 49152`, `h = 576`, `i = 1536`, `L = 30`, 9 query heads, 3 KV heads,
tied embeddings. transformers derives `head_dim = h / heads = 576 / 9 = 64`, so the
query projection is square while K and V are a third as wide.

| term | expression | value |
|---|---|---|
| embedding | `V · h` | 28,311,552 |
| Q | `h · (9 · 64)` = `h · h` | 331,776 |
| K | `h · (3 · 64)` | 110,592 |
| V | `h · (3 · 64)` | 110,592 |
| O | `(9 · 64) · h` = `h · h` | 331,776 |
| attention subtotal | | 884,736 |
| MLP (SwiGLU, three matrices) | `3 · h · i` | 2,654,208 |
| two RMSNorms | `2h` | 1,152 |
| **per layer** | | **3,540,096** |
| all layers | `L · 3,540,096` | 106,202,880 |
| final norm | `h` | 576 |
| tied `lm_head` | shares the embedding | 0 |
| **total** | | **134,515,008** |

134.5M, which is where the "135M" in the name comes from.

The tie is load-bearing for everything downstream: `parameters()` deduplicates by
identity, so the 28.3M embedding is counted once. If it were counted twice, AdamW would
hold two sets of moments for it and the budget would be out by 28.3M × 8 B = 226 MB.

### Model states

fp32 parameters + fp32 gradients + AdamW's `exp_avg` and `exp_avg_sq`, all fp32:

    4 + 4 + (4 + 4) = 16 bytes/parameter
    134,515,008 × 16 = 2,152,240,128 B = 2052.6 MiB

AdamW allocates its two moments lazily, on the first `step()` — not in `__init__`.
Reading memory before that first step under-counts by exactly 8 B/param.

### Initial loss

A randomly initialized model is uniform over the vocabulary, and cross-entropy against
a uniform distribution over `V` classes is `ln V`:

    ln 49152 = 10.8027

This is the cheapest correctness check in the lab. If it does not land here, the loss
computation is wrong and every number after it is noise. It has to be read from a
forward on *pristine* weights — one AdamW step moves each parameter by roughly the
learning rate, which is small enough that a contaminated reading still looks plausible.

### Activations

2048 tokens per step (`4 × 512`). Counting what autograd saves per token per layer, in
elements:

| saved tensor | elements |
|---|---|
| normalized input to Q/K/V | 576 |
| Q, K, V | 576 + 192 + 192 |
| Q, K after RoPE | 576 + 192 |
| K, V expanded to 9 heads by `repeat_kv` | 576 + 576 |
| attention output, logsumexp | 576 + 9 |
| O projection output | 576 |
| first residual sum | 576 |
| normalized input to the MLP | 576 |
| gate, up projections | 1536 + 1536 |
| SiLU output, the product | 1536 + 1536 |
| down projection output | 576 |
| second residual sum | 576 |
| **per token per layer** | **≈ 13,065** |

    13,065 × 4 B × 30 layers  = 1.57 MB/token
    × 2048 tokens             = 3062 MiB

Then the vocabulary-sized tensors, which are one tensor each but large:

    logits          2048 × 49152 × 4 B = 384 MiB
    log_softmax     same shape again    = 384 MiB

Total estimate **≈ 3840 MiB**. This is the number to be least confident about — hence
the saved-tensor ledger, which reports the real inventory by category and by dtype so
the residual can be attributed rather than shrugged at.

### Peak

At the moment backward starts, all activations are live and model states are complete.
Gradients then grow while activations are released, so the two roughly trade off:

    2053 + 3840 ≈ 5893 MiB, predicted as 6000

### Throughput

    6 · N · tokens = 6 × 134.5e6 × 2048 = 1.65 TFLOP/step

A 4090 does perhaps 35 TFLOPS of real fp32 GEMM, which suggests 21 steps/s. Thirty
layers of small matmuls will not reach that, so: **12 steps/s**.

## Result

<!-- paste the harness table -->

## What surprised me

<!-- -->

## Bare torch → accelerate

The loop is bare torch on purpose. These are the calls that would replace each line, and
the file each one lives in:

| here | accelerate | where |
|---|---|---|
| `net.to(device)` | `accelerator.prepare(model)` | `accelerator.py`, `_prepare_model` |
| `loss.backward()` | `accelerator.backward(loss)` | `accelerator.py:2818` |
| `clip_grad_norm_(net.parameters(), n)` | `accelerator.clip_grad_norm_(...)` | `accelerator.py:2946` |
| `torch.manual_seed(seed)` | `accelerate.utils.set_seed(seed)` | `utils/random.py:40` |
| `DataLoader(generator=...)` | `SeedableRandomSampler` | `data_loader.py:73` |
| `torch.device("cuda")` | `accelerator.device` | `state.py` |
| `print(...)` | `accelerator.print(...)` | `accelerator.py` |

Two of these are doing more than they appear to, and they are worth reading before
step 2 rather than after:

- `Accelerator.backward` **divides the loss** by `gradient_accumulation_steps`
  (`accelerator.py:2840`) before calling backward, and routes through the grad scaler
  when one exists (`:2846`). At `gradient_accumulation_steps=1` that division is a
  no-op, which is why the substitution looks harmless here and stops being harmless at
  step 3.
- `Accelerator.clip_grad_norm_` **unscales first** (`:2944`). Clipping scaled gradients
  would clip to the wrong norm — the threshold would be multiplied by whatever the
  scaler chose. This is the ordering constraint step 2 has to respect by hand.
