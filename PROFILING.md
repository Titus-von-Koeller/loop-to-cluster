# Profiling contract

Every `scripts/NN_topic_profiled.py` follows this exactly. Numbers from two topics are
only comparable if both were gathered the same way, so this file is a specification, not
a suggestion. If a script needs to deviate, change this file first.

## The two-file rule

| | |
| --- | --- |
| `scripts/NN_topic.py` | hand-written. Trains, prints the loss. **No measurement code.** |
| `scripts/NN_topic_profiled.py` | generated from it. Same training, plus this contract. |

Physical separation, not a `--profile` flag. A flag leaves the instrumentation interleaved
with the logic on the page, and the reader still has to think "that line is measurement, I
can ignore it." That mental step is the cost being avoided.

**The twin is derived, never hand-edited.** When the script changes, regenerate. The two
must agree on training: same seed, same data order, same optimizer, same number of steps.

## The recipe

Order matters. Each step exists because skipping it produces a plausible wrong number.

```python
# 1. Device. No fallback — a CPU run reports zero memory, which looks like a measurement.
device = "cuda:0"

# 2. Build model, optimizer, dataloader exactly as the script does.

# 3. Warmup. Not optional, and not only about cuBLAS handles and autotuning:
#    AdamW allocates exp_avg and exp_avg_sq lazily on the FIRST step(), so a run
#    profiled from iteration 0 under-counts memory by 8 bytes per parameter.
for _ in range(WARMUP_STEPS):  # 5 is enough
    ...full training step...

# 4. Start the measured window.
torch.cuda.synchronize(device)
torch.cuda.reset_peak_memory_stats(device)

# 5. Fixed number of steps, timed with CUDA events, no synchronize inside the loop.
#    No loss.item() inside the loop either — it synchronizes implicitly and hides the
#    launch gaps that wall-clock throughput is supposed to include. Collect detached
#    loss tensors and move them to the host once, after the loop.

# 6. Close the window.
torch.cuda.synchronize(device)
peak = torch.cuda.max_memory_allocated(device)
```

**Peak memory is `max_memory_allocated`.** Not `nvidia-smi`, which includes the CUDA
context and driver overhead. Not `memory_reserved`, which includes cached free blocks.

**Report per-step time, not total.** Median over the measured steps, with p10 and p90.
The median reports the machine; the mean reports the slowest step.

## The four answers to "how much memory"

Conflating these is what makes memory arithmetic feel unpredictable.

| | |
| --- | --- |
| requested | sum of tensor storage bytes — what theory predicts |
| `memory_allocated` | sum of allocator *block* bytes — requested plus block padding |
| `memory_reserved` | what the allocator holds — allocated plus cached free blocks |
| `nvidia-smi` | reserved plus CUDA context (~0.3–0.6 GB) plus driver overhead |

Check arithmetic against **requested** (`harness.measure.requested_bytes`), and report the
allocator's view next to it so padding is quantified rather than mistaken for a broken
prediction.

## Output schema

One JSON per run via `harness.runs.save`, into `bench/results/`. Identical keys across
every topic — that is what makes them diffable.

```
script, timestamp, commit, config, environment, predicted, measured
```

`commit` matters: a change to the training loop moves the numbers for reasons no plot can
see. Never fit a line through runs from two revisions.

Plus one figure per run into `bench/figures/`: loss curve and memory over the step.

## Verification signals — print these every time

| Signal | Expected | Why |
| --- | --- | --- |
| Parameter count | `sum(p.numel() for p in model.parameters())` | the input to every memory and FLOP prediction |
| Initial loss | `ln(V)` | a randomly initialized model predicts uniformly. Read it from a forward on **pristine** weights — one optimizer step moves it just enough to look plausible while being wrong |
| Loss decreases | monotone-ish | the cheapest convergence check |
| Peak memory | matches the analytic prediction | the point of the exercise |

A prediction that missed is the most valuable output. Print the delta; never smooth it.

## Traps that move numbers silently

Each of these was measured here, not recalled.

- **Slicing logits before flattening.** `logits[:, :-1].flatten(0, 1)` copies a
  vocabulary-sized tensor, because the slice is not contiguous. Shift the *labels*
  instead and the flatten stays a view. Worth hundreds of MiB at a 49k vocabulary.
- **Holding a name on the logits through `backward()`.** Cross-entropy saves its own
  output, not its input, so the raw logits are dead after the loss is computed — unless a
  live variable keeps them. Same cost again.
- **Inheriting `float32_matmul_precision`.** At `"high"` an fp32 matmul is truncated to
  TF32 and runs on the tensor cores. Same dtypes, same memory, very different time. An
  fp32 baseline that inherits this is not a baseline. Deferred by review to a "torch
  tricks" script — do not silently enable or disable it in a topic script.
- **`memory_allocated` is not the sum of tensor sizes.** The caching allocator rounds
  each request up and hands unusable remainders to the block, so an identical request can
  occupy different amounts depending on pool state.
- **Autograd saves a Linear's weight transposed**, and a weight whose input does not
  require grad is never saved at all — which is why the ledger classifies by provenance
  rather than shape, and why its count invariant is `<=` and not `==`.

## What accelerate does with the same concerns

Verified against the installed source, file and line. Do not restate these from memory.

- `Accelerator.backward` divides the loss by `gradient_accumulation_steps` —
  `accelerator.py:2840`
- `clip_grad_norm_` (`:2946`) unscales first, via `unscale_gradients` (`:2944`)
- TF32 is coupled to torch.compile, not left to the caller — `state.py:1023-1028`, gated
  on `dynamo_plugin.backend != NO and mixed_precision == "no" and cuda`
- autocast is installed by wrapping `model.forward`, not the call site —
  `accelerator.py:1818-1824`
- the grad scaler is built from the distributed type — `accelerator.py:583`
