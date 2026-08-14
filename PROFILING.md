# Profiling contract

Numbers from two topics are comparable only if both were gathered the same way, so this is a
specification rather than a suggestion. A twin that needs to deviate means this file is wrong:
change it here first.

## Two files

| | |
| --- | --- |
| the study script | hand-written. Trains, prints the loss. **No measurement code.** |
| `<script>_profiled.py` | generated from it. Same training, plus this contract. |

Physical separation rather than a `--profile` flag. A flag leaves instrumentation interleaved
with the logic, and the reader still has to think "that line is measurement, I can ignore it" —
that mental step is the cost being avoided.

**The twin is derived, never hand-edited.** When the script changes, regenerate. The two must
agree on training: same seed, same data order, same optimizer, same number of steps, and the
same model construction — including config fields like `dtype` and `use_cache`, which change
what is measured while changing nothing a shape-based check would notice.

## The recipe

Order matters. Each step exists because skipping it produces a plausible wrong number.

**Device.** `measure.require_cuda()`, which raises rather than falling back. A CPU run reports
zero memory, and zero looks like a measurement.

**Build model, optimizer and dataloader exactly as the script does.**

**Warm up**, counting steps globally so a nested epoch loop is unaffected. Five steps is enough.
Warmup is not only about cuBLAS handles and autotuning: AdamW allocates `exp_avg` and
`exp_avg_sq` lazily on the *first* `step()`, so a window opened at iteration zero under-counts
memory by 8 bytes per parameter.

**Open the window** after warmup, with `torch.cuda.synchronize(device)` then
`measure.reset_peak(device)`.

**Measure a fixed number of steps**, timed with `measure.StepTimer(device, capacity)` — CUDA
events, no synchronize inside the loop. No `loss.item()` inside it either: it synchronizes
implicitly and hides the launch gaps that throughput is supposed to include. Collect detached
loss tensors and move them to the host once, after the loop. Warmup and measured steps are both
real training steps, so the twin trains on the same batches in the same order as the script.

**Close the window** with `torch.cuda.synchronize(device)`, then read
`torch.cuda.max_memory_allocated(device)`.

## Which memory number

Conflating these is what makes memory arithmetic feel unpredictable.

| | |
| --- | --- |
| requested | sum of tensor storage bytes — what theory predicts |
| `memory_allocated` | sum of allocator *block* bytes: requested plus block padding |
| `memory_reserved` | what the allocator holds: allocated plus cached free blocks |
| `nvidia-smi` | reserved plus CUDA context (~0.3-0.6 GB) plus driver overhead |

**Peak is `max_memory_allocated`.** Check arithmetic against requested
(`measure.requested_bytes`) and report the allocator's view beside it, so padding is quantified
rather than mistaken for a broken prediction.

**Report per-step time, not total.** Median over the measured steps, with p10 and p90. The
median reports the machine; the mean reports the slowest step.

## Output

One JSON per run via `runs.save(script, config=..., environment=..., measured=..., predicted=...)`,
into `bench/results/`. It records `commit` itself, which matters because a change to the
training loop moves the numbers for reasons no plot can see — never fit a line through runs from
two revisions.

Plus one figure per run into `bench/figures/`: loss curve and memory over the step.

## Signals to print every time

| Signal | Expected |
| --- | --- |
| Parameter count | `sum(p.numel() for p in model.parameters())`, the input to every prediction |
| Initial loss | just above `ln(V)`, read on weights that have taken zero optimizer steps |
| Loss falls | compare the first measured steps against the last; per-step values on shuffled data are noisy and not monotone |
| Peak memory | against the analytic prediction |

The initial-loss check catches a checkpoint loaded when random weights were intended, a wrong
reduction, and a vocabulary mismatch between tokenizer and model. It is blind to which token was
scored, so it says nothing about the label shift.

**A prediction that missed is the most valuable output.** Print the delta; never smooth it.

## Traps that move numbers silently

**Inheriting `float32_matmul_precision`.** At `"high"` an fp32 matmul is truncated to TF32 and
runs on the tensor cores: same dtypes, same memory, very different time. An fp32 baseline that
inherits this is not a baseline. Set it deliberately in the topic that studies it, and nowhere
else.

**`memory_allocated` is not the sum of tensor sizes.** The caching allocator rounds each request
up and hands the unusable remainder to the block, so an identical request can occupy different
amounts depending on pool state.

**Autograd saves a `Linear`'s weight transposed**, and a weight whose input does not require a
gradient is never saved at all. The ledger therefore classifies by provenance rather than by
shape, and its count invariant is `<=` rather than `==`.
