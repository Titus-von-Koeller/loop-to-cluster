# loop-to-cluster

Learning distributed training by building it up one concept at a time, from a single-GPU
training loop to a sharded multi-GPU one — predicting the numbers before measuring them.

This is a **lab notebook, not a codebase**. Each `steps/stepN_*/` directory is a complete,
runnable record of one concept, frozen at the point it was understood.

```text
l2c/                     installable package — shared, never edited in a lesson
  harness/               measurement, memory arithmetic, reporting
    measure.py           CUDA-event timing, the memory staircase, requested-vs-block bytes
    ledger.py            what autograd actually saves, by category and dtype
    predict.py           model states, the autocast weight cache, ln(V)
    precision.py         the four precision arms
    report.py            predicted-versus-measured tables, results.jsonl
    fit.py               least squares over a sweep
    runner.py            one subprocess per configuration, results cached by flags
  common/                model construction, tokenized data
  paths.py               where caches and results land
tests/                   the harness's own arithmetic, CPU-only where possible
steps/
  step1_training_loop/   train.py + prediction.toml + NOTES.md
  step2_mixed_precision/ train.py + compare.py + prediction.toml + NOTES.md
  sweep.sh
results.jsonl            every run, appended (gitignored)
```

## The one rule

> **If a lesson touches it, duplicate it. If no lesson ever touches it, it goes in the
> package.**

The training loop is duplicated in every step, deliberately. Factoring the variation
behind an interface would hide exactly what you are trying to see, and
`diff steps/step1_*/train.py steps/step2_*/train.py` *is* the lesson. It also pastes
straight into a write-up.

`common/` may hold data preparation and model construction. It may never hold a training
loop. When you are tempted, that is the abstraction trap.

Drift between steps is fine — `step2/` records what you knew at step 2, and you do not
refactor a notebook entry. With one exception, which the measurements depend on:

> **A comparison never spans two step directories.** If a step claims "X is faster than
> Y", that step ran both arms itself, with one seed and one measurement path. Step 2's
> `compare.py` is the pattern.

Drift is harmless between steps and fatal inside a delta.

## Install

```bash
pixi install
pixi run python -c 'import torch; print(torch.cuda.is_available())'
```

`l2c` is installed editable via pixi, so `import l2c` resolves from any working directory —
notebooks under `notebooks/`, step scripts under `steps/`, and subprocesses the comparison
driver spawns. `pyproject.toml` carries build metadata only; dependencies live in
`pixi.toml` where the lock file governs them.

Pin one GPU for anything you intend to measure:

```bash
export CUDA_VISIBLE_DEVICES=0
```

A second card driving a display starts several hundred MiB down and its clocks move with
whatever the compositor is doing.

## Workflow per step

1. Read the theory. Write the derivation in `NOTES.md`.
2. Fill in `prediction.toml` — **before running anything**, and commit it.
3. Run. The harness prints predicted versus measured.
4. Paste the table into `NOTES.md`; write what surprised you.
5. `cp -r` the directory forward, add one concept, repeat.

Committing the prediction in its own commit is what makes it a prediction.
`prediction.toml` is read with `tomllib`, so there is no YAML dependency; omit a key and
the harness just reports the measurement.

## Run

```bash
python steps/step1_training_loop/train.py       # the fp32 baseline
./steps/sweep.sh                                # six depths, then fit the line
python steps/step2_mixed_precision/compare.py   # fp32 / tf32 / bf16 / fp16
pytest tests/                                   # the harness's own arithmetic
```

The tests cover the bookkeeping every measurement rests on — byte accounting, ledger
categorization, the least-squares fit, the precision arms — not the measurements
themselves. Whether a 4090 takes 56 ms for a step is not a testable proposition; whether
`requested_bytes` deduplicates a tied weight is, and getting it wrong would move every
memory number at once.

The first run tokenizes wikitext-2 into `.cache/` (a minute or two). Every run after that
is offline and instant.

One matching number is a coincidence; six points on a line is a validated formula. The
sweep varies **depth**, because every quantity of interest is linear in it — so the fitted
slope is the per-layer cost and the intercept is the embedding table, two checks from one
sweep.

## What has been measured

Model: SmolLM2-135M's architecture, built from a config rather than a checkpoint, so the
initial loss must be `ln(49152) = 10.8027`. Data: wikitext-2, packed into fixed-length
blocks. One RTX 4090, 4 × 512 = 2048 tokens per step.

| | predicted | measured |
| --- | --- | --- |
| parameters | 134,515,008 | 134,515,008 |
| bytes/parameter | 16.00 | 16.00 |
| initial loss | 10.8027 | 10.8992 |
| model states | 2052.5 MiB | 2052.5 MiB |
| autocast weight cache (bf16) | 256.5 MiB / 211 tensors | 256.5 MiB / 211 tensors |
| per-layer parameters (fitted slope, R²=1.0) | 3,540,096 | 3,540,096.00 |

Mixed precision, bf16 against fp32: loss unchanged (max divergence 0.0018 against a
baseline noise floor of 0.1748), **1.89× faster**, model states **up** by exactly one
eighth, peak memory **down** 1.38×.

Three findings worth the price of the harness:

- **`memory_allocated()` is the sum of allocator *block* sizes, not tensor sizes.** For
  this model that is 90.3 MiB of padding, and the same request was seen occupying two
  different block sizes depending on pool state. Arithmetic is checked against summed
  storage bytes; the padding is reported as its own line.
- **Flash attention is dtype-gated, so activation memory has a `T²` term at fp32.** The
  only usable SDPA backend for fp32 is MATH, which materializes a `(B, heads, T, T)`
  matrix — 36 of the 129 MiB per layer here. Going to bf16 does not merely halve tensors,
  it makes a fused kernel available and deletes the term.
- **38% of saved bytes stay fp32 under autocast**, because RMSNorm's fp32 weight promotes
  the bf16 hidden states back up. And the vocabulary-sized logits are 384 MiB in *every*
  arm, since `ForCausalLMLoss` upcasts them.

## Steps

| Step | Concept | Key prediction to test | State |
| --- | --- | --- | --- |
| 1 | bare training loop | parameter count from config; 16 B/param; initial loss = ln(V) | done |
| 2 | mixed precision | model states do **not** shrink — they grow by 2 B/param; only activations shrink | done |
| 2b | TF32 as a control | same dtypes and memory, tensor-core kernels: isolates compute from storage | done |
| 3 | activation checkpointing | block activations collapse to roughly one layer's worth, for ~30% more step time | next |
| 4 | gradient accumulation | equal global batch ⇒ equal loss; peak memory drops | |
| 5 | DDP | throughput scales; per-rank memory rises by the gradient bucket, not by zero | |
| 6 | FSDP2 | per-rank model states fall to ~1/world_size | |
| 7 | DeepSpeed ZeRO | the same arithmetic as FSDP, different ownership of the optimizer | |

Step 4 is where the data format changes from packed fixed-length to padded
variable-length. Dividing an accumulated loss by `grad_accum_steps` is exactly right only
when every microbatch holds the same number of tokens, so the correctness bug is
structurally invisible until the format changes.

Step 5's prediction is deliberately not "per-rank memory unchanged": DDP allocates
gradient reduction buckets (`bucket_cap_mb`, 25 MiB by default), so the rise is
predictable rather than absent.

## Orientation for accelerate

Every step is bare torch on purpose, and every mechanism built by hand has a counterpart
in accelerate. The docstrings name the file and class that owns each one, because knowing
which file owns what is most of what onboarding to that codebase consists of. Each step's
`NOTES.md` ends with a translation table.

The three that are doing more than they appear to:

- `Accelerator.backward` divides the loss by `gradient_accumulation_steps`
  (`accelerator.py:2840`) and routes through the grad scaler when one exists.
- `Accelerator.clip_grad_norm_` unscales gradients *first* (`accelerator.py:2944`).
  Clipping scaled gradients compares their norm against an inflated threshold.
- `AcceleratedOptimizer.step` (`optimizer.py:162-175`) gates on
  `GradientState.sync_gradients` and infers gradient overflow from whether the inner step
  ran, exposing it as `step_was_skipped()`.

`mixed_precision` has no `tf32` value, and TF32 is not left to the caller either — it is
coupled to torch.compile. `AcceleratorState` sets `allow_tf32 = True` exactly when a
dynamo backend is requested *and* `mixed_precision == "no"`
([`state.py:1023`](https://github.com/huggingface/accelerate/blob/main/src/accelerate/state.py)).
So enabling compilation moves the fp32 baseline onto the tensor cores as a side effect.
Given the 1.36× measured for TF32 alone on this box, the same unchanged bf16 change reads
as 1.89× without a dynamo backend and 1.39× with one. transformers couples them
identically, gated on `torch_compile` (`training_args.py:1604`), but does expose an
explicit `tf32` flag.

## Environment

| | |
| --- | --- |
| Python | 3.14 |
| PyTorch | 2.13, CUDA 13 wheels from PyPI |
| transformers | 5.x |

- PEP 649 makes annotations lazy in 3.14, so `from __future__ import annotations` is
  obsolete here and ruff is configured to flag it.
- `attn_implementation` is pinned to `sdpa` rather than left to transformers' default. The
  choice changes which tensors are saved for backward, so an unpinned baseline moves the
  day something installs flash-attention.
- `torch` comes from PyPI rather than conda-forge: since 2.11 the default PyPI wheels *are*
  the CUDA 13 build. The tradeoff is that anything depending on torch must also come from
  PyPI.
- `libcuda.so.1` comes from the host driver, since no wheel can bundle it — it has to match
  the loaded kernel module.
- `pixi.toml` and `pixi.lock` are committed here on purpose. A global gitignore keeps them
  untracked in cloned upstream repos; `.gitignore` negates that for this project, where
  they are the environment definition.
- `direnv` activates the environment on `cd` via `.envrc`, which is intentionally not
  committed — it is machine-local, and layers this pixi environment on top of the CUDA and
  native toolchain from the nix devShell at `~/src`.
