# Decisions

What was chosen, and why. Reasons rather than rules: **a reason that no longer holds is grounds
to revisit a decision, not to comply with it.** Dated, so staleness is visible — two of the three
routing rules in the deleted `docs/CONVENTIONS.md` had already been contradicted by the structure
they governed, and nobody noticed.

The one prescriptive section is *Do not resurrect*, at the bottom.

## The baseline script

*Recorded in `docs/START_HERE.md` at the 2026-08-07 restructure, which followed a code review by
Marc Sun.*

**Random init** via `AutoConfig.from_pretrained` then `from_config`. A pretrained checkpoint
forfeits the `ln(V)` prediction, the only analytic number available before the first step.

**Real text, not a synthetic batch.** The loop is where model, data, optimizer and loss meet; a
synthetic tensor removes one of the four.

**`model(ids, labels=ids).loss`**, which HuggingFace's own course uses. Measured identical in peak
memory to an explicit `cross_entropy` with `del logits`.

**No learning-rate schedule.** A decaying schedule makes the learning rate a function of the
total step count, so a later script that changes the step budget is no longer comparable. The
cost of this choice is that warmup's role never becomes visible here; it is on the backlog.

**Keep `cfg.dtype` and `cfg.use_cache = False`.** Neither is canonical and both prevent a silent
error: bfloat16 parameters would halve every ledger row, and a key-value cache belongs to no row.
That cache is a memo between *consecutive* forward passes during incremental decoding — training
runs one pass over all positions at once and has nothing to reuse it.

**No eval loop for now.** Which makes this script a demonstration that the mechanics of a
training loop run, not that training works: with no held-out split, a falling loss cannot
separate learning from memorizing.

*Corrected 2026-08-20.* **Flat step loop over `cycle()`**, not the nested epoch loop the earlier
note described. One monotonic counter, so logs align across scripts and the loop shape survives
the arrival of gradient accumulation; `itertools.cycle` would cache the first pass and stop
reshuffling. The earlier entry also claimed 128 steps per epoch — it is 126 (1,015 blocks, batch
8, `drop_last`).

*2026-08-20.* **`torch.set_float32_matmul_precision("highest")`**, replacing
`torch.backends.cudnn.allow_tf32 = False`. cuDNN governs convolutions, which this model does not
have; the flag that decides whether an fp32 matmul stays fp32 is the matmul one. The loss curve
is byte-identical across the swap, which confirms the old line was a no-op in effect. Under torch
2.13 the per-backend `fp32_precision` fields are all `none` and defer to the legacy booleans;
`"highest"` is the setting that survives that migration.

## How scripts are compared

*2026-08-20. Moved here out of the `00` docstring, where it was instruction for whoever writes
script 03 rather than help for whoever reads script 00.*

A later script changes one topic and pins everything that topic does not require: the seed, the
data order, the initial weights, the global batch size, and the number of updates. Pinning the
global batch is why a distributed script divides `batch_size` by the world size rather than
multiplying what the baseline saw.

This is not an edit count. Distributed data parallel needs process-group setup, a sampler, a
wrapper and teardown — "one change" as a diff-size rule would forbid the topics that matter most,
and it rewards hiding setup behind a helper.

Accumulation, DDP and FSDP only relocate arithmetic, so a correct one reproduces this baseline's
gradients for the same global batch — checkable at step 1, before numeric drift has anywhere to
hide. Mixed precision changes the arithmetic on purpose, so it gets a measured tolerance instead
of an equality.

## Where a file goes

*2026-08-20.*

`scripts/` holds anything you will come back to, read again, or compare against. `notebooks/`
holds anything you are doing once to see what happens. Whether a run is deliberately broken has
no bearing on which: an experiment that turns out to be worth keeping moves to `scripts/` as a
variant sharing its parent's number — `03-ddp-unsynced.py`.

## How a run is observed

*2026-08-20.*

Metric logging is inline in the study script, because real training code contains it and reading
it is part of learning the loop. Memory snapshots are external, through `snapshot.py`, because
recording allocation history is studying-the-training rather than training. This replaces a
generated `*_profiled.py` twin governed by a written contract: one runner works on every script
present and future, needs no template, and cannot drift from what it measures.

## Do not resurrect

Prescriptive, unlike the rest of this file. These were tried and cost something.

- `steps/stepN_*/` directories, `prediction.toml`, `NOTES.md` as a per-step lab notebook
- a shared `collect` / `publish` / row-spec reporting layer
- `build_model` / `build_loader` / `build_batches` / `training_step` wrappers
- a shared argparse module, or a subprocess runner driving several arms at once
- `results.jsonl`

Study scripts are read linearly and modified independently, so shared helpers cost comprehension
and let one experiment perturb another.

*Added 2026-08-20:*

- **Edit-count rules** — "one change per script", "~50 lines", "~two paragraphs". A count is
  never a specification: it is gameable, it goes stale the moment the thing it counts changes,
  and it deters the topics that legitimately need thirty lines of setup.
- **A document governing prose.** `docs/CONVENTIONS.md` was a page standard for a writing process
  that has been abandoned, and it had already drifted out of agreement with the Notion structure
  it governed.
- **Generated `*_profiled.py` twins** with a measurement contract, a template, and a
  regeneration rule. Replaced by inline logging plus one external runner.
