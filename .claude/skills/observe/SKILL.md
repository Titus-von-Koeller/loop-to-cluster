---
name: observe
description: Context for observing a study script — what the instruments are, how to run them, which quantities are derivable before looking, and how to read the memory picture. Use when working on what a run costs in memory or throughput.
---

# Observing a run

## The instruments

```bash
CUDA_VISIBLE_DEVICES=0 pixi run python scripts/NN-name.py
```

logs per-step metrics to trackio: loss, gradient norm, learning rate, step time, tokens/s, peak
memory. Dashboard with `trackio show --project loop-to-cluster`. Note that trackio also logs
CPU and system metrics automatically when psutil is present — those arrive without being asked
for, which is worth knowing before reading a crowded dashboard as signal.

```bash
CUDA_VISIBLE_DEVICES=0 pixi run python snapshot.py scripts/NN-name.py
```

runs the same script unmodified and writes `bench/snapshots/NN-name.pickle`, around 24 MB for
the baseline. Open it at <https://pytorch.org/memory_viz>. The entry cap fills during the early
steps, so the picture covers startup and the first updates rather than the whole run — which is
where the interesting allocations are.

## What is derivable before looking, and what is not

Derivable from the parameter count alone, under fp32 AdamW: parameters at 4 bytes each,
gradients at 4, and the optimizer's two moments at 8 — the 16-bytes-per-parameter figure. The
parameter count prints at the top of every run. Those are the flat bands.

Not derivable yet: the activation sawtooth. It scales with batch × sequence × depth, but the
coefficient depends on the attention implementation and on what autograd chooses to save, so it
needs either a formula — Transformer Math 101, in the Notion resource list — or a measurement.
"I can't estimate this yet" is a real answer here, and it names what to learn next.

## Reading the picture

The bands are the memory ledger drawn against time, so *born* and *dies* is what the shape
shows. Parameters exist before step 1 and never die. Gradients live from backward to
`zero_grad`. Activations are born in forward and consumed by backward, which is the sawtooth.
The optimizer's two bands appear one step in rather than at the start, because AdamW allocates
`exp_avg` and `exp_avg_sq` lazily on the first `step()` — a window opened at iteration zero
undercounts by 8 bytes per parameter.

Four different numbers are all called "memory": requested bytes (what theory predicts),
`memory_allocated` (requested plus allocator block padding), `memory_reserved` (allocated plus
cached free blocks), and what `nvidia-smi` shows (reserved plus a CUDA context of roughly
0.3–0.6 GB plus driver overhead). Peak means `max_memory_allocated`.

## Whose work is whose

The notes are Titus's, in Notion; Marc reviews weekly and needs the reasoning visible,
predictions that missed included — those are the most useful lines on a page. A guess made
before looking is worth more than an accurate number found afterwards. A guess with no basis is
worth nothing, and where there's no basis yet, going to get one is the work.
