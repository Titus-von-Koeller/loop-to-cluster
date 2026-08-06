"""Step 1 — the bare training loop. Single GPU, fp32, nothing else.

This is the baseline every later step is diffed against. It is deliberately boring:

    forward -> loss -> backward -> clip -> update -> reset

Run from the repo root:

    python steps/step1_training_loop/train.py
    python steps/step1_training_loop/train.py --num-layers 6

The four phases are the methodology, not decoration:

* **phases** — one step with a synchronize between each part of it, which is the only
  way to see parameters, activations, gradients and optimizer state appear separately.
  It is also the only forward that ever runs on pristine weights, so the
  `ln(vocab_size)` check is read here.
* **warmup** — absorbs everything that happens exactly once: cuBLAS handle creation,
  kernel autotuning, and the caching allocator growing its pools. Timing across warmup
  measures start-up, not steady state.
* **timed** — touches nothing but the step. No `loss.item()`, no memory queries, no
  hooks. Losses land in a device buffer and come back to the host once, at the end; a
  per-step `.item()` would synchronize every step and hide the launch gaps that
  wall-clock throughput is supposed to include.
* **anatomy** — the saved-tensor ledger, which costs a Python call per saved tensor and
  therefore runs only after timing is finished.

The equivalent in accelerate, for orientation. Everything below is bare torch; these are
the calls that would replace it, and step 4 is where the difference starts to matter:

    net.to(device)                          accelerator.prepare(model)
    loss.backward()                         accelerator.backward(loss)
    clip_grad_norm_(net.parameters(), n)    accelerator.clip_grad_norm_(model.parameters(), n)
    torch.manual_seed(seed)                 accelerate.utils.set_seed(seed)
    device = torch.device("cuda")           accelerator.device
    print(...)                              accelerator.print(...)
"""

import argparse
from pathlib import Path

import torch

from l2c.common import data
from l2c.common import model as model_lib
from l2c.harness import cli, collect, ledger, measure, predict, report, rows

STEP = "step1_training_loop"
STEP_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    cli.common_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = measure.require_cuda()
    # Sampled before anything is allocated. Includes this process's CUDA context, so the
    # signal is the excess over an idle run: a busy GPU invalidates every timing below.
    memory_at_start = measure.memory_in_use(device)

    # Pinned, never inherited. At "highest" an fp32 matmul keeps all 24 significand bits
    # on the CUDA cores; at "high" it is truncated to TF32's 11 bits and runs on the
    # tensor cores instead — same dtypes, same memory, different kernels, and a large
    # difference in step time. A baseline that inherits this is not reproducible, and a
    # speedup measured against it means nothing. Step 2 sweeps it deliberately.
    torch.set_float32_matmul_precision("highest")

    preset = model_lib.preset_for(args.num_layers)

    net = model_lib.build_model(preset, seed=args.seed).to(device)
    net.train()
    num_params = model_lib.count_parameters(net)

    dataset = data.build_packed_dataset(model_lib.TOKENIZER, seq_len=args.seq_len)
    loader = data.build_loader(dataset, batch_size=args.batch_size, seed=args.seed)
    batches = data.endless(loader)

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate)

    def loss_from(module: torch.nn.Module, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        input_ids = batch[0].to(device, non_blocking=True)
        return module(input_ids=input_ids, labels=input_ids).loss

    def training_step(batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        loss = loss_from(net, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return loss

    # ---- phases: the memory staircase, and step 0 ---------------------------------
    measure.reset_peak(device)
    phases = measure.measure_phases(net, optimizer, next(batches), device, loss_from)
    # Between the step and the zero_grad: gradients are still live and AdamW's moments
    # now exist, which is the only moment all three buckets can be counted directly.
    inventory_states = measure.state_inventory(net, optimizer)
    optimizer.zero_grad(set_to_none=True)

    # ---- warmup -------------------------------------------------------------------
    for _ in range(args.warmup_steps):
        training_step(next(batches))

    # ---- timed --------------------------------------------------------------------
    measure.reset_peak(device)
    timer = measure.StepTimer(device, capacity=args.num_steps)
    loss_history = torch.empty(args.num_steps, device=device)

    for index in range(args.num_steps):
        batch = next(batches)
        with timer.step():
            loss = training_step(batch)
        loss_history[index] = loss.detach()

    timings = timer.finish()
    losses = loss_history.tolist()
    peak = measure.snapshot(device)

    # ---- anatomy ------------------------------------------------------------------
    with ledger.record(net, vocab_size=preset.vocab_size) as inventory:
        held = loss_from(net, next(batches))
    del held
    optimizer.zero_grad(set_to_none=True)

    # ---- predicted versus measured ------------------------------------------------
    actual = collect.run(
        num_params=num_params,
        tokens_per_step=args.batch_size * args.seq_len,
        phases=phases,
        model_states=inventory_states,
        timings=timings,
        peak=peak,
        saved=inventory,
        losses=losses,
    )

    report.publish(
        STEP,
        rows.TRAINING_LOOP,
        step_dir=STEP_DIR,
        preset=model_lib.preset_dict(preset),
        run=cli.run_args(args),
        environment=report.environment(device, memory_in_use_bytes=memory_at_start),
        actual=actual,
        headline=f"{num_params:,} parameters",
        derived=predict.exact(num_params, preset.vocab_size),
    )


if __name__ == "__main__":
    main()
