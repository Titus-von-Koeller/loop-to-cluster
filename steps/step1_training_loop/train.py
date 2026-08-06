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
import json
from pathlib import Path

import torch

from l2c.common import data
from l2c.common import model as model_lib
from l2c.harness import cli, ledger, measure, predict, report

STEP = "step1_training_loop"
STEP_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    cli.common_args(parser)
    parser.add_argument("--json-out", type=Path, default=None)
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
    states = predict.model_states(num_params, optimizer="adamw")
    prediction = report.load_prediction(STEP_DIR)
    tokens_per_step = args.batch_size * args.seq_len

    actual = {
        "num_params": num_params,
        "bytes_per_param": inventory_states.total_bytes / num_params,
        "initial_loss": phases.initial_loss,
        "final_loss": losses[-1],
        "params_mib": measure.mib(inventory_states.param_bytes),
        "gradients_mib": measure.mib(inventory_states.grad_bytes),
        "optimizer_states_mib": measure.mib(inventory_states.optimizer_bytes),
        "model_states_mib": measure.mib(inventory_states.total_bytes),
        "model_states_allocated_mib": measure.mib(phases.after_step),
        "block_padding_mib": measure.mib(phases.after_step - inventory_states.total_bytes),
        "activations_mib": measure.mib(phases.activations),
        "peak_mib": measure.mib(peak.peak_allocated),
        "peak_reserved_mib": measure.mib(peak.peak_reserved),
        "allocator_overhead_mib": measure.mib(peak.allocator_overhead),
        "median_step_ms": timings.median_ms,
        "p10_step_ms": timings.p10_ms,
        "p90_step_ms": timings.p90_ms,
        "steps_per_second": timings.steps_per_second,
        "tokens_per_second": timings.tokens_per_second(tokens_per_step),
        "saved_tensors": inventory.summary(),
        "loss_curve": losses,
    }

    # Tolerances say how exact each claim is. Model states are arithmetic and must match
    # to a rounding error; the activation and throughput rows are first guesses.
    rows = [
        report.Row("parameters", prediction.get("num_params"), num_params, ",.0f", 0.0),
        report.Row(
            "bytes/param",
            prediction.get("bytes_per_param"),
            actual["bytes_per_param"],
            ",.2f",
        ),
        report.Row(
            "initial loss",
            predict.expected_initial_loss(preset.vocab_size),
            phases.initial_loss,
            ",.4f",
        ),
        report.Row(
            "params (MiB)", measure.mib(states.param_bytes), actual["params_mib"], ",.1f", 0.1
        ),
        report.Row(
            "gradients (MiB)",
            measure.mib(states.grad_bytes),
            actual["gradients_mib"],
            ",.1f",
            0.1,
        ),
        report.Row(
            "optim states (MiB)",
            measure.mib(states.optimizer_bytes),
            actual["optimizer_states_mib"],
            ",.1f",
            0.1,
        ),
        report.Row(
            "model states (MiB)",
            measure.mib(states.total_bytes),
            actual["model_states_mib"],
            ",.1f",
            0.1,
        ),
        report.Row("block padding (MiB)", None, actual["block_padding_mib"], ",.1f"),
        report.Row(
            "activations (MiB)",
            prediction.get("activations_mib"),
            actual["activations_mib"],
            ",.1f",
            10.0,
        ),
        report.Row(
            "peak allocated (MiB)",
            prediction.get("peak_mib"),
            actual["peak_mib"],
            ",.1f",
            10.0,
        ),
        report.Row(
            "steps/sec",
            prediction.get("steps_per_second"),
            actual["steps_per_second"],
            ",.2f",
            20.0,
        ),
    ]

    print(f"\n{STEP}   {num_params:,} parameters   {tokens_per_step:,} tokens/step\n")
    print(report.table(rows))

    print(
        f"\nstep time   median {timings.median_ms:.1f} ms   "
        f"p10 {timings.p10_ms:.1f}   p90 {timings.p90_ms:.1f}"
    )
    print(f"throughput  {actual['tokens_per_second']:,.0f} tokens/sec")
    print(f"loss        {phases.initial_loss:.4f} -> {losses[-1]:.4f}")
    print(
        f"memory      peak allocated {actual['peak_mib']:,.1f} MiB   "
        f"reserved {actual['peak_reserved_mib']:,.1f} MiB   "
        f"allocator overhead {actual['allocator_overhead_mib']:,.1f} MiB"
    )

    print("\nsaved for backward:")
    for category, byte_count in inventory.by_category().items():
        print(f"  {category:<14}{measure.mib(byte_count):>10,.1f} MiB")
    print("  by dtype:")
    for dtype, byte_count in inventory.by_dtype().items():
        print(f"    {dtype:<12}{measure.mib(byte_count):>10,.1f} MiB")

    environment = report.environment(device, memory_in_use_bytes=memory_at_start)
    run = cli.run_args(args)
    path = report.record(
        STEP,
        preset=model_lib.preset_dict(preset),
        run=run,
        environment=environment,
        predicted=prediction,
        actual=actual,
    )
    print(f"\nappended to {path}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {
                    "step": STEP,
                    "preset": model_lib.preset_dict(preset),
                    "run": run,
                    "environment": environment,
                    "predicted": prediction,
                    "actual": actual,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
