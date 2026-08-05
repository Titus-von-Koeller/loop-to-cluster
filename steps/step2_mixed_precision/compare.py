"""Drive all four precision arms and reconcile them against one another.

The comparison lives here rather than spanning two step directories on purpose. A delta
is only meaningful if both sides were measured the same way, and steps are allowed to
drift — so any claim of the form "X is faster than Y" is made by a single script that ran
both, with one seed, one dataset and one measurement path.

Each arm is a separate process (see `l2c.harness.runner`): `max_memory_allocated` is a
high-water mark for the life of a process, so two arms in one process would have the
second inherit the first's peak. Results are cached by flags, so re-running this to
reformat the table costs nothing.

    python steps/step2_mixed_precision/compare.py
    python steps/step2_mixed_precision/compare.py --force
"""

import argparse
import itertools
import statistics
from pathlib import Path

from l2c.harness import report, runner
from l2c.harness.precision import Precision

STEP_DIR = Path(__file__).resolve().parent
WORKER = STEP_DIR / "train.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-run every arm")
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    return parser.parse_args()


def column(results: dict[Precision, dict], key: str) -> dict[Precision, float]:
    return {arm: result["actual"][key] for arm, result in results.items()}


def main() -> None:
    args = parse_args()
    shared = [
        "--num-steps",
        str(args.num_steps),
        "--batch-size",
        str(args.batch_size),
        "--seq-len",
        str(args.seq_len),
    ]

    results = {
        arm: runner.run_worker(
            WORKER, ["--precision", str(arm), *shared], force=args.force, quiet=True
        )
        for arm in Precision
    }
    prediction = report.load_prediction(STEP_DIR)
    baseline = Precision.FP32

    tokens = args.batch_size * args.seq_len
    print(f"\nfour precision arms, {tokens:,} tokens/step, {args.num_steps} timed steps\n")

    def show(title: str, key: str, fmt: str, *, relative: bool = False) -> None:
        values = column(results, key)
        cells = "".join(f"{values[arm]:>13{fmt}}" for arm in Precision)
        print(f"{title:<26}{cells}")
        if relative:
            base = values[baseline]
            ratios = "".join(f"{base / values[arm]:>12.2f}x" for arm in Precision)
            print(f"{'  vs fp32':<26}{ratios}")

    header = "".join(f"{arm!s:>13}" for arm in Precision)
    print(f"{'quantity':<26}{header}")
    print("-" * (26 + 13 * len(Precision)))
    show("model states (MiB)", "model_states_mib", ",.1f")
    show("weight cache (MiB)", "weight_cache_mib", ",.1f")
    show("weight cache tensors", "weight_cache_tensors", ",.0f")
    show("activations (MiB)", "activations_excl_cache_mib", ",.1f", relative=True)
    show("  of which still fp32", "saved_still_fp32_mib", ",.1f")
    show("  vocab-sized logits", "saved_logits_mib", ",.1f")
    show("peak allocated (MiB)", "peak_mib", ",.1f", relative=True)
    show("median step (ms)", "median_step_ms", ",.1f", relative=True)
    show("tokens/sec", "tokens_per_second", ",.0f")
    show("initial loss", "initial_loss", ",.4f")
    show("final loss", "final_loss", ",.4f")
    show("updates skipped", "steps_skipped", ",.0f")
    show("grad scale", "final_grad_scale", ",.0f")

    print("\n--- predicted versus measured ------------------------------------------")
    rows = [
        report.Row(
            "model states (MiB)",
            prediction.get("model_states_mib"),
            results[arm]["actual"]["model_states_mib"],
            ",.1f",
            0.1,
        )
        for arm in (baseline,)
    ]
    for arm in Precision:
        suffix = str(arm)
        rows += [
            report.Row(
                f"weight cache {suffix} (MiB)",
                prediction.get(f"weight_cache_mib_{suffix}", 0.0),
                results[arm]["actual"]["weight_cache_mib"],
                ",.1f",
                0.1,
            ),
            report.Row(
                f"activations {suffix} (MiB)",
                prediction.get(f"activations_mib_{suffix}"),
                results[arm]["actual"]["activations_excl_cache_mib"],
                ",.1f",
                15.0,
            ),
            report.Row(
                f"peak {suffix} (MiB)",
                prediction.get(f"peak_mib_{suffix}"),
                results[arm]["actual"]["peak_mib"],
                ",.1f",
                10.0,
            ),
            report.Row(
                f"median step {suffix} (ms)",
                prediction.get(f"median_step_ms_{suffix}"),
                results[arm]["actual"]["median_step_ms"],
                ",.1f",
                20.0,
            ),
        ]
    print(report.table(rows))

    print("\n--- is the loss curve unchanged? ---------------------------------------")
    curves = {arm: result["actual"]["loss_curve"] for arm, result in results.items()}
    base_curve = curves[baseline]
    # The noise floor: consecutive-step variation within the baseline itself. A gap
    # between arms only means something if it exceeds the wobble already present in one.
    floor = statistics.stdev(
        second - first for first, second in itertools.pairwise(base_curve)
    )
    print(f"baseline step-to-step stdev (the noise floor): {floor:.4f}\n")
    print(f"{'arm':<10}{'mean |diff|':>14}{'max |diff|':>14}{'vs floor':>12}")
    for arm, curve in curves.items():
        diffs = [abs(a - b) for a, b in zip(base_curve, curve, strict=True)]
        mean_diff = statistics.fmean(diffs)
        verdict = "within" if max(diffs) < floor else "EXCEEDS"
        print(f"{arm!s:<10}{mean_diff:>14.4f}{max(diffs):>14.4f}{verdict:>12}")


if __name__ == "__main__":
    main()
