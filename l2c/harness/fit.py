"""Fit a line through a sweep, so a formula is validated rather than a point.

One number matching a prediction is a coincidence. Six points on a line is a formula.
Sweeping `num_hidden_layers` is the cleanest sweep available, because every quantity of
interest is *linear* in depth:

    num_params = embedding + final_norm + num_layers * per_layer_params

So the fitted slope is the per-layer parameter cost and the intercept is everything
outside the blocks. One sweep, two independent checks against a hand derivation, and a
residual that reveals any term left out of it.

Run after `steps/sweep.sh`:

    python -m l2c.harness.fit --step step1_training_loop --x num_hidden_layers \\
        --y num_params peak_mib activations_mib
"""

import argparse
from dataclasses import dataclass

import numpy as np

from l2c.harness.runs import load_all


@dataclass(frozen=True, slots=True)
class Fit:
    slope: float
    intercept: float
    r_squared: float
    num_points: int

    def predict(self, x: float) -> float:
        return self.slope * x + self.intercept


def least_squares(xs: list[float], ys: list[float]) -> Fit:
    """Ordinary least squares, plus R^2 so a bad fit cannot pass as a good one."""
    if len(xs) < 2:
        raise ValueError(f"need at least two points to fit a line, got {len(xs)}")
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    total = y - y.mean()
    r_squared = 1.0 if total.dot(total) == 0 else 1 - residual.dot(residual) / total.dot(total)
    return Fit(float(slope), float(intercept), float(r_squared), len(xs))


def pluck(entry: dict, key: str) -> float | None:
    """Find `key` in a result entry, wherever it lives.

    Config knobs sit under `preset` or `run` and measurements under `actual`, so the
    caller can name an axis without knowing which section holds it.
    """
    for section in ("preset", "run", "actual"):
        value = entry.get(section, {}).get(key)
        if isinstance(value, int | float):
            return float(value)
    return None


def fit_axis(entries: list[dict], x_key: str, y_key: str) -> Fit:
    pairs = [(pluck(e, x_key), pluck(e, y_key)) for e in entries]
    usable = [(x, y) for x, y in pairs if x is not None and y is not None]
    if not usable:
        raise ValueError(f"no entry carries both {x_key!r} and {y_key!r}")
    xs, ys = zip(*sorted(usable), strict=True)
    return least_squares(list(xs), list(ys))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", required=True)
    parser.add_argument("--x", default="num_hidden_layers")
    parser.add_argument("--y", nargs="+", required=True)
    args = parser.parse_args()

    entries = load_all(args.step)
    if not entries:
        raise SystemExit(f"no results recorded for step {args.step!r}")

    revisions = {e.get("commit") for e in entries}
    if len(revisions) > 1:
        named = ", ".join(sorted(str(r) for r in revisions))
        print(f"WARNING: fitting across {len(revisions)} revisions ({named}).")
        print("A change to the training loop moves these points for reasons the fit")
        print("cannot see. Re-run the sweep before trusting the residual.\n")

    print(f"{len(entries)} runs of {args.step}, fitting against {args.x}\n")
    header = f"{'quantity':<22}{'slope':>18}{'intercept':>18}{'R^2':>10}{'points':>8}"
    print(header)
    print("-" * len(header))
    for y_key in args.y:
        fit = fit_axis(entries, args.x, y_key)
        print(
            f"{y_key:<22}{fit.slope:>18,.2f}{fit.intercept:>18,.2f}"
            f"{fit.r_squared:>10.5f}{fit.num_points:>8}"
        )
    print(f"\nslope = marginal cost per unit of {args.x}")
    print("intercept = everything that does not scale with it")


if __name__ == "__main__":
    main()
