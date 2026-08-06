"""The knobs every step shares, declared once.

A step's argument list should read as its subject. When each file repeats the same
eight knobs, the one flag that belongs to the lesson is buried among them; declared
here, a step adds only what it is about and `--precision` stands out as step 2's.

`run_args` defines what is recorded alongside a measurement. Every argument a step
takes steers the run, so every one of them is recorded and every one of them is part
of the key a result is stored under.

In accelerate the counterpart is `accelerate.commands.launch`, whose parser is
assembled from argument groups shared across `launch`, `estimate-memory` and the
config commands.
"""

import argparse
from typing import Any


def common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the knobs shared by every step. Step-specific flags are declared by the step."""
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    # The sweep knob. Depth is the right axis: parameters are linear in it, so a fit
    # returns the per-layer cost as its slope and the embedding table as its intercept.
    parser.add_argument("--num-layers", type=int, default=None)
    return parser


def run_args(args: argparse.Namespace) -> dict[str, Any]:
    """The arguments recorded with a result, in a form JSON round-trips unchanged."""
    return {
        name: value if _is_native(value) else str(value)
        for name, value in sorted(vars(args).items())
    }


def _is_native(value: object) -> bool:
    return value is None or isinstance(value, str | int | float | bool)
