"""Run exactly one config and write one JSON file.

One run per process, on purpose. A fresh process gets a fresh caching allocator, fresh
cuBLAS workspaces and no autotuning state carried over from a previous precision mode, so
runs cannot contaminate each other's timing or memory numbers. It is also the shape that
survives the move to `torchrun` later.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from .config import config_from_args
from .report import format_run
from .train import run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="l2c", description=__doc__)
    p.add_argument(
        "--precision", default="fp32", choices=["fp32", "tf32", "amp_bf16", "amp_fp16"]
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=50, help="timed steps")
    p.add_argument("--warmup", type=int, default=10, help="untimed steps before measuring")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--d-ff", type=int, default=None, help="default 4*d_model")
    p.add_argument("--optim-impl", default="foreach", choices=["foreach", "fused", "single"])
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--no-anatomy", action="store_true", help="skip memory marks and ledger")
    p.add_argument("--snapshot", default=None, help="write an allocator snapshot .pickle here")
    p.add_argument("--out", default=None, help="write result JSON here")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    result = run(cfg)

    if args.out:
        path = pathlib.Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
    if not args.quiet:
        print(format_run(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
