"""Grid orchestration: turn a list of configs into a list of result dicts.

One subprocess per run, for the reasons in `cli.py`. Results are cached on disk and keyed
by the flags that produced them, so re-running an experiment to fix a table costs nothing
and the numbers in a report cannot silently drift from the numbers that were measured.

`--force` re-runs everything; deleting one JSON re-runs one cell.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time
from typing import Any, Iterable

Flags = dict[str, Any]


def tag_of(flags: Flags) -> str:
    """Filename-safe identity of a run. Two runs with the same tag are the same run."""
    parts = [f"{k}{v}" for k, v in sorted(flags.items()) if v is not None]
    return "-".join(parts).replace("_", "").replace(".", "p")


def _argv(flags: Flags, out: pathlib.Path) -> list[str]:
    argv = [sys.executable, "-m", "l2c.cli", "--quiet", "--out", str(out)]
    for k, v in flags.items():
        if v is None or v is False:
            continue
        flag = "--" + k.replace("_", "-")
        argv += [flag] if v is True else [flag, str(v)]
    return argv


def run_one(flags: Flags, out_dir: pathlib.Path, force: bool = False) -> dict[str, Any]:
    out = out_dir / f"{tag_of(flags)}.json"
    if out.exists() and not force:
        return json.loads(out.read_text())
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    proc = subprocess.run(
        _argv(flags, out), capture_output=True, text=True, cwd=pathlib.Path(__file__).parent.parent
    )
    if proc.returncode != 0:
        raise RuntimeError(f"run {tag_of(flags)} failed:\n{proc.stderr[-4000:]}")
    print(f"  ran {tag_of(flags):48s} {time.perf_counter() - started:5.1f}s", flush=True)
    return json.loads(out.read_text())


def run_grid(
    grid: Iterable[Flags], out_dir: pathlib.Path, force: bool = False
) -> list[dict[str, Any]]:
    grid = list(grid)
    print(f"grid: {len(grid)} runs -> {out_dir}", flush=True)
    cached = sum(1 for f in grid if (out_dir / f"{tag_of(f)}.json").exists() and not force)
    if cached:
        print(f"  {cached} cached, {len(grid) - cached} to run", flush=True)
    return [run_one(f, out_dir, force) for f in grid]


def select(results: list[dict[str, Any]], **where: Any) -> list[dict[str, Any]]:
    """Filter results by config field. Nested fields use dotted keys: `data.batch_size`."""

    def get(result: dict[str, Any], key: str) -> Any:
        node: Any = result["config"]
        for part in key.split("."):
            node = node[part]
        return node

    return [r for r in results if all(get(r, k) == v for k, v in where.items())]


def one(results: list[dict[str, Any]], **where: Any) -> dict[str, Any]:
    hits = select(results, **where)
    if len(hits) != 1:
        raise LookupError(f"expected exactly 1 result for {where}, got {len(hits)}")
    return hits[0]
