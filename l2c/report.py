"""Formatting, and the predicted-versus-measured reconciliation.

Kept apart from `train.py` so measurement never depends on presentation. Everything here
reads a result dict and returns text; nothing here touches the GPU.

Memory is presented as a ladder, because there are three different true answers to "how
much memory does this use" and conflating them is what makes memory maths feel
unpredictable:

    theory (requested bytes)
      + allocator block padding      blocks are not always split
      + non-tensor blocks            cuBLAS/cuBLASLt workspaces
    = torch.cuda.memory_allocated
      + reserved but unhanded-out    fragmentation
    = torch.cuda.memory_reserved

Only the first line is predictable from a config. The harness asserts that one exactly and
measures the rest, rather than tuning the prediction until it matches a number that
includes an allocator policy.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Iterable, Sequence

MIB = 1024 * 1024


def mib(n: float) -> float:
    return n / MIB


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]], indent: str = "  ") -> str:
    body = [[str(c) for c in r] for r in rows]
    widths = [len(h) for h in headers]
    for r in body:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))

    def line(cells: Sequence[str]) -> str:
        return (
            indent
            + "  ".join(
                c.ljust(widths[i]) if i == 0 else c.rjust(widths[i]) for i, c in enumerate(cells)
            ).rstrip()
        )

    sep = indent + "  ".join("-" * w for w in widths)
    return "\n".join([line(list(headers)), sep, *(line(r) for r in body)])


def _delta(pred: float, meas: float) -> str:
    d = meas - pred
    if d == 0:
        return "exact"
    return f"{mib(d):+.2f}" + (f" ({d / pred * 100:+.2f}%)" if pred else "")


# --------------------------------------------------------------------------------------
# theory vs measured
# --------------------------------------------------------------------------------------


def reconcile(result: dict[str, Any]) -> dict[str, Any]:
    """Predicted versus measured for every quantity the theory determines exactly."""
    pred = result["prediction"]
    p_terms = {t["label"]: t["bytes"] for t in pred["terms"]["static"]}
    measured = result["memory"].get("tensor_bytes", {})
    kinds = result.get("ledger", {}).get("totals_by_kind", {})

    rows = [
        ("parameters", p_terms["parameters (fp32 master)"], measured.get("parameters", 0)),
        ("gradients", p_terms["gradients (fp32)"], measured.get("gradients", 0)),
        (
            "optimizer state (exp_avg + exp_avg_sq)",
            p_terms["AdamW exp_avg"] + p_terms["AdamW exp_avg_sq"],
            measured.get("optimizer_state", 0),
        ),
        ("autocast weight cache", pred["weight_cache_bytes"], kinds.get("weight_cast", 0)),
        ("saved activations", pred["activation_bytes"], kinds.get("activation", 0)),
    ]
    total_pred = sum(r[1] for r in rows)
    total_meas = sum(r[2] for r in rows)
    return {
        "rows": [{"name": n, "predicted": p, "measured": m} for n, p, m in rows],
        "predicted_total": total_pred,
        "measured_total": total_meas,
        "exact": all(p == m for _, p, m in rows),
        "worst_relative_error": max(
            (abs(m - p) / p for _, p, m in rows if p > 0), default=0.0
        ),
    }


def format_reconciliation(result: dict[str, Any]) -> str:
    rec = reconcile(result)
    rows = [
        (
            r["name"],
            f"{mib(r['predicted']):.2f}",
            f"{mib(r['measured']):.2f}",
            _delta(r["predicted"], r["measured"]),
        )
        for r in rec["rows"]
    ]
    rows.append(
        (
            "persistent total",
            f"{mib(rec['predicted_total']):.2f}",
            f"{mib(rec['measured_total']):.2f}",
            _delta(rec["predicted_total"], rec["measured_total"]),
        )
    )
    verdict = "all terms exact" if rec["exact"] else f"worst term off by {rec['worst_relative_error'] * 100:.2f}%"
    return "\n".join(
        [
            f"A. theory vs measured tensor bytes, MiB  [{verdict}]",
            table(["term", "predicted", "measured", "delta"], rows),
        ]
    )


def format_ladder(result: dict[str, Any]) -> str:
    c = result["memory"].get("census")
    if not c:
        return ""
    rows = [
        ("tensor bytes (requested)", f"{mib(c['tensor_requested']):.2f}", ""),
        (
            "+ allocator block padding",
            f"{mib(c['block_padding']):.2f}",
            "large-pool blocks split only if >1 MiB would remain",
        ),
        (
            "+ non-tensor blocks",
            f"{mib(c['nontensor_blocks']):.2f}",
            "cuBLAS/cuBLASLt workspaces: "
            + ", ".join(f"{mib(s):.3f}" for s in c["nontensor_sizes"]),
        ),
        ("= memory_allocated", f"{mib(c['allocated']):.2f}", ""),
        (
            "+ reserved, not handed out",
            f"{mib(c['reserved'] - c['allocated']):.2f}",
            "caching-allocator slack and fragmentation",
        ),
        ("= memory_reserved", f"{mib(c['reserved']):.2f}", ""),
    ]
    out = ["B. from tensor bytes to what the GPU reports, at rest, MiB", table(["", "MiB", "note"], rows)]
    if c["unexplained"]:
        out.append(f"  WARNING unexplained {mib(c['unexplained']):.4f} MiB")
    return "\n".join(out)


def format_phases(result: dict[str, Any]) -> str:
    """Per-phase peaks: which phase sets the peak, and by how much.

    Without this, a peak number invites the wrong fix. An optimiser-step peak is cured by
    `fused=True`; a backward peak needs checkpointing or a smaller batch.
    """
    steady = result["memory"].get("steady_step")
    if not steady:
        return ""
    peaks, marks = steady["peaks"], steady["marks"]
    rows = [
        (
            "forward",
            f"{mib(peaks['forward']):.2f}",
            f"{mib(marks['after_forward'] - marks['before_forward']):+.2f}",
        ),
        (
            "backward",
            f"{mib(peaks['backward']):.2f}",
            f"{mib(marks['after_backward'] - marks['after_forward']):+.2f}",
        ),
        (
            "optimizer (incl. clip)",
            f"{mib(peaks['optimizer']):.2f}",
            f"{mib(marks['after_optimizer'] - marks['after_backward']):+.2f}",
        ),
    ]
    return "\n".join(
        [
            "C. per-phase peak and net allocation, MiB",
            table(["phase", "peak allocated", "net change"], rows),
            f"  steady-state peak over the timed region: "
            f"{mib(result['memory']['peak_allocated']):.2f} MiB",
        ]
    )


def format_activation_buckets(result: dict[str, Any]) -> str:
    pred = result["prediction"]["activation_buckets"]
    meas = result.get("ledger", {}).get("activation_buckets", {})
    keys = sorted(set(pred) | set(meas), key=lambda k: -max(pred.get(k, 0), meas.get(k, 0)))
    rows = [
        (
            k,
            f"{mib(pred.get(k, 0)):.2f}",
            f"{mib(meas.get(k, 0)):.2f}",
            _delta(pred.get(k, 0), meas.get(k, 0)),
        )
        for k in keys
    ]
    return "\n".join(
        [
            "D. saved activations by module, MiB",
            table(["bucket", "predicted", "measured", "delta"], rows),
        ]
    )


# --------------------------------------------------------------------------------------
# single run
# --------------------------------------------------------------------------------------


def format_run(result: dict[str, Any]) -> str:
    cfg, env, t = result["config"], result["env"], result["timing"]
    m, d = cfg["model"], cfg["data"]
    loss = result["loss"]
    head = [
        f"=== {cfg['precision']}  seed={cfg['seed']}  B={d['batch_size']} S={d['seq_len']} ===",
        f"  {env['gpu']}  sm{env['capability']}  torch {env['torch']}",
        f"  model    {result['param_count'] / 1e6:.2f}M params "
        f"(L={m['n_layer']} d={m['d_model']} ff={m['d_ff']} V={m['vocab_size']}), "
        f"{result['matmul_param_count'] / 1e6:.2f}M in Linear",
        f"  kernels  sdpa={env['sdpa_backend']}  fp32_matmul={env['fp32_matmul_precision']}"
        f"  adamw={cfg['optim']['impl']}  grad_clip={cfg['optim']['grad_clip']}",
        "",
        "timing",
        f"  median {t['median_ms']:.2f} ms/step   p10 {t['p10_ms']:.2f}   p90 {t['p90_ms']:.2f}"
        f"   IQR {t['iqr_ms']:.2f}",
        f"  {t['tokens_per_s'] / 1e3:.1f}k tokens/s   {t['steps_per_s']:.2f} steps/s"
        f"   busy_fraction {t['busy_fraction']:.3f}",
        "",
        "loss",
        f"  step 0 {loss[0]:.4f}   final {loss[-1]:.4f}   "
        f"ln(V) = {math.log(m['vocab_size']):.4f}",
    ]
    if result.get("loss_scale"):
        scales = [s for s in result["loss_scale"] if s > 0]
        if scales:
            head.append(f"  loss scale {scales[0]:.0f} -> {scales[-1]:.0f}")
    parts = ["\n".join(head), "", format_reconciliation(result)]
    for section in (format_ladder(result), format_phases(result), format_activation_buckets(result)):
        if section:
            parts += ["", section]
    return "\n".join(parts)


# --------------------------------------------------------------------------------------
# cross-run comparison
# --------------------------------------------------------------------------------------


def loss_agreement(a: list[float], b: list[float]) -> dict[str, float]:
    dif = [abs(x - y) for x, y in zip(a, b)]
    return {
        "max_abs": max(dif),
        "mean_abs": statistics.fmean(dif),
        "final_abs": dif[-1],
        "final_a": a[-1],
        "final_b": b[-1],
        "bitwise_identical": max(dif) == 0.0,
    }
