"""Generate the wiki figures.

Run: cd /home/titus/src/loop-to-cluster && pixi run python figures/make_figures.py

Every figure is labelled measured or schematic in its footnote. Measured
figures use numbers verified in verify_params.py / verify_facts.py; schematic
figures illustrate a shape and carry no data claim.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from figstyle import (
    BASELINE,
    BLUE,
    BLUE_ORDINAL,
    GOLD,
    INK,
    INK_2,
    MUTED,
    NEUTRAL,
    ROSE,
    SURFACE,
    despine,
    note,
    save,
    title,
    use_house_style,
)
from matplotlib.patches import Rectangle

use_house_style()
OUT = os.path.dirname(os.path.abspath(__file__)) + "/out"
os.makedirs(OUT, exist_ok=True)

# Verified in figures/verify_params.py
P_TOTAL = 134_515_008
P_MLP, P_EMBED, P_ATTN, P_NORM = 79_626_240, 28_311_552, 26_542_080, 35_136
P_TEXTBOOK = 147_750_912
MIB = 1024**2


# ---------------------------------------------------------------- 1
def fig_param_breakdown():
    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    rows = [
        ("MLP  (3 gated matrices)", P_MLP),
        ("Embedding  (tied)", P_EMBED),
        ("Attention  (GQA, 3 KV heads)", P_ATTN),
        ("RMSNorm scales", P_NORM),
    ]
    labels = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    colors = [BLUE_ORDINAL[3], BLUE_ORDINAL[2], BLUE_ORDINAL[1], BLUE_ORDINAL[0]][::-1]
    y = np.arange(len(vals))
    ax.barh(y, vals, color=colors, height=0.62, edgecolor=SURFACE, linewidth=2)
    for i, v in enumerate(vals):
        pct = v / P_TOTAL
        lab = f"{v:,}   ({pct:.1%})" if pct > 0.001 else f"{v:,}   ({pct:.2%})"
        ax.text(v + P_TOTAL * 0.012, i, lab, va="center", fontsize=10, color=INK_2)
    ax.set_yticks(y, labels, fontsize=10.5)
    ax.set_xlim(0, P_TOTAL * 0.86)
    ax.set_xticks([])
    ax.grid(False)
    despine(ax, keep=())
    title(
        ax,
        "Where 134,515,008 parameters actually live",
        "SmolLM2-135M. The MLP is nearly three fifths of the model.",
    )
    note(
        fig,
        "Measured: sum(p.numel()) over the constructed model, deduplicated for the tied embedding.",
    )
    save(fig, f"{OUT}/01_param_breakdown.png")


# ---------------------------------------------------------------- 2
def fig_textbook_error():
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    names = ["Textbook formula\n12·L·d² + V·d", "Actual\nsum(p.numel())"]
    vals = [P_TEXTBOOK, P_TOTAL]
    bars = ax.bar(
        names, vals, color=[NEUTRAL, BLUE], width=0.44, edgecolor=SURFACE, linewidth=2
    )
    for b, v in zip(bars, vals, strict=True):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 5.0e6,
            f"{v:,}",
            ha="center",
            fontsize=11,
            color=INK,
            fontweight="bold",
        )
    # Dashed guides at both bar tops bound the gap the caption quantifies. The span is
    # too short for arrowheads, which would overlap each other at this scale.
    for y in (P_TOTAL, P_TEXTBOOK):
        ax.plot([0.24, 1.30], [y, y], color=ROSE, lw=0.9, ls=(0, (3, 3)), zorder=1)
    ax.text(
        1.38,
        (P_TOTAL + P_TEXTBOOK) / 2,
        f"+9.84%\n{P_TEXTBOOK - P_TOTAL:,} too many\n← entirely GQA",
        fontsize=10,
        color=ROSE,
        va="center",
        ha="left",
        fontweight="bold",
    )
    ax.set_xlim(-0.55, 2.6)
    ax.set_ylim(0, P_TEXTBOOK * 1.16)
    ax.set_yticks([])
    ax.grid(False)
    despine(ax, keep=("bottom",))
    title(
        ax,
        "The textbook parameter formula is wrong for modern models",
        "It assumes multi-head attention. Grouped-query attention makes K and V a third as wide.",
    )
    note(
        fig, "Measured. The gated-MLP term happens to be exact here: 3·d·1536 = 8d² for d=576."
    )
    save(fig, f"{OUT}/02_textbook_error.png")


# ---------------------------------------------------------------- 3
def fig_memory_buckets():
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    per = P_TOTAL * 4 / MIB
    names = [
        "Parameters\nfp32",
        "Gradients\nfp32",
        "Adam m\nfirst moment",
        "Adam v\nsecond moment",
    ]
    left = 0.0
    for i, n in enumerate(names):
        ax.barh(
            [0],
            [per],
            left=left,
            color=BLUE_ORDINAL[i],
            height=0.5,
            edgecolor=SURFACE,
            linewidth=2.5,
        )
        ax.text(
            left + per / 2,
            0,
            f"{n}\n4 B/param",
            ha="center",
            va="center",
            fontsize=9.5,
            color="white" if i >= 2 else INK,
            fontweight="bold",
        )
        left += per
    ax.text(
        left / 2,
        0.44,
        f"16 bytes per parameter  ·  {left:,.0f} MiB  ≈  {left / 1024:.2f} GiB",
        ha="center",
        fontsize=11.5,
        color=INK,
        fontweight="bold",
    )
    ax.set_xlim(0, left * 1.005)
    ax.set_ylim(-0.42, 0.72)
    ax.set_yticks([])
    ax.set_xlabel("MiB")
    ax.grid(False)
    despine(ax, keep=())
    title(
        ax,
        "The four model-state buckets",
        "Fixed by the model and optimizer. Activations sit on top of this.",
    )
    note(
        fig,
        "Measured: 134,515,008 params x 4 B. Mixed precision does not move any of these four.",
    )
    save(fig, f"{OUT}/03_memory_buckets.png")


# ---------------------------------------------------------------- 4
def fig_optimizer_cost():
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    names = ["SGD", "SGD\n+momentum", "AdamW", "8-bit AdamW\n(bitsandbytes)"]
    states = [0, 4, 8, 2]
    base = 8
    ax.bar(
        names,
        [base] * 4,
        color=BLUE_ORDINAL[1],
        width=0.5,
        edgecolor=SURFACE,
        linewidth=2,
        label="params + grads (8 B)",
    )
    ax.bar(
        names,
        states,
        bottom=[base] * 4,
        color=GOLD,
        width=0.5,
        edgecolor=SURFACE,
        linewidth=2,
        label="optimizer state",
    )
    for i, s in enumerate(states):
        ax.text(
            i,
            base + s + 0.35,
            f"{base + s} B",
            ha="center",
            fontsize=11,
            color=INK,
            fontweight="bold",
        )
    ax.set_ylabel("bytes per parameter")
    ax.set_ylim(0, 19)
    ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(0, 1.02))
    despine(ax, keep=("bottom",))
    title(
        ax,
        "The optimizer is the cheapest memory lever you have",
        "Swapping AdamW for 8-bit AdamW removes six bytes per parameter without touching the model.",
    )
    note(
        fig, "fp32 parameters and gradients throughout. Adam's two moments are the difference."
    )
    save(fig, f"{OUT}/04_optimizer_cost.png")


# ---------------------------------------------------------------- 5
def fig_float_formats():
    """Bit counts go inside the boxes; field names go in a key.

    An earlier version put 'exponent · 8 bits' inside an eight-unit box and
    the text overflowed into the neighbouring sign field.
    """
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    fmts = [
        ("fp32", 1, 8, 23, "32 bits"),
        ("fp16", 1, 5, 10, "16 bits"),
        ("bf16", 1, 8, 7, "16 bits"),
    ]
    for row, (name, s, e, m, tot) in enumerate(fmts):
        y = 2 - row
        x = 0
        for width, color in ((s, INK_2), (e, BLUE), (m, GOLD)):
            ax.add_patch(
                Rectangle(
                    (x, y - 0.29),
                    width,
                    0.58,
                    facecolor=color,
                    edgecolor=SURFACE,
                    linewidth=2.5,
                )
            )
            if width > 1:
                ax.text(
                    x + width / 2,
                    y,
                    str(width),
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="white",
                    fontweight="bold",
                )
            x += width
        ax.text(
            -0.5, y, name, ha="right", va="center", fontsize=12.5, fontweight="bold", color=INK
        )
        ax.text(x + 0.6, y, tot, ha="left", va="center", fontsize=10, color=MUTED)

    ax.annotate(
        "",
        xy=(9, 2.62),
        xytext=(1, 2.62),
        arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.7),
    )
    ax.text(
        5,
        2.80,
        "same exponent width  →  same range",
        ha="center",
        fontsize=10,
        color=BLUE,
        fontweight="bold",
    )
    for xv in (1, 9):
        ax.plot([xv, xv], [-0.05, 2.56], color=BLUE, lw=1.1, ls=(0, (3, 3)), zorder=0)

    for i, (color, lab) in enumerate(
        ((INK_2, "sign"), (BLUE, "exponent → range"), (GOLD, "mantissa → precision"))
    ):
        lx = i * 12.0
        ax.add_patch(Rectangle((lx, -0.78), 1.1, 0.36, facecolor=color, edgecolor="none"))
        ax.text(lx + 1.7, -0.60, lab, ha="left", va="center", fontsize=10, color=INK_2)

    ax.set_xlim(-6, 39)
    ax.set_ylim(-1.5, 3.15)
    ax.axis("off")
    ax.text(
        -6,
        -1.18,
        "bf16 keeps fp32's exponent and pays for it out of the mantissa. "
        "fp16 does the opposite — and falls off both ends of the range.",
        fontsize=9.8,
        color=INK_2,
        ha="left",
    )
    fig.suptitle(
        "Two different bets on how to spend 16 bits",
        x=0.052,
        y=1.02,
        ha="left",
        fontsize=13.5,
        fontweight="bold",
        color=INK,
    )
    save(fig, f"{OUT}/05_float_formats.png")


# ---------------------------------------------------------------- 6
def fig_fp16_underflow():
    fig, ax = plt.subplots(figsize=(8.0, 3.9))
    rng = np.random.default_rng(0)
    g = 10 ** rng.normal(-7.2, 1.05, 200_000)
    bins = np.logspace(-12, -1, 220)
    ax.hist(g, bins=bins, color=BLUE_ORDINAL[1], edgecolor="none")
    fp16_normal, fp16_sub = 6.1e-5, 5.96e-8
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.42)  # headroom so the threshold labels stay inside the axes
    ax.axvline(fp16_normal, color=ROSE, lw=2)
    ax.axvline(fp16_sub, color=ROSE, lw=2, ls=(0, (4, 3)))
    ax.axvspan(1e-12, fp16_sub, color=ROSE, alpha=0.11)

    # Threshold labels live above the plot with leader lines, so they never
    # sit on top of the distribution.
    for xv, lab, ha in (
        (fp16_sub, "smallest subnormal\n6.0e-8", "right"),
        (fp16_normal, "smallest normal\n6.1e-5", "left"),
    ):
        ax.annotate(
            lab,
            xy=(xv, top),
            xytext=(xv * (0.55 if ha == "right" else 1.8), top * 1.14),
            fontsize=9.5,
            color=ROSE,
            fontweight="bold",
            ha=ha,
            va="bottom",
            annotation_clip=False,
            arrowprops=dict(arrowstyle="-", color=ROSE, lw=1.1),
        )
    ax.text(
        3e-10,
        top * 0.55,
        "flushed to zero\nunder fp16",
        fontsize=10.5,
        color=ROSE,
        ha="center",
        fontweight="bold",
    )
    ax.annotate(
        "",
        xy=(3e-4, top * 0.13),
        xytext=(4e-8, top * 0.13),
        arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=2.4, mutation_scale=18),
    )
    ax.text(
        1.1e-6,
        top * 0.175,
        "loss scaling multiplies every gradient by S",
        fontsize=9.8,
        color="#7d5f10",
        ha="center",
        fontweight="bold",
    )
    ax.set_xscale("log")
    ax.set_xlim(1e-12, 1e-1)
    ax.set_yticks([])
    ax.set_ylabel("how many gradients")
    ax.set_xlabel("gradient magnitude")
    despine(ax, keep=("bottom",))
    title(
        ax,
        "Why fp16 needs loss scaling and bf16 does not",
        "bf16's smallest normal is ~1.2e-38 — far off this chart to the left. Nothing here underflows.",
    )
    note(
        fig,
        "Schematic: the distribution is illustrative, not measured, so read the shape and not the area. "
        "The two fp16 thresholds are exact.",
    )
    save(fig, f"{OUT}/06_fp16_underflow.png")


# ---------------------------------------------------------------- 7
def fig_loss_anatomy():
    """Four panels sharing one y-axis and one ln(V) reference line.

    Layout is set explicitly rather than by tight_layout: the suptitle, the
    panel headings and the per-panel captions all need reserved bands, and
    letting them auto-place collided the heading into the subtitle.
    """
    lnV = 10.8027
    fig, axes = plt.subplots(1, 4, figsize=(11.6, 4.3), sharey=True)
    fig.subplots_adjust(top=0.70, bottom=0.30, left=0.075, right=0.985, wspace=0.14)

    x = np.linspace(0, 1, 400)
    healthy = lnV * np.exp(-x * 6.2) + 2.1 + 0.045 * np.sin(x * 60) * np.exp(-x * 3)

    spike = healthy.copy()
    spike[150:168] += np.array(
        [
            0,
            1.4,
            3.2,
            4.1,
            3.4,
            2.5,
            1.8,
            1.2,
            0.8,
            0.5,
            0.33,
            0.2,
            0.13,
            0.08,
            0.05,
            0.03,
            0.02,
            0.01,
        ]
    )
    plateau = np.full_like(x, lnV) - 0.05 * x
    diverge = healthy.copy()
    diverge[215:] = np.nan
    tail = np.full_like(x, np.nan)
    tail[214:262] = diverge[214] + np.exp(np.linspace(-1.2, 2.35, 48)) * 1.35

    panels = [
        (
            "Healthy",
            healthy,
            BLUE,
            None,
            "Starts at ln(V), drops steeply,\nthen a long slow tail. Boring is correct.",
        ),
        (
            "Benign spike",
            spike,
            GOLD,
            None,
            "One bad batch. Recovers on its own\nwithin a few hundred steps.",
        ),
        (
            "Plateau",
            plateau,
            ROSE,
            "stuck at ln(V)",
            "Never left the starting value.\nLR near zero, or no gradient flowing.",
        ),
        (
            "Divergence",
            diverge,
            ROSE,
            "NaN",
            "A spike that never recovers.\nLR too high, or fp16 overflow.",
        ),
    ]

    for ax, (name, yv, color, flag, _) in zip(axes, panels, strict=True):
        ax.axhline(lnV, color=BASELINE, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.plot(x, yv, color=color, lw=2.3, zorder=3)
        if name == "Divergence":
            ax.plot(x, tail, color=ROSE, lw=2.3, zorder=3)
            ax.plot([x[261]], [tail[261]], "x", color=ROSE, ms=11, mew=3, zorder=4)
            ax.text(
                x[261] + 0.03,
                tail[261],
                "NaN",
                fontsize=11,
                color=ROSE,
                fontweight="bold",
                va="center",
            )
        elif flag:
            ax.text(
                0.5, lnV + 0.75, flag, fontsize=10, color=ROSE, fontweight="bold", ha="center"
            )
        ax.set_title(name, loc="left", fontsize=12, pad=10, color=INK)
        ax.set_xticks([])
        ax.set_ylim(0, 16.5)
        ax.set_xlim(0, 1)
        despine(ax, keep=())
        ax.grid(axis="y")

    axes[0].set_yticks([2, lnV], ["2", "ln(V) = 10.80"])
    axes[0].set_ylabel("loss")

    for ax, (_, _, _, _, caption) in zip(axes, panels, strict=True):
        box = ax.get_position()
        fig.text(
            box.x0, box.y0 - 0.055, caption, fontsize=9.3, color=INK_2, va="top", ha="left"
        )

    fig.text(
        0.008,
        0.965,
        "What the loss curve is telling you",
        fontsize=14,
        fontweight="bold",
        color=INK,
        ha="left",
        va="top",
    )
    fig.text(
        0.008,
        0.885,
        "Every run starts from the same place: a randomly initialized "
        "language model must begin at ln(vocab_size).",
        fontsize=10.3,
        color=INK_2,
        ha="left",
        va="top",
    )
    fig.text(
        0.008,
        0.045,
        "Schematic: curve shapes, not measurements. "
        "The ln(V) reference line is exact for V = 49,152.",
        fontsize=8.8,
        color=MUTED,
        ha="left",
        va="top",
    )
    fig.savefig(f"{OUT}/07_loss_anatomy.png", dpi=200)
    plt.close(fig)
    print(f"  wrote {OUT}/07_loss_anatomy.png")


# ---------------------------------------------------------------- 8
def fig_memory_over_step():
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    x = np.linspace(0, 3, 600)
    states = 2.0
    act = np.piecewise(
        x,
        [x < 1, (x >= 1) & (x < 2), x >= 2],
        [lambda t: 2.6 * (t**1.25), lambda t: 2.6 * (1 - (t - 1) ** 1.5), lambda t: 0.0],
    )
    grads = np.piecewise(
        x,
        [x < 1, (x >= 1) & (x < 2), x >= 2],
        [lambda t: 0.0, lambda t: 2.0 * (t - 1), lambda t: 2.0],
    )
    ax.fill_between(x, 0, states, color=BLUE_ORDINAL[3], label="parameters + optimizer state")
    ax.fill_between(x, states, states + grads, color=BLUE_ORDINAL[1], label="gradients")
    ax.fill_between(x, states + grads, states + grads + act, color=GOLD, label="activations")
    peak = np.argmax(states + grads + act)
    ax.plot(x[peak], (states + grads + act)[peak], "o", color=ROSE, ms=9, zorder=5)
    ax.annotate(
        "peak: everything stored,\nnothing yet released",
        xy=(x[peak], (states + grads + act)[peak]),
        xytext=(x[peak] - 0.62, (states + grads + act)[peak] + 1.25),
        fontsize=9.8,
        color=ROSE,
        fontweight="bold",
        ha="right",
        arrowprops=dict(arrowstyle="->", color=ROSE, lw=1.6, connectionstyle="arc3,rad=-0.2"),
    )
    # foreach optimizers batch elementwise ops across every parameter, allocating
    # intermediate buffers, so the step phase is not flat. Labelled directly rather
    # than as a fourth legend entry, which the three-hue palette has no room for.
    transient = np.where(x >= 2, 0.45 * np.exp(-(((x - 2.5) / 0.30) ** 2)), 0.0)
    top = states + grads + act
    ax.fill_between(x, top, top + transient, color=BLUE_ORDINAL[0], linewidth=0)
    ax.text(
        2.5,
        states + 2.0 + 0.62,
        "foreach transients",
        ha="center",
        fontsize=9.3,
        color=INK_2,
        fontweight="bold",
    )
    for xpos, lab in ((0.5, "FORWARD"), (1.5, "BACKWARD"), (2.5, "OPTIMIZER STEP")):
        ax.text(xpos, -0.55, lab, ha="center", fontsize=9.5, color=INK_2, fontweight="bold")
    for b in (1, 2):
        ax.axvline(b, color=BASELINE, lw=1.2, ls=(0, (3, 3)))
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 7.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("memory in use")
    ax.legend(loc="upper right", ncol=1)
    ax.grid(False)
    despine(ax, keep=("bottom",))
    title(
        ax,
        "Memory over one training step",
        "Activations accumulate through forward and are consumed during backward. The peak is usually at the boundary.",
    )
    note(
        fig,
        "Schematic: the shape of one steady-state iteration. Not to scale, and not the first iteration — "
        "Adam's moments do not exist until the first step() completes.\n"
        "The bump in the step phase is foreach's intermediate buffers, which can exceed the boundary peak "
        "when activations are small.",
    )
    save(fig, f"{OUT}/08_memory_over_step.png")


# ---------------------------------------------------------------- 9
def fig_zero_stages():
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    stages = [
        "No sharding\n(DDP)",
        "ZeRO-1\noptimizer",
        "ZeRO-2\n+ gradients",
        "ZeRO-3\n+ parameters",
    ]
    N = 4
    params = [4, 4, 4, 4 / N]
    grads = [4, 4, 4 / N, 4 / N]
    optim = [8, 8 / N, 8 / N, 8 / N]
    ax.bar(
        stages,
        params,
        color=BLUE_ORDINAL[3],
        width=0.5,
        edgecolor=SURFACE,
        linewidth=2,
        label="parameters",
    )
    ax.bar(
        stages,
        grads,
        bottom=params,
        color=BLUE_ORDINAL[1],
        width=0.5,
        edgecolor=SURFACE,
        linewidth=2,
        label="gradients",
    )
    ax.bar(
        stages,
        optim,
        bottom=np.add(params, grads),
        color=GOLD,
        width=0.5,
        edgecolor=SURFACE,
        linewidth=2,
        label="optimizer state (m, v)",
    )
    for i in range(4):
        tot = params[i] + grads[i] + optim[i]
        ax.text(
            i,
            tot + 0.35,
            f"{tot:.0f} B" if tot == int(tot) else f"{tot:.1f} B",
            ha="center",
            fontsize=11,
            color=INK,
            fontweight="bold",
        )
    ax.set_ylabel("bytes per parameter, per GPU")
    ax.set_ylim(0, 19)
    ax.legend(loc="upper right")
    despine(ax, keep=("bottom",))
    title(
        ax,
        "What each ZeRO stage actually shards",
        "Shown for 4 ranks. The optimizer is the biggest bucket, which is why stage 1 goes after it first.",
    )
    note(
        fig,
        "Arithmetic, for N=4 ranks. Communication volume rises as the sharding deepens; that cost is not shown.",
    )
    save(fig, f"{OUT}/09_zero_stages.png")


# ---------------------------------------------------------------- 10
def fig_ragged_shard():
    fig, ax = plt.subplots(figsize=(8.4, 3.3))
    n_samples, ranks, bs = 26, 4, 2
    per = n_samples // (ranks * bs)
    for r in range(ranks):
        for b in range(per + 1):
            for s in range(bs):
                idx = r * bs + s + b * ranks * bs
                x, y = b * (bs + 0.55) + s, ranks - 1 - r
                if idx < n_samples:
                    color = BLUE_ORDINAL[1] if b < per else ROSE
                    ax.add_patch(
                        Rectangle(
                            (x, y - 0.34),
                            0.92,
                            0.68,
                            facecolor=color,
                            edgecolor=SURFACE,
                            linewidth=2,
                        )
                    )
                    ax.text(
                        x + 0.46,
                        y,
                        str(idx),
                        ha="center",
                        va="center",
                        fontsize=8.5,
                        color="white",
                        fontweight="bold",
                    )
                else:
                    ax.add_patch(
                        Rectangle(
                            (x, y - 0.34),
                            0.92,
                            0.68,
                            facecolor="none",
                            edgecolor=ROSE,
                            linewidth=1.6,
                            linestyle=(0, (3, 2)),
                        )
                    )
                    ax.text(
                        x + 0.46, y, "—", ha="center", va="center", fontsize=10, color=ROSE
                    )
    ax.set_yticks(range(ranks), [f"rank {ranks - 1 - r}" for r in range(ranks)], fontsize=10)
    ax.set_xticks([])
    ax.set_xlim(-0.3, (per + 1) * (bs + 0.55))
    ax.set_ylim(-0.65, ranks - 0.25)
    ax.grid(False)
    despine(ax, keep=())
    ax.text(
        0,
        -0.62,
        "26 samples · 4 ranks · batch size 2  →  three clean rounds, then a ragged one.\n"
        "drop_last=True discards samples 24-25. drop_last=False leaves ranks 1-3 with nothing in the final "
        "step, and an unguarded loss average is then wrong.",
        fontsize=9.5,
        color=INK_2,
        ha="left",
        va="top",
        transform=ax.get_yaxis_transform(which="grid"),
    )
    title(
        ax,
        "The last batch never divides evenly",
        "Every rank must take the same number of steps, or the collective deadlocks.",
    )
    save(fig, f"{OUT}/10_ragged_shard.png")


for f in (
    fig_param_breakdown,
    fig_textbook_error,
    fig_memory_buckets,
    fig_optimizer_cost,
    fig_float_formats,
    fig_fp16_underflow,
    fig_loss_anatomy,
    fig_memory_over_step,
    fig_zero_stages,
    fig_ragged_shard,
):
    f()
print("done")
