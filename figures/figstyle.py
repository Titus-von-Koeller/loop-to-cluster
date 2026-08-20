"""House style for the training-fundamentals figures.

The palette is measured, not chosen. Candidate schemes were run through a
CVD (colour-vision-deficiency) validator and scored on worst-adjacent-pair
separation in OKLab deltaE. Tol high-contrast won at 21.3 deutan; every teal
or green added to it collapsed separation to 4.5-9.3, because that is the
red-green axis. Hence three hues, and lightness for everything past three.

Figures render on an opaque light surface rather than transparency: Notion
serves one static PNG to both its light and dark themes, and dark ink on
transparency is unreadable on a dark page.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# --- surfaces and ink -------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# --- categorical: three hues, CVD deltaE 21.3 -------------------------
BLUE = "#2166AC"
GOLD = "#DDAA33"
ROSE = "#BB5566"
CATEGORICAL = [BLUE, GOLD, ROSE]

# --- ordinal: one hue, lightness carries identity ---------------------
BLUE_ORDINAL = ["#86b6ef", "#3987e5", "#256abf", "#184f95"]

# --- semantic ---------------------------------------------------------
GOOD = "#2166AC"  # deliberately not green
BAD = "#BB5566"
NEUTRAL = "#b9b7ae"


def use_house_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.28,
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 1.0,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.titlepad": 14,
            "axes.labelpad": 8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.9,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
            "lines.solid_capstyle": "round",
        }
    )


def despine(ax, keep=("bottom",)) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def title(ax, text: str, subtitle: str | None = None) -> None:
    """Bold title with an optional recessive subtitle beneath it."""
    ax.set_title(text, loc="left", pad=22 if subtitle else 14)
    if subtitle:
        ax.text(
            0,
            1.045,
            subtitle,
            transform=ax.transAxes,
            fontsize=10,
            color=INK_2,
            va="bottom",
            ha="left",
        )


def note(fig, text: str) -> None:
    """Footnote: says whether a figure is measured or schematic."""
    fig.text(0.0, -0.035, text, fontsize=8.8, color=MUTED, ha="left", va="top")


def save(fig, path: str) -> None:
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  wrote {path}")
