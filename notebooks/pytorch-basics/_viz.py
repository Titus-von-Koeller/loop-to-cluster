"""The one viewing vocabulary for these notebooks.

Shared on purpose: every notebook that draws imports from here, so how a tensor or a chart
looks is decided once and evolves once — evolve this module, never fork it into a notebook.
The accepted cost is that a notebook no longer runs as a standalone file.

Color policy: magnitude is carried by lightness and only by lightness; hue never carries
identity alone. Categorical hues come from Okabe and Ito's palette, and continuous scales
are cividis (sequential) and blue-orange (diverging) — established palettes whose hue
pairs stay reliably separable under red-green color-vision deficiency and whose pictures
survive grayscale. A hand-tuned hex value in a notebook is a bug.
"""

import altair as alt
import marimo as mo
import pandas as pd
import torch

# Okabe & Ito (2008): the categorical palette designed for color-vision deficiency.
OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
}

# The two-role pair most charts need: the thing itself, and the thing pointed at.
BASE = OKABE_ITO["blue"]
ACCENT = OKABE_ITO["vermillion"]

# Vega scheme names for continuous data, same constraint.
SEQUENTIAL_SCHEME = "cividis"
DIVERGING_SCHEME = "blueorange"

# show()'s own ramps predate the policy and satisfy it: both carry magnitude as lightness,
# the diverging one adds hue only for sign, and the text-contrast crossovers inside show()
# are measured against exactly these values -- change them together or not at all.
#
# Magnitude is carried by lightness and only by lightness, so the picture survives being
# read by someone who cannot separate red from green, and survives being printed gray.
# Hue carries sign and nothing else. There are no axes and no chart title: the numbers
# are in the squares and the caption says what the object is.
RAMP = ["#dbe7f7", "#a8c6ec", "#6b9ede", "#2a78d6", "#17457c"]
POLARITY = ["#8f3413", "#d95926", "#eaa886", "#e8e8e6", "#93bae9", "#2a78d6", "#173f6e"]

# Ink for text set on a known square fill — never on the page, whose color belongs to the
# reader's theme. The pair show()'s contrast crossovers are calibrated against; exhibits
# that color their own squares use the same pair rather than re-tuning it locally.
INK_LIGHT = "#ffffff"
INK_DARK = "#15181d"


def show(tensor, title=None, cell=54, facts=True):
    """Render a small tensor as its own numbers, colored by magnitude."""
    values = torch.as_tensor(tensor).detach().cpu()
    grid = values.reshape(1, 1) if values.dim() == 0 else values if values.dim() == 2 else values.reshape(1, -1)
    numbers = [[float(v) for v in row] for row in grid.tolist()]
    signed = min(min(row) for row in numbers) < 0
    limit = max((max(abs(v) for v in row) for row in numbers), default=1.0) or 1.0
    digits = ".0f" if not values.dtype.is_floating_point else ".2f"

    frame = pd.DataFrame([{"col": j, "row": i, "v": v} for i, row in enumerate(numbers) for j, v in enumerate(row)])
    # The gap between squares is left transparent, so it takes the color of whatever
    # theme the notebook is being read in rather than a white I chose.
    at = {
        "x": alt.X("col:O", axis=None, scale=alt.Scale(paddingInner=0.06)),
        "y": alt.Y("row:O", axis=None, scale=alt.Scale(paddingInner=0.06)),
    }
    # Ink on a square is chosen against that square's fill, which is known here, rather
    # than against the page, which is not. The crossovers are measured, not guessed: white
    # only overtakes near-black at 0.73 of the sequential ramp, and at 0.71 of the
    # diverging one, taking the later of its two arms so neither switches early.
    on_dark = f"abs(datum.v) > {0.71 * limit}" if signed else f"datum.v > {0.73 * limit}"
    picture = (
        alt.Chart(frame)
        .mark_rect()
        .encode(
            **at,
            color=alt.Color(
                "v:Q",
                scale=alt.Scale(range=POLARITY, domain=[-limit, limit])
                if signed
                else alt.Scale(range=RAMP, domain=[0, limit]),
                legend=None,
            ),
            tooltip=[alt.Tooltip("v:Q", format=".4f", title="value"), "row:O", "col:O"],
        )
        + alt.Chart(frame)
        .mark_text(fontSize=13, fontWeight=500)
        .encode(
            **at,
            text=alt.Text("v:Q", format=digits),
            color=alt.condition(on_dark, alt.value(INK_LIGHT), alt.value(INK_DARK)),
        )
    ).properties(width=cell * len(numbers[0]), height=cell * len(numbers))

    caption = (
        f"`{tuple(values.shape)}` · `{str(values.dtype).removeprefix('torch.')}` · "
        f"stride `{values.stride()}`" + ("" if values.is_contiguous() else " · **not contiguous**")
    )
    parts = (
        ([mo.md(f"**{title}**")] if title else [])
        + [picture]
        + ([mo.md(f"<small>{caption}</small>")] if facts else [])
    )
    return mo.vstack(parts, align="center", gap=0.2)
