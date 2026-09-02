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

# The constants live in _palette.py (importable without torch — see its docstring);
# re-exported here so every existing importer keeps its single entry point.
from _palette import (  # noqa: F401
    ACCENT,
    BASE,
    DIVERGING_SCHEME,
    INK_DARK,
    INK_LIGHT,
    OKABE_ITO,
    POLARITY,
    RAMP,
    SEQUENTIAL_SCHEME,
)


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
