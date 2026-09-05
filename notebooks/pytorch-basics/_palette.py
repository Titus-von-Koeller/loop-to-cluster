"""The color constants of the viewing vocabulary, importable without torch.

Split from _viz.py so that pages which only need colors (the vision-calibration
instrument) never import torch: a fresh marimo run session instantiates in a worker
thread, and importing torch from a non-main thread can die mid-import ("module 'torch'
has no attribute 'cuda'", measured 2026-09-02), taking every dependent cell with it.
_viz.py re-exports everything here, so notebooks that already import from _viz are
unaffected. The color policy and its rationale live in _viz.py's docstring; these values
change together with the crossovers measured there, or not at all.
"""

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

# cividis at 0, .25, .5, .75, 1 (matplotlib 3.11.1) — the CVD-optimized sequential map,
# replacing the original single-hue blue ramp. Dark end is LOW: ink logic in _viz flips
# to white below 0.48 of the scale (measured against these values).
RAMP = ["#00224e", "#434e6c", "#7d7c78", "#bcae6c", "#fee838"]
POLARITY = ["#8f3413", "#d95926", "#eaa886", "#e8e8e6", "#93bae9", "#2a78d6", "#173f6e"]

# Ink for text set on a known square fill — never on the page, whose color belongs to the
# reader's theme. The pair show()'s contrast crossovers are calibrated against.
# >>> measured:viz-furniture
# Written by theme-calibration (`python -m theme.appliers.viz`) from the published palette;
# never by hand. The ink on a coloured data fill is the paper -- day's on the dark end of a
# ramp, night's on the light end -- as it is on a button. The data palettes above are chosen
# by discriminability and are not furniture: the applier refuses any write that changes them.
INK_LIGHT = "#f9ecdd"
INK_DARK = "#222325"

# Graph FURNITURE, keyed by polarity: a chart's canvas is the code paper, the page under it
# the notebook page, its axis ink the measured ink, tick labels the comment step, gridlines
# and axis lines the border tint. Only the notebook knows which polarity it is read on
# (mo.app_meta().theme), so both are here.
FURNITURE = {
    "day": {
        "paper": "#f9ecdd",
        "page": "#efe2d3",
        "ink": "#474442",
        "label": "#56524f",
        "grid": "#cebda6",
        "axis": "#cebda6",
    },
    "night": {
        "paper": "#222325",
        "page": "#1b1c1e",
        "ink": "#c0bfc0",
        "label": "#a2a2a4",
        "grid": "#36383b",
        "axis": "#36383b",
    },
}
# <<< measured:viz-furniture


def tint(color, toward_white):
    """The palette hue mixed toward a card's white, as a literal hex.

    For renderers that cannot take scheme names (graphviz, raw CSS): fills stay derived
    from the constants above instead of hand-tuned hexes appearing per notebook.
    """
    channels = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{round(c + (255 - c) * toward_white):02x}" for c in channels)
