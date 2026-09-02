# /// script
# [tool.marimo.runtime]
# on_cell_change = "autorun"
# ///

# The repository default is lazy, which marks a cell stale rather than running it when
# something upstream changes -- correct for a notebook holding a model on the GPU, and
# fatal for a trial loop, whose whole point is that the next trial appears on click.
import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *A sidecar to the theme gallery — the instrument that replaces its simulations with you.*

    # Calibrating the palettes against your eyes

    The gallery's deuteranopia column is a population model; what decides legibility is what
    *your* eyes distinguish on *this* screen, in the theme and light you actually read in. This
    notebook measures that directly: each trial shows four squares on one of the two Horizon
    page grounds — three of one color, one of another, both drawn from the palettes under
    evaluation (the editor theme's own accents included). Click the odd one out. Chance is 25%;
    a pair you can no longer beat chance on is, for you, one color.

    Protocol: glance, decide within about a second, click — hesitation measures reasoning, not
    perception. Sixty-plus trials make a sitting; every response appends to
    `calibration-responses.json` beside this file, so sittings accumulate across days, themes,
    and ambient light. The screen itself is uncalibrated for now (parked in the queue) — that
    limits absolute claims, not relative ones: which pairs and which palettes fail *you* on
    *this* screen is exactly what accumulates below.
    """)
    return


@app.cell(hide_code=True)
def _():
    import json
    import random
    import re
    from datetime import datetime, timezone
    from pathlib import Path

    import matplotlib as mpl
    import pandas as pd
    from _viz import OKABE_ITO, POLARITY, RAMP
    from cmcrameri import cm as cmc

    def _hexes(cmap, n=7):
        cmap = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
        if hasattr(cmap, "colors") and len(cmap.colors) < 30:
            picks = list(cmap.colors)[:n]
        else:
            picks = [cmap(i / (n - 1)) for i in range(n)]
        return ["#" + "".join(f"{round(255 * v):02x}" for v in c[:3]) for c in picks]

    def _horizon_accents():
        """The editor theme's own accents, alpha composited onto their page."""
        found = {}
        for label, name in (("horizon-day", "horizon-bright-bold.json"), ("horizon-night", "horizon-bold.json")):
            candidates = list(Path.home().glob(f".vscode/extensions/*horizon*/themes/{name}"))
            if not candidates:
                continue
            theme = json.loads(re.sub(r"//[^\n\"]*$", "", candidates[0].read_text(), flags=re.M))
            page = theme["colors"]["editor.background"]
            accents = [theme["colors"].get("textLink.foreground")]
            wanted = ["keyword", "string", "variable", "entity.name.function", "comment"]
            for entry in theme.get("tokenColors", []):
                scopes = entry.get("scope", [])
                scopes = [scopes] if isinstance(scopes, str) else scopes
                color = entry.get("settings", {}).get("foreground")
                if color and any(s == w or s.startswith(w) for w in list(wanted) for s in scopes):
                    accents.append(color)
                    wanted = [w for w in wanted if not any(s == w or s.startswith(w) for s in scopes)]
            bg = [int(page.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)]

            def _flat(c, bg=bg):
                raw = c.lstrip("#")
                a = int(raw[6:8], 16) / 255 if len(raw) == 8 else 1.0
                rgb = [int(raw[i : i + 2], 16) for i in (0, 2, 4)]
                return "#" + "".join(f"{round(a * v + (1 - a) * b):02x}" for v, b in zip(rgb, bg, strict=True))

            found[label] = [_flat(c) for c in accents if c]
        return found

    PALETTES = {
        "RAMP": RAMP,
        "POLARITY": POLARITY,
        "okabe-ito": list(OKABE_ITO.values())[:8],
        "cividis": _hexes("cividis"),
        "batlow": _hexes(cmc.batlow),
        "viridis": _hexes("viridis"),
        "tab10": _hexes("tab10", 8),
        "Set1": _hexes("Set1", 8),
        **_horizon_accents(),
    }

    # Every within-palette pair is a candidate trial; neighbors in sequential ramps measure
    # exactly the local contrast the gallery could only assert.
    PAIRS = [
        (name, a, b) for name, hexes in PALETTES.items() for i, a in enumerate(hexes) for b in hexes[i + 1 :] if a != b
    ]
    GROUNDS = {"day": "#fdf0ed", "night": "#1c1e26"}
    LOG = Path(__file__).parent / "calibration-responses.json"

    return GROUNDS, LOG, PAIRS, datetime, json, pd, random, timezone


@app.cell(hide_code=True)
def _(LOG, json, mo):
    _existing = json.loads(LOG.read_text()) if LOG.exists() else []
    get_responses, set_responses = mo.state(_existing)
    return get_responses, set_responses


@app.cell(hide_code=True)
def _(GROUNDS, LOG, PAIRS, datetime, get_responses, json, mo, random, set_responses, timezone):
    _n = len(get_responses())
    _rng = random.Random(_n * 2654435761 % (2**31))
    _palette, _a, _b = _rng.choice(PAIRS)
    if _rng.random() < 0.5:
        _a, _b = _b, _a
    _ground_name = ("day", "night")[_n % 2]
    _ground = GROUNDS[_ground_name]
    _odd = _rng.randrange(4)
    _colors = [_a] * 4
    _colors[_odd] = _b

    def _record(choice, odd=_odd, palette=_palette, a=_a, b=_b, ground=_ground_name):
        _entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "palette": palette,
            "base": a,
            "odd_color": b,
            "ground": ground,
            "odd_position": odd,
            "choice": choice,
            "correct": choice == odd,
        }
        _new = [*get_responses(), _entry]
        LOG.write_text(json.dumps(_new, indent=0))
        set_responses(_new)

    _squares = "".join(
        f'<span style="display:inline-block;width:72px;height:72px;border-radius:8px;'
        f'margin:0 14px;background:{c}"></span>'
        for c in _colors
    )
    _buttons = [mo.ui.button(label=str(i + 1), on_change=lambda _, i=i: _record(i)) for i in range(4)]
    mo.vstack(
        [
            mo.md(f"**Trial {_n + 1}** — which square is the odd one out?"),
            mo.Html(
                f'<div style="background:{_ground};padding:36px 22px;border-radius:10px;'
                f'display:inline-block">{_squares}</div>'
            ),
            mo.hstack(_buttons, justify="start", gap=3.4),
        ],
        gap=0.8,
    )
    return


@app.cell(hide_code=True)
def _(get_responses, mo, pd):
    _log = get_responses()
    if not _log:
        _out = mo.md("*No responses yet — the analysis fills in as you answer.*")
    else:
        _frame = pd.DataFrame(_log)
        _frame["pair"] = _frame.apply(lambda r: " / ".join(sorted([r.base, r.odd_color])), axis=1)
        _acc = _frame.correct.mean()
        _by_pair = (
            _frame.groupby(["palette", "pair"]).agg(n=("correct", "size"), accuracy=("correct", "mean")).reset_index()
        )
        _tested = _by_pair[_by_pair.n >= 3].sort_values("accuracy")
        _by_palette = (
            _frame.groupby("palette")
            .agg(trials=("correct", "size"), accuracy=("correct", "mean"))
            .reset_index()
            .sort_values("accuracy")
        )
        _by_ground = _frame.groupby("ground").correct.mean()
        _out = mo.vstack(
            [
                mo.hstack(
                    [
                        mo.stat(f"{len(_frame):,}", label="responses", bordered=True),
                        mo.stat(f"{100 * _acc:.0f}%", label="overall accuracy (chance 25%)", bordered=True),
                        *[
                            mo.stat(f"{100 * v:.0f}%", label=f"on the {g} ground", bordered=True)
                            for g, v in _by_ground.items()
                        ],
                    ],
                    justify="start",
                    gap=1,
                ),
                mo.md("**Palettes, hardest first for your eyes** (accuracy over all their tested pairs):"),
                mo.ui.table(_by_palette, selection=None),
                mo.md("**Most confused pairs so far** (at least three trials each):"),
                mo.ui.table(_tested.head(12), selection=None) if len(_tested) else mo.md("*none with n ≥ 3 yet*"),
            ],
            gap=0.8,
        )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reading the numbers, and what happens to them

    A pair at 25% is invisible to you; at 100% it is trivially yours; sequential ramps live or
    die by their *adjacent* pairs, categorical palettes by their worst pair anywhere. Grounds
    are logged because simultaneous contrast shifts discrimination — the same pair can pass on
    one page and fail on the other. Trials accumulate in `calibration-responses.json`, which is
    committed like any measurement: future sessions (and future exhibits) read it to weight
    palette choices by *your measured* confusions instead of the population model. When enough
    trials exist, the next step is written in the queue: fit your personal confusion axis from
    the misses and re-rank the theme gallery's dropdown with it.
    """)
    return


if __name__ == "__main__":
    app.run()
