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
    `calibration-responses.jsonl` beside this file, so sittings accumulate across days, themes,
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
    from _palette import OKABE_ITO, POLARITY, RAMP
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
    LOG = Path(__file__).parent / "calibration-responses.jsonl"

    return GROUNDS, LOG, PAIRS, datetime, json, pd, random, timezone


@app.cell(hide_code=True)
def _(LOG, json, mo):
    _existing = [json.loads(_line) for _line in LOG.read_text().splitlines() if _line.strip()] if LOG.exists() else []
    get_responses, set_responses = mo.state(_existing)
    return get_responses, set_responses


@app.cell(hide_code=True)
def _(GROUNDS, PAIRS, random):
    def _pair_key(palette, a, b):
        return (palette, *sorted((a, b)))

    def trial_for(n, responses):
        """The nth trial: Bayesian uncertainty sampling over per-pair Beta posteriors.

        Each pair's accuracy carries a Beta(1+correct, 1+wrong) posterior; the next trial
        goes to the pair whose posterior variance is largest — untested pairs first, then
        the ones the responses keep failing, so measurement concentrates where your eyes
        are least readable. Deterministic given n and the first n responses, so every
        surface poses the same trial. (For continuous just-noticeable-difference
        staircases between two colors, the right tool is QUEST+ — parked in the queue.)
        """
        _stats = {}
        for _r in responses[:n]:
            _k = _pair_key(_r["palette"], _r["base"], _r["odd_color"])
            _c, _w = _stats.get(_k, (0, 0))
            _stats[_k] = (_c + 1, _w) if _r["correct"] else (_c, _w + 1)

        def _variance(k):
            _c, _w = _stats.get(k, (0, 0))
            _a, _b = _c + 1, _w + 1
            return (_a * _b) / ((_a + _b) ** 2 * (_a + _b + 1))

        _rng = random.Random(n * 2654435761 % (2**31))
        if _rng.random() < 0.2:
            # Exploration keeps the sampler honest: without it, a pair judged easy in one
            # sitting's light is never revisited, and the posterior can fixate on early luck.
            _palette, _a, _b = _rng.choice(PAIRS)
        else:
            _best = max(_variance(_pair_key(*_p)) for _p in PAIRS)
            _palette, _a, _b = _rng.choice([_p for _p in PAIRS if _variance(_pair_key(*_p)) >= _best - 1e-12])
        if _rng.random() < 0.5:
            _a, _b = _b, _a
        _ground_name = ("day", "night")[n % 2]
        return {
            "palette": _palette,
            "base": _a,
            "odd_color": _b,
            "ground": _ground_name,
            "ground_hex": GROUNDS[_ground_name],
            "odd_position": _rng.randrange(4),
        }

    return (trial_for,)


@app.cell(hide_code=True)
def _(get_responses, mo, trial_for):
    # The trial number doubles as a staleness indicator: if it ever disagrees with the
    # squares below, the surface lagged and clicks are being dropped by the guard.
    _n = len(get_responses())
    mo.md(f"**Trial {_n + 1}** — click the odd square.")
    return


@app.cell(hide_code=True)
def _(LOG, datetime, get_responses, json, mo, set_responses, timezone, trial_for):
    _n = len(get_responses())
    _t = trial_for(_n, get_responses())
    _colors = [_t["base"]] * 4
    _colors[_t["odd_position"]] = _t["odd_color"]

    # A real widget instead of styled buttons: the squares are plain clickable divs on one
    # ground, so nothing of a button's chrome shows. anywidget syncs the click back as the
    # chosen index; a fresh widget renders per trial and the guard drops stale clicks.
    import anywidget
    import traitlets

    class _OddOneOut(anywidget.AnyWidget):
        _esm = """
        function render({ model, el }) {
          const wrap = document.createElement("div");
          wrap.style.cssText = `background:${model.get("ground")};padding:30px 22px;` +
            `border-radius:10px;display:inline-flex;gap:28px`;
          model.get("colors").forEach((c, i) => {
            const sq = document.createElement("div");
            sq.style.cssText = `width:72px;height:72px;border-radius:8px;background:${c};` +
              `cursor:pointer`;
            sq.onclick = () => {
              model.set("clicks", model.get("clicks") + 1);
              model.set("choice", i);
              model.save_changes();
            };
            wrap.appendChild(sq);
          });
          el.replaceChildren(wrap);
        }
        export default { render };
        """
        colors = traitlets.List([]).tag(sync=True)
        ground = traitlets.Unicode("#ffffff").tag(sync=True)
        choice = traitlets.Int(-1).tag(sync=True)
        clicks = traitlets.Int(0).tag(sync=True)

    answer_squares = mo.ui.anywidget(_OddOneOut(colors=_colors, ground=_t["ground_hex"]))
    answer_squares
    return (answer_squares,)


@app.cell(hide_code=True)
def _(LOG, answer_squares, datetime, get_responses, json, set_responses, timezone, trial_for):
    # Recording watches the widget's synced traits. Only the FIRST click of a fresh widget
    # counts (clicks == 1): later clicks on the same trial, and clicks on an orphaned stale
    # widget, record nothing — the guard below double-checks against the response count.
    _n = len(get_responses())

    def _record(choice, n=_n):
        # The squares are the buttons, so they re-render per trial — which reintroduces the
        # stale-surface risk. The guard converts it from data corruption into a dropped
        # click: a rendering whose trial is no longer current records nothing.
        if n != len(get_responses()):
            return
        _now = trial_for(n, get_responses())
        _entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "palette": _now["palette"],
            "base": _now["base"],
            "odd_color": _now["odd_color"],
            "ground": _now["ground"],
            "odd_position": _now["odd_position"],
            "choice": choice,
            "correct": choice == _now["odd_position"],
        }
        # Append-only, one record per line: concurrent sessions interleave instead of
        # overwriting each other's history.
        with LOG.open("a") as _f:
            _f.write(json.dumps(_entry) + "\n")
        set_responses([*get_responses(), _entry])

    _v = answer_squares.value
    if _v.get("clicks") == 1 and _v.get("choice", -1) >= 0:
        _record(_v["choice"])
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
    one page and fail on the other. Trials accumulate in `calibration-responses.jsonl`, which is
    committed like any measurement: future sessions (and future exhibits) read it to weight
    palette choices by *your measured* confusions instead of the population model. When enough
    trials exist, the next step is written in the queue: fit your personal confusion axis from
    the misses and re-rank the theme gallery's dropdown with it.
    """)
    return


if __name__ == "__main__":
    app.run()
