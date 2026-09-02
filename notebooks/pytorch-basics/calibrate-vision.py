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

    What is being optimized is not your score but your **threshold surface**: a Bayesian
    observer model (a Weibull psychometric over weighted opponent-space distance — the QUEST+
    family) learns how far apart two colors must be, per direction (red–green, blue–yellow,
    lightness) and per ground, before you can tell them apart. Each trial is *generated* to be
    maximally informative about that model, which parks it near your ~75%-correct zone —
    **feeling hard means it is working**, and every answer moves the whole surface, not one
    pair's tally. A fraction of trials stay easy palette pairs, as anchors and breathers.

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
    import numpy as np
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

    return GROUNDS, LOG, PAIRS, datetime, json, np, pd, random, timezone


@app.cell(hide_code=True)
def _(LOG, json, mo):
    _existing = [json.loads(_line) for _line in LOG.read_text().splitlines() if _line.strip()] if LOG.exists() else []
    get_responses, set_responses = mo.state(_existing)
    return get_responses, set_responses


@app.cell(hide_code=True)
def _(GROUNDS, PAIRS, np, random):
    # The observer model. sRGB -> cone-opponent space: linearize, project to LMS
    # (Hunt-Pointer-Estevez on D65 XYZ), cube-root compress, then opponent axes —
    # lum = L+M, rg = L-M, by = S-(L+M)/2. Deliberately approximate; the fitted weights
    # absorb each axis's scale. Probability correct in 4AFC is a Weibull psychometric over
    # the weighted opponent distance, with a 2% lapse ceiling.
    _SRGB2XYZ = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    _XYZ2LMS = np.array([[0.4002, 0.7076, -0.0808], [-0.2263, 1.1653, 0.0457], [0.0, 0.0, 0.9182]])
    _OPP = np.array([[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-0.5, -0.5, 1.0]])
    _RGB2LMS = _XYZ2LMS @ _SRGB2XYZ
    _LMS2RGB = np.linalg.inv(_RGB2LMS)
    _OPP_INV = np.linalg.inv(_OPP)

    def opp(hex_color):
        c = np.array([int(hex_color.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)])
        lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        return _OPP @ np.cbrt(np.clip(_RGB2LMS @ lin, 0.0, None))

    def _to_hex(opp_vec):
        lin = np.clip(_LMS2RGB @ (_OPP_INV @ opp_vec) ** 3, 0.0, 1.0)
        srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
        return "#" + "".join(f"{round(255 * v):02x}" for v in srgb)

    # Parameter grid, QUEST+-style: exact discrete posterior, no sampler to tune. Axis
    # weights are relative to lightness (fixed 1); thresholds are per ground, on a scale
    # set by the palette pairs' own distance distribution.
    _dref = np.array([np.linalg.norm(opp(_a) - opp(_b)) for _pal, _a, _b in PAIRS])
    _TAUS = np.geomspace(float(np.quantile(_dref, 0.05)) / 10, float(np.quantile(_dref, 0.9)), 12)
    # Lapse is a fitted axis, not a constant: the model can attribute rare misses to the
    # finger instead of the eyes, and stimulus placement marginalizes over that belief —
    # an accidental miss or lucky guess costs a few misplaced trials, never a bad path,
    # because the posterior is global and self-correcting.
    GRID = np.stack(
        np.meshgrid(
            np.geomspace(0.05, 12.0, 12),
            np.geomspace(0.05, 6.0, 10),
            _TAUS,
            _TAUS,
            np.array([0.005, 0.02, 0.05, 0.1]),
            indexing="ij",
        )
    )

    def p_correct(delta, ground):
        """P(correct) over the whole grid for one trial's opponent delta."""
        _d2 = delta[0] ** 2 + GRID[0] * delta[1] ** 2 + GRID[1] * delta[2] ** 2
        _tau = GRID[2] if ground == "day" else GRID[3]
        return 0.25 + (0.75 - GRID[4]) * (1.0 - np.exp(-_d2 / _tau**2))

    def posterior_for(responses):
        if not responses:
            _post = np.ones(GRID.shape[1:])
            return _post / _post.sum()
        # Vectorized over trials: deltas (n,3), grounds (n,), one broadcast against the grid.
        _da = np.array([np.abs(opp(_r["base"]) - opp(_r["odd_color"])) for _r in responses])
        _night = np.array([_r["ground"] == "night" for _r in responses])
        _ok = np.array([bool(_r["correct"]) for _r in responses])
        _flat = GRID.reshape(5, -1)
        _d2 = _da[:, 0:1] ** 2 + _flat[0] * _da[:, 1:2] ** 2 + _flat[1] * _da[:, 2:3] ** 2
        _tau = np.where(_night[:, None], _flat[3], _flat[2])
        _p = 0.25 + (0.75 - _flat[4]) * (1.0 - np.exp(-_d2 / _tau**2))
        _logp = np.log(np.where(_ok[:, None], _p, 1.0 - _p)).sum(axis=0)
        _logp -= _logp.max()
        _post = np.exp(_logp).reshape(GRID.shape[1:])
        return _post / _post.sum()

    def trial_for(n, responses):
        """The nth trial, generated to maximize expected information about the model.

        Candidate stimuli are built from a palette color plus an offset along the opponent
        axes at magnitudes bracketing the current threshold estimate; the winner maximizes
        mutual information between the response and the posterior — which parks trials near
        the ~75%-correct zone, where each answer says the most. 15% of trials stay plain
        palette pairs, as anchors against model misspecification. Deterministic given the
        shared log.
        """
        _rng = random.Random(n * 2654435761 % (2**31))
        # Blocked, not alternating: flipping the page every click churns light/dark
        # adaptation and adds measurement noise. Sixteen-trial blocks keep the eye in one
        # adapted state while still balancing the two grounds over a sitting.
        _ground = ("day", "night")[(n // 16) % 2]
        _pal, _a, _b = _rng.choice(PAIRS)
        if _rng.random() < 0.15:
            if _rng.random() < 0.5:
                _a, _b = _b, _a
            _base, _odd, _kind = _a, _b, _pal
        else:
            _post = posterior_for(responses[:n])
            _base_o = opp(_a)
            _s2 = 1 / np.sqrt(2)
            _dirs = [
                np.array(_v)
                for _v in [
                    (0, 1, 0),
                    (0, -1, 0),
                    (0, 0, 1),
                    (0, 0, -1),
                    (1, 0, 0),
                    (-1, 0, 0),
                    (0, _s2, _s2),
                    (0, -_s2, _s2),
                ]
            ]

            def _entropy(q):
                return -(q * np.log(q + 1e-12) + (1 - q) * np.log(1 - q + 1e-12))

            _best, _best_hex = -1.0, None
            for _dv in _dirs:
                for _m in np.geomspace(_TAUS[0] * 0.5, _TAUS[-1] * 1.5, 6):
                    _cand = _to_hex(_base_o + _dv * _m)
                    if _cand == _a:
                        continue
                    _pth = p_correct(np.abs(_base_o - opp(_cand)), _ground)
                    _pbar = float((_post * _pth).sum())
                    _eig = _entropy(_pbar) - float((_post * _entropy(_pth)).sum())
                    if _eig > _best:
                        _best, _best_hex = _eig, _cand
            _base, _odd, _kind = _a, _best_hex, "probe"
        return {
            "palette": _kind,
            "base": _base,
            "odd_color": _odd,
            "ground": _ground,
            "ground_hex": GROUNDS[_ground],
            "odd_position": _rng.randrange(4),
        }

    return GRID, posterior_for, trial_for


@app.cell(hide_code=True)
def _(get_responses, mo, trial_for):
    # The trial number doubles as a staleness indicator: if it ever disagrees with the
    # squares below, the surface lagged and clicks are being dropped by the guard.
    _n = len(get_responses())
    mo.hstack([mo.md(f"**Trial {_n + 1}** — click the odd square.")], justify="center")
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
          el.style.cssText = `display:block;width:100%`;
          const wrap = document.createElement("div");
          wrap.style.cssText = `background:${model.get("ground")};padding:22px;` +
            `border-radius:10px;display:flex;justify-content:center;align-items:center;` +
            `gap:12px;width:100%;box-sizing:border-box;aspect-ratio:1.618/1`;
          model.get("colors").forEach((c, i) => {
            const sq = document.createElement("div");
            // Fixed pixels on purpose: patch size AND separation are stimulus parameters
            // (spatial summation; near-abutting fields give the most sensitive
            // comparison, and match how adjacent glyphs and chart marks are actually
            // read). Both are logged with every response.
            sq.style.cssText = `width:104px;height:104px;border-radius:10px;background:${c};` +
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
            # Patch size is a stimulus parameter; log it so size changes stay analyzable.
            "size_px": 104,
            "gap_px": 12,
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
def _(GRID, get_responses, mo, np, pd, posterior_for):
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
        _post = posterior_for(_log)
        _wrg, _wby, _td, _tn, _lapse = (float((GRID[_i] * _post).sum()) for _i in range(5))
        _out = mo.vstack(
            [
                mo.md(
                    "**What the model has learned about your eyes** (lightness sensitivity = 1; "
                    "a smaller weight means that axis needs a larger difference before you see it):"
                ),
                mo.hstack(
                    [
                        mo.stat(f"{_wrg:.2f}", label="red–green weight", bordered=True),
                        mo.stat(f"{_wby:.2f}", label="blue–yellow weight", bordered=True),
                        mo.stat(f"{_td:.3f}", label="threshold, day ground", bordered=True),
                        mo.stat(f"{_tn:.3f}", label="threshold, night ground", bordered=True),
                        mo.stat(f"{100 * _lapse:.1f}%", label="your fitted slip rate", bordered=True),
                    ],
                    justify="start",
                    gap=1,
                ),
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
