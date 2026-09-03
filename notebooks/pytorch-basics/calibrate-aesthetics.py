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
    *A sidecar to calibrate-vision — that instrument measured what your eyes can distinguish;
    this one learns what they prefer.*

    # Calibrating the theme against your taste

    Legibility floors are measurable and measured; above them, theming has been guesswork.
    This notebook replaces the guess with a model: a latent **aesthetic utility** over a
    CAM16-UCS-parametrized theme space — page lightness and warmth, the accent set's hue,
    chroma, contrast and spread, how far comments recede, and VSCode's find-highlight as its
    own salience-versus-beauty axis. Each trial is one of three quick acts:

    - **A duel**: two candidate pages render the same real code in the fonts and pixel sizes
      you actually read. Click the one you would rather live in — trust the first pull.
    - **A comprehension probe**: one page, one instruction — *click the function name*. Your
      time to land on it measures what is truly easy to grasp, not what merely looks tidy.
    - **A find hunt**: the page shows search matches; click the current one. Time-to-find
      calibrates how loud `editor.findMatchBackground` must be before it stops earning its
      salience.

    Under the trials sits preferential Bayesian optimization: a Gaussian-process posterior
    over utility, a Bradley–Terry likelihood over your choices sharpened by reaction time
    (a fast, consistent click is strong evidence; a slow one reads as a near-tie, the way
    drift-diffusion models read decision time), and each duel *generated* to be maximally
    informative — the model's best guess against the challenger that would teach it most,
    with a small share of uniform probes as insurance against the model fooling itself.
    The candidates it chooses between are **bred fresh every trial**, not drawn from a fixed
    list: the themes it already rates highly, their mutated and recombined children, and a
    steady trickle of low-discrepancy newcomers, so the search can sit between any two
    themes it has shown you and can always still reach ground it has never visited.
    Every page is **code you have never seen** — generated, or lifted from a corner of the
    standard library — because a reused page turns time-to-find into a memory test.
    Your measured discrimination thresholds (from `calibration-responses.jsonl`, re-expressed
    in CAM16-UCS) and APCA/WCAG contrast floors are **hard constraints, never objectives**:
    every candidate you see is already legible; you are only ever asked which is *better*.

    Trials run in twenty-four-trial blocks per polarity (light page, dark page), each a run
    of sixteen duels, then four comprehension probes, then four find hunts — same-kind
    trials batched so one instruction serves a run and you never switch task mid-stride; a
    begin button gates each run. Blocks by polarity so your adaptation state is part of the
    measurement, not noise in it — and the **whole page**, not just the band, takes the
    ground under test, because in full screen the surround is most of what your eyes adapt
    to. A duel keeps the polarity's neutral surround, since the two candidates have
    different grounds and painting the page with either would advantage it; a single-card
    trial paints the page with the theme under test, which is what a theme owning the screen
    actually looks like. Every response appends to `aesthetics-responses.jsonl` beside this
    file with the full stimulus, the surround and both timestamps; sittings accumulate.

    Nothing is asked of you but clicks. Which colors you love is **inferred, never
    declared**: the prior mean carries only the field's general harmony models, and your own
    hues emerge from the duels — which is why the search deliberately keeps exploring hue
    rather than settling on lightness alone, and why a stated favourite would be worth less
    than a measured one anyway.
    """)
    return


@app.cell(hide_code=True)
def _():
    import json
    import math
    import random
    from datetime import datetime, timezone
    from pathlib import Path

    import colour
    import numpy as np
    import pandas as pd
    from scipy.stats import qmc

    LOG = Path(__file__).parent / "aesthetics-responses.jsonl"
    VISION_LOG = Path(__file__).parent / "calibration-responses.jsonl"
    return LOG, VISION_LOG, colour, datetime, json, math, np, pd, qmc, random, timezone


@app.cell(hide_code=True)
def _(colour, np):
    # The color engine. All appearance math runs in CAM16-UCS (Li et al. 2017 via
    # colour-science) under fixed, documented viewing conditions: D65 white, average
    # surround, L_A 40, Y_b 20 — a desktop monitor in a lit room. The screen itself is
    # uncalibrated (parked in the queue), which limits absolute claims, not the relative
    # structure the instrument learns.
    _VC = colour.VIEWING_CONDITIONS_CAM16["Average"]
    _WHITE_XY = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
    _XYZ_W = colour.xy_to_XYZ(_WHITE_XY) * 100.0
    _LA, _YB = 40.0, 20.0

    def hex_to_rgb(hexes):
        _h = [hexes] if isinstance(hexes, str) else list(hexes)
        return np.array([[int(s.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)] for s in _h])

    def rgb_to_hex(rgb):
        rgb = np.clip(np.atleast_2d(rgb), 0, 1)
        return ["#" + "".join(f"{round(255 * float(v)):02x}" for v in row) for row in rgb]

    def rgb_to_ucs(rgb):
        _xyz = colour.sRGB_to_XYZ(np.atleast_2d(rgb)) * 100.0
        _spec = colour.XYZ_to_CAM16(_xyz, _XYZ_W, L_A=_LA, Y_b=_YB, surround=_VC)
        return colour.JMh_CAM16_to_CAM16UCS(np.stack([_spec.J, _spec.M, _spec.h], axis=-1))

    def ucs_to_rgb(ucs):
        _jmh = colour.CAM16UCS_to_JMh_CAM16(np.atleast_2d(ucs))
        _spec = colour.CAM_Specification_CAM16(J=_jmh[..., 0], M=_jmh[..., 1], h=_jmh[..., 2])
        _xyz = colour.CAM16_to_XYZ(_spec, _XYZ_W, L_A=_LA, Y_b=_YB, surround=_VC)
        return np.clip(colour.XYZ_to_sRGB(_xyz / 100.0), 0.0, 1.0)

    def ucs_dist(hex_a, hex_b):
        _a = rgb_to_ucs(hex_to_rgb(hex_a))
        _b = rgb_to_ucs(hex_to_rgb(hex_b))
        return np.linalg.norm(_a - _b, axis=-1)

    def rel_lum(rgb):
        rgb = np.atleast_2d(rgb)
        _lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        return _lin @ np.array([0.2126, 0.7152, 0.0722])

    def wcag(rgb_a, rgb_b):
        _la, _lb = rel_lum(rgb_a), rel_lum(rgb_b)
        _hi, _lo = np.maximum(_la, _lb), np.minimum(_la, _lb)
        return (_hi + 0.05) / (_lo + 0.05)

    def apca_lc(txt_rgb, bg_rgb):
        """APCA-W3 0.1.9 (4g) lightness contrast, signed; |Lc| 60 ~ body-text bar."""

        def _y(rgb):
            _v = (np.atleast_2d(rgb) ** 2.4) @ np.array([0.2126729, 0.7151522, 0.0721750])
            _lift = np.where(_v < 0.022, 0.022 - _v, 0.0) ** 1.414
            return _v + np.where(_v < 0.022, _lift, 0.0)

        _yt, _yb = _y(txt_rgb), _y(bg_rgb)
        _sapc = np.where(_yb > _yt, (_yb**0.56 - _yt**0.57) * 1.14, (_yb**0.65 - _yt**0.62) * 1.14)
        return np.where(np.abs(_sapc) < 0.1, 0.0, np.where(_sapc > 0, (_sapc - 0.027) * 100, (_sapc + 0.027) * 100))

    def composite(fg_hex, alpha, bg_hex):
        """fg at `alpha` over bg, as hex — contrast is only ever stated on composited color."""
        _fg, _bg = hex_to_rgb(fg_hex)[0], hex_to_rgb(bg_hex)[0]
        return rgb_to_hex(alpha * _fg + (1 - alpha) * _bg)[0]

    def solve_j(ab, ground_rgb, ratio, lighter, iters=16):
        """Walk lightness to the contrast bar: bisect J' per row of `ab` (CAM16-UCS a', b')
        so each color meets its `ratio` (scalar or per-row) WCAG contrast against the
        ground — the theme-design rule "keep hue and saturation, walk lightness" made
        executable. Vectorized: one inverse-CAM16 call per bisection step for all rows."""
        ab = np.atleast_2d(ab)
        _n = len(ab)
        _ratio = np.broadcast_to(np.asarray(ratio, dtype=float), (_n,))
        _lo, _hi = np.full(_n, 2.0), np.full(_n, 98.0)
        _g = np.atleast_2d(ground_rgb)
        _g = np.repeat(_g, _n, 0) if len(_g) == 1 else _g
        for _ in range(iters):
            _mid = (_lo + _hi) / 2
            _rgb = ucs_to_rgb(np.column_stack([_mid, ab]))
            _too_low = wcag(_rgb, _g) < _ratio
            # Contrast rises with J when the text is lighter than the ground, falls when
            # it is darker; too-low contrast moves the bound that walks away from the page.
            if lighter:
                _lo = np.where(_too_low, _mid, _lo)
                _hi = np.where(_too_low, _hi, _mid)
            else:
                _hi = np.where(_too_low, _mid, _hi)
                _lo = np.where(_too_low, _lo, _mid)
        _j = (_lo + _hi) / 2
        return _j, ucs_to_rgb(np.column_stack([_j, ab]))

    return apca_lc, composite, hex_to_rgb, rgb_to_hex, rgb_to_ucs, solve_j, ucs_dist, ucs_to_rgb, wcag


@app.cell(hide_code=True)
def _(VISION_LOG):
    # The observer is fit in ONE place — _observer.py, the home of the measurement<->
    # preference interlock. It refits lazily from the shared vision log (cached beside it),
    # so every new vision trial sharpens these constraints automatically and no instrument
    # carries its own copy of the model. v2 fits the psychometric slope, the lapse, a
    # chromatic confusion-axis rotation, and threshold as a smooth function of ground
    # lightness — all in CAM16-UCS, the same geometry this notebook searches.
    from _observer import fit as _observer_fit

    if VISION_LOG.exists():
        _fit = _observer_fit(VISION_LOG)
        DE_MIN = {"day": _fit.de_min_day, "night": _fit.de_min_night}
        THRESH_DETAIL = {"day": _fit.de_dir_day, "night": _fit.de_dir_night}
        VISION_N = _fit.n
    else:
        # No vision data on this machine: the v2.0 fit at 748 trials (2026-09-03),
        # flagged in the analysis so the substitution is never silent.
        DE_MIN = {"day": 3.2, "night": 2.5}
        THRESH_DETAIL = {"day": {}, "night": {}}
        VISION_N = 0
    return DE_MIN, THRESH_DETAIL, VISION_N


@app.cell(hide_code=True)
def _(DE_MIN, apca_lc, composite, hex_to_rgb, math, np, rgb_to_hex, rgb_to_ucs, solve_j, ucs_dist, ucs_to_rgb, wcag):
    # Theme space and its realization. Nine axes, each in [0, 1]; polarity (light page /
    # dark page) is a block factor, not an axis — trials alternate in blocks and the model
    # carries it as a tenth, binary coordinate so learning transfers between the two
    # without conflating them.
    AXES = [
        "ground lightness",
        "ground warmth",
        "accent hue rotation",
        "accent chroma",
        "body contrast",
        "hue spread",
        "comment recede",
        "find hue",
        "find salience",
    ]

    # Horizon's own token colors anchor the accent hues — evolve, don't repaint. Night
    # anchors carry alpha in the theme file and are composited onto their page before any
    # appearance math (the rule that caught the 30%-alpha comments). Literals are ONE
    # family on purpose: Horizon's day string #F6661E and number #F77D26 sit ~3 dE apart
    # in CAM16-UCS — inside 2x your measured day threshold — so a string/number split
    # would search a distinction your eyes cannot cash.
    _ANCHORS = {
        "day": {
            "keyword": "#8A31B9",
            "function": "#1D8991",
            "string": "#F6661E",
            "ground": "#FDF0ED",
        },
        "night": {
            "keyword": composite("#B877DB", 0.902, "#1C1E26"),
            "function": composite("#25B0BC", 0.902, "#1C1E26"),
            "string": composite("#FAB795", 0.902, "#1C1E26"),
            "ground": "#1C1E26",
        },
    }
    _ROLE_ORDER = ("keyword", "function", "string")

    def _anchor_polar(polarity):
        _ucs = rgb_to_ucs(hex_to_rgb([_ANCHORS[polarity][r] for r in _ROLE_ORDER]))
        _m = np.linalg.norm(_ucs[:, 1:], axis=1)
        _h = np.degrees(np.arctan2(_ucs[:, 2], _ucs[:, 1])) % 360
        return _h, _m

    _ANCHOR_HM = {p: _anchor_polar(p) for p in ("day", "night")}

    # Realization and prior are pure functions of (theta, polarity); the caches make the
    # per-trial local-refinement candidates (and every posterior call over the pool) pay
    # for their appearance math exactly once per kernel.
    _REALIZE_CACHE = {}
    _PRIOR_CACHE = {}

    def _theta_key(theta, polarity):
        return (tuple(round(float(_v), 6) for _v in theta), polarity)

    # NOTE (marimo name mangling, measured 2026-09-03): a cell-local (underscore) name
    # referenced from inside an exported function resolves only if it is defined ABOVE
    # that function in the cell — a later definition stays unmangled in the function body
    # and NameErrors at call time under `marimo run`, invisibly to script execution.
    # Helpers therefore precede their exported callers.
    def _realize_uncached(theta, polarity):
        _t = np.asarray(theta, dtype=float)
        _night = polarity == "night"
        # Ground: lightness within the polarity's family, warmth as a signed warm/cool axis.
        _gj = 8.0 + 14.0 * _t[0] if _night else 86.0 + 9.0 * _t[0]
        _w = 2.0 * _t[1] - 1.0
        _gh = math.radians(74.0 if _w >= 0 else 256.0)
        _gm = abs(_w) * 6.0
        _g_ucs = np.array([_gj, _gm * math.cos(_gh), _gm * math.sin(_gh)])
        _g_rgb = ucs_to_rgb(_g_ucs[None])[0]
        _ground = rgb_to_hex(_g_rgb)[0]

        # Accents: rotate and spread Horizon's hues, scale their chroma, then walk each
        # color's lightness to the body-contrast bar (capped: 12:1 was chosen over the
        # theme's native 18:1 because near-maximum contrast vibrates).
        _h0, _m0 = _ANCHOR_HM[polarity]
        _mu = math.degrees(math.atan2(np.sin(np.radians(_h0)).mean(), np.cos(np.radians(_h0)).mean())) % 360
        _rot = (_t[2] - 0.5) * 120.0
        _spread = 0.4 + 1.2 * _t[5]
        _dh = (_h0 - _mu + 180.0) % 360.0 - 180.0
        _hues = (_mu + _spread * _dh + _rot) % 360.0
        _chroma = (0.6 + 0.8 * _t[3]) * _m0
        _r_body = 4.5 + 4.5 * _t[4]
        _ab = np.column_stack([_chroma * np.cos(np.radians(_hues)), _chroma * np.sin(np.radians(_hues))])
        # Neutral family (ink, comment, punctuation): the ground's own hue at a whisper of
        # chroma, so page and text agree in temperature.
        _nd = np.array([math.cos(_gh), math.sin(_gh)])
        _r_comment = max(4.5, _r_body * (0.55 + 0.35 * _t[6]))
        _r_punct = max(4.5, _r_body * 0.75)
        _neut_ab = np.array([_nd * 1.5, _nd * 2.0, _nd * 1.5])  # ink, comment, punct
        _all_ab = np.vstack([_ab, _neut_ab])
        _ratios = np.minimum([_r_body] * 3 + [max(_r_body, 5.5), _r_comment, _r_punct], 12.0)
        # WCAG sets the first target; APCA is the stricter master on dark grounds (4.5:1
        # there is only Lc ~54), so rows short of their Lc floor walk further from the page
        # until both bars hold. The 1.03 margin keeps bisection from converging a hair under.
        _target = _ratios * 1.03
        _lc_floor = np.array([60.0, 60.0, 60.0, 60.0, 45.0, 45.0])
        _g6 = np.repeat(_g_rgb[None], 6, 0)
        for _ in range(4):
            _js, _rgbs = solve_j(_all_ab, _g_rgb, _target, lighter=_night)
            _lc = apca_lc(_rgbs, _g6)
            _short = np.abs(_lc) < _lc_floor
            if not _short.any():
                break
            _target = np.where(_short, np.minimum(_target * 1.18, 14.0), _target)
        _hexes = rgb_to_hex(_rgbs)
        _roles = dict(zip(["keyword", "function", "string", "ink", "comment", "punct"], _hexes, strict=True))

        # Find highlight: a fill near the page's lightness whose loudness is the salience
        # axis. Emitted with alpha (how VSCode layers it); every constraint and every
        # rendered pixel uses the composited result.
        _s = _t[8]
        _fh = math.radians(360.0 * _t[7])
        _fm = 8.0 + 26.0 * _s
        _fj = _gj + (4.0 + 14.0 * _s) * (1 if _night else -1)
        _fill = rgb_to_hex(ucs_to_rgb(np.array([[_fj, _fm * math.cos(_fh), _fm * math.sin(_fh)]])))[0]
        _cur = composite(_fill, 0.85, _ground)
        _oth = composite(_fill, 0.45, _ground)

        # Hard floors, checked on what will actually render. One CAM16 conversion for all
        # eight colors, then plain numpy distances: colour-science's cost is per call, not
        # per color, and this block once spent 14 calls per theme (measured: 1.0 s of a
        # 1.7 s duel generation).
        _de = DE_MIN[polarity]
        _lc = apca_lc(_rgbs, _g6)
        _rr = wcag(_rgbs, _g6)
        if (_rr < 4.5 - 1e-6).any() or (np.abs(_lc[:4]) < 60).any() or (np.abs(_lc[4:]) < 45).any():
            return None
        _names = ["keyword", "function", "string", "ink", "comment"]
        _u = rgb_to_ucs(hex_to_rgb([_roles[r] for r in _names] + [_ground, _cur, _oth]))
        _K, _F, _S, _I, _C, _G, _CUR, _OTH = range(8)

        def _d(a, b):
            return float(np.linalg.norm(_u[a] - _u[b]))

        for _i in (_K, _F, _S):
            if _d(_i, _I) < 2 * _de:
                return None
            for _j2 in (_K, _F, _S):
                if _j2 > _i and _d(_i, _j2) < 2 * _de:
                    return None
        if _d(_C, _I) < _de:
            return None
        if _d(_CUR, _G) < 1.5 * _de or _d(_CUR, _OTH) < _de:
            return None
        # Text must survive sitting on either fill.
        _fills = hex_to_rgb([_cur, _oth])
        if (wcag(np.repeat(_rgbs[3:4], 2, 0), _fills) < 4.0).any() or (
            wcag(np.repeat(_rgbs[2:3], 2, 0), _fills) < 3.5
        ).any():
            return None
        _sal = min(_d(_CUR, _i) for _i in (_G, _K, _F, _S, _I))
        return {
            "ground": _ground,
            **_roles,
            "number": _roles["string"],
            "variable": _roles["ink"],
            "find_fill": _fill,
            "find_current": _cur,
            "find_other": _oth,
            "salience": round(_sal, 2),
            "body_ratio": round(float(_rr[:4].min()), 2),
        }

    def realize(theta, polarity):
        """theta in [0,1]^9 -> a full, floor-satisfying theme (hexes + meta), or None when
        the hard constraints cannot be met. Floors are constraints, never objectives: WCAG
        4.5:1 and APCA |Lc| >= 60 for body tokens (comments >= 4.5:1, |Lc| >= 45), and
        pairwise CAM16-UCS separation >= 2x your measured 104-px threshold between any two
        colored roles and ink — doubled because discrimination collapses toward glyph
        scale; the comprehension probes measure the truth of that margin directly."""
        _key = _theta_key(theta, polarity)
        if _key in _REALIZE_CACHE:
            return _REALIZE_CACHE[_key]
        _theme = _realize_uncached(theta, polarity)
        _REALIZE_CACHE[_key] = _theme
        return _theme

    # ------------------------------------------------------------------ the prior mean
    def _lab(hexes):
        _xyz = np.atleast_2d(hex_to_rgb(hexes))
        import colour as _colour

        return _colour.XYZ_to_Lab(
            _colour.sRGB_to_XYZ(_xyz),
            _colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"],
        )

    def _ou_luo_pair(lab1, lab2):
        """Two-colour harmony CH = HC + HL + HH, Ou & Luo (2006), transcribed from the
        published model. Prior-mean duty only: it tilts where the search starts, your
        clicks decide where it ends."""
        _L1, _a1, _b1 = lab1
        _L2, _a2, _b2 = lab2
        _C1, _C2 = math.hypot(_a1, _b1), math.hypot(_a2, _b2)
        _h1, _h2 = math.degrees(math.atan2(_b1, _a1)) % 360, math.degrees(math.atan2(_b2, _a2)) % 360
        _dhab = math.radians((_h1 - _h2 + 180) % 360 - 180)
        _dH = 2 * math.sqrt(max(_C1 * _C2, 0.0)) * abs(math.sin(_dhab / 2))
        _dC = math.hypot(_dH, (_C1 - _C2) / 1.46)
        _hc = 0.04 + 0.53 * math.tanh(0.8 - 0.045 * _dC)
        _hl = (0.28 + 0.54 * math.tanh(-3.88 + 0.029 * (_L1 + _L2))) + (
            0.14 + 0.15 * math.tanh(-2 + 0.2 * abs(_L1 - _L2))
        )

        def _hsy(_L, _C, _h):
            _ec = 0.5 + 0.5 * math.tanh(-2 + 0.5 * _C)
            _hs = -0.08 - 0.14 * math.sin(math.radians(_h + 50)) - 0.07 * math.sin(math.radians(2 * _h + 90))
            _y = (90 - _h) / 10
            _ey = ((0.22 * _L - 12.8) / 10) * math.exp(min(_y - math.exp(_y), 50))
            return _ec * (_hs + _ey)

        return _hc + _hl + _hsy(_L1, _C1, _h1) + _hsy(_L2, _C2, _h2)

    def _raw_prior(theta, polarity, theme):
        _t = np.asarray(theta, dtype=float)
        _hx = [theme[r] for r in _ROLE_ORDER] + [theme["ground"]]
        _labs = _lab(_hx)
        _pairs = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
        _harm = float(np.mean([_ou_luo_pair(_labs[a], _labs[b]) for a, b in _pairs]))
        # Berlyne: pleasure peaks at intermediate complexity — interior optima on the
        # complexity axes, never a monotone pull to either wall.
        _berlyne = -1.2 * sum((float(_t[i]) - 0.55) ** 2 for i in (3, 4, 5))
        # Ecological-valence stand-in until Titus names his loved colors: his stated warm
        # preference, gently.
        _warm = 0.5 * (float(_t[1]) - 0.5)
        return _harm + _berlyne + _warm

    # A fixed, deterministic candidate pool per polarity: the acquisition shops here (plus
    # per-trial local refinements around the champion), the prior is standardized here, and
    # infeasible corners are carved away by the floors rather than penalized.
    _pool_rng = np.random.default_rng(0xA55)
    POOL_THETA = _pool_rng.random((512, 9))
    POOL = {}
    _PRIOR_STATS = {}
    for _p in ("day", "night"):
        _items = []
        for _idx in range(len(POOL_THETA)):
            _th = POOL_THETA[_idx]
            _theme = realize(_th, _p)
            if _theme is not None:
                _items.append((_th, _theme, _raw_prior(_th, _p, _theme)))
        _pr = np.array([_it[2] for _it in _items])
        _PRIOR_STATS[_p] = (float(_pr.mean()), float(_pr.std() + 1e-9))
        POOL[_p] = [(_it[0], _it[1]) for _it in _items]

    def prior_mean(theta, polarity, theme=None):
        """Standardized prior utility (mean 0, sd 0.8 over the feasible pool) so the GP's
        signal variance, not the prior's arbitrary units, sets the scale."""
        _key = _theta_key(theta, polarity)
        if _key in _PRIOR_CACHE:
            return _PRIOR_CACHE[_key]
        theme = theme or realize(theta, polarity)
        if theme is None:
            _val = 0.0
        else:
            _m, _s = _PRIOR_STATS[polarity]
            _val = 0.8 * (_raw_prior(theta, polarity, theme) - _m) / _s
        _PRIOR_CACHE[_key] = _val
        return _val

    return AXES, POOL, POOL_THETA, prior_mean, realize


@app.cell(hide_code=True)
def _():
    import html as _html
    import io as _io
    import keyword as _kw
    import tokenize as _tokenize

    import _codegen

    # Stimuli are real code, embedded verbatim from this repo's own notebooks (07's
    # training loop, 05's model, _palette's tint) — the code Titus actually reads, not
    # lorem ipsum. Embedded rather than read at render time so the stimulus set is stable
    # across sessions; each record carries the snippet id.
    _SOURCES = {
        "train-loop": (
            "07-optimization-loop.py",
            """def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = (loss.item(), batch * 64 + len(X))
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
""",
            "loss",
        ),
        "build-model": (
            "05-build-model.py",
            """class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
""",
            "nn",
        ),
        "tint": (
            "_palette.py",
            '''def tint(color, toward_white):
    """The palette hue mixed toward a card's white, as a literal hex.

    For renderers that cannot take scheme names (graphviz, raw CSS):
    fills stay derived from the constants above instead of hand-tuned
    hexes appearing per notebook.
    """
    channels = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{round(c + (255 - c) * toward_white):02x}" for c in channels)
''',
            "color",
        ),
        "tensor-ops": (
            "02-tensors.py",
            """y2 = tensor_2.matmul(tensor_2.T)
torch.matmul(tensor_2, tensor_2.T, out=y3)
z1 = tensor_2 * tensor_2
z2 = tensor_2.mul(tensor_2)
# This computes the element-wise product; z1, z2 will match
torch.mul(tensor_2, tensor_2, out=z3)
t1 = torch.cat([tensor_2, tensor_2, tensor_2], dim=1)
agg = tensor_2.sum()
agg_item = agg.item()
""",
            "tensor_2",
        ),
    }

    def _tokenize_roles(code):
        """Role spans via the stdlib tokenizer: (text, role, line, col). Definition and
        call names are `function`; control words `keyword`; strings and numbers are the
        one literal family; dotted-name reads and everything else recede as variable/punct."""
        _spans = []
        _toks = list(_tokenize.generate_tokens(_io.StringIO(code).readline))
        _prev_sig = None
        for _i, _tok in enumerate(_toks):
            _typ, _txt, (_sr, _sc), (_er, _ec), _ = _tok
            if _typ in (_tokenize.NEWLINE, _tokenize.NL, _tokenize.INDENT, _tokenize.DEDENT, _tokenize.ENDMARKER):
                continue
            if _typ == _tokenize.COMMENT:
                _role = "comment"
            elif _typ == _tokenize.STRING or _typ in (
                getattr(_tokenize, "FSTRING_START", -1),
                getattr(_tokenize, "FSTRING_MIDDLE", -2),
                getattr(_tokenize, "FSTRING_END", -3),
            ):
                _role = "string"
            elif _typ == _tokenize.NUMBER:
                _role = "number"
            elif _typ == _tokenize.OP:
                _role = "punct"
            elif _typ == _tokenize.NAME:
                if _kw.iskeyword(_txt):
                    _role = "keyword"
                elif _prev_sig in ("def", "class"):
                    _role = "function"
                else:
                    _nxt = next(
                        (_t2 for _t2 in _toks[_i + 1 :] if _t2.type not in (_tokenize.NL, _tokenize.NEWLINE)),
                        None,
                    )
                    _role = "function" if (_nxt is not None and _nxt.string == "(") else "variable"
            else:
                _role = "variable"
            _spans.append({"text": _txt, "role": _role, "sr": _sr, "sc": _sc, "er": _er, "ec": _ec})
            if _typ == _tokenize.NAME or (_typ == _tokenize.OP and _txt in "()[]{}.,:"):
                _prev_sig = _txt
        return _spans

    # One page per trial, never the same twice: _codegen writes it. The four embedded
    # sources above stay as the cold-start and as a familiarity CONTROL -- a page he knows
    # is the reference against which a fresh page's reaction time is read -- but they are
    # no longer the corpus. Memoized per seed so the widget, the recorder and the analysis
    # cell all resolve the same page without regenerating it.
    # _CONTROL is built BEFORE snippet_for on purpose: a cell-local name referenced
    # inside an exported function resolves only if it is defined above that function,
    # and only under `marimo run`/`edit` -- a script run shares one namespace and
    # never mangles, so the wrong order passes every check and fails only when served.
    _CONTROL = []
    for _sid, (_prov, _code, _ident) in _SOURCES.items():
        _sp = _tokenize_roles(_code)
        _CONTROL.append(
            {
                "id": _sid,
                "provenance": _prov,
                "code": _code,
                "spans": _sp,
                "fn_ids": [_i for _i, _s in enumerate(_sp) if _s["role"] == "function"],
                "ident": _ident,
                "ident_ids": [_i for _i, _s in enumerate(_sp) if _s["text"] == _ident],
                "hash": f"control-{_sid}",
                "kind": "control",
            }
        )

    _SNIP_MEMO = {}

    def snippet_for(seed, width=None, target_kind=None, lines=None):
        """The page for this trial seed: fresh procedural or obscure-stdlib code.

        width is the column ceiling: two duel cards side by side hold about eighty columns
        at 14px, and the stimulus <pre> is overflow:hidden, so a wider line would be
        silently clipped -- a clipped stimulus is a different stimulus. Line count and role
        mix stay at the generator's calibrated default: freshness alone makes the
        comprehension probe hard now that no page is ever shown twice, and a longer page
        would trade away the identical-role-statistics property that lets two reaction
        times be compared at all.
        """
        _key = (int(seed), width, target_kind, lines)
        if _key in _SNIP_MEMO:
            return _SNIP_MEMO[_key]
        # Width and length are PREFERENCES; freshness is the requirement. The generator
        # cannot promise every shape for every seed -- a narrow 28-line page is a tall
        # order, and asking for 64 columns alone lost half the seeds -- so the request
        # relaxes in a declared order: hold the narrow column and shorten, then widen a
        # step and shorten again, and only if every combination fails fall back to a
        # control page, which is code he has already seen and therefore the last resort.
        _lines_ladder = [int(lines), int(lines) - 4, int(lines) - 8, None] if lines else [None]
        _width_ladder = [int(width), int(width) + 8, int(width) + 16, None] if width else [None]
        _s = None
        for _w in _width_ladder:
            for _ln in _lines_ladder:
                try:
                    _kw = {}
                    if _w:
                        _kw["max_width"] = _w
                    if _ln:
                        _kw["lines"] = _ln
                    if target_kind:
                        _kw["target_kind"] = target_kind
                    _s = dict(_codegen.snippet(int(seed), **_kw))
                    _s.setdefault("ident", _s.get("target"))
                    break
                except Exception:
                    continue
            if _s is not None:
                break
        if _s is None:
            _s = _CONTROL[int(seed) % len(_CONTROL)]
        _SNIP_MEMO[_key] = _s
        return _s

    _PROSE_TAIL = (
        "The consumer holds the lock only while it copies out, so a slow reader delays the "
        "next fill rather than corrupting the one in flight."
    )
    _OUTPUT_TAIL = "queue depth 3  drained 1284  blocked 0.4%  last fill 2.1 ms"
    _PROSE = (
        "A buffer is filled once per frame and drained by the consumer thread; the queue "
        "length bounds how far the two can drift apart before a reader blocks."
    )

    # The three surfaces Titus actually reads. A theme is one theme, but it is *seen* in
    # three arrangements, and the one that wins on a bare code page need not win where
    # prose and code interleave. Surface is a stimulus factor, not a theme axis: utility
    # stays defined over the theme, and the surface is logged so a later analysis can test
    # for a surface-by-theme interaction rather than assuming there is none.
    #
    #   editor    a page of code with a line of prose above it -- the plain editor
    #   panel     the Claude Code chat surface: serif turns, a raised code card between
    #             them, the proportions of an assistant answer
    #   notebook  the marimo/VSCode notebook: a centred prose column at the measured 42rem
    #             reading measure, then a raised code card, then an output block
    SURFACES = ("editor", "panel", "notebook")

    def render_card(theme, snippet, code_px, find_current=None, task=False, prose=True, surface="editor"):
        """One candidate page as HTML: prose in IBM Plex Serif 17px, code in Iosevka at the
        true editor pixel size, on the candidate ground. find_current=None hides the find
        layer; an int marks that occurrence as the current match, the rest as plain
        highlights. task=True makes every span a click target (data-tid), visually inert.
        surface selects the arrangement (see SURFACES above)."""
        _lines = snippet["code"].split("\n")
        _cursor = {}
        _out = []
        _card_open, _card_close = "", ""
        if surface in ("panel", "notebook"):
            # Machine text sits on a raised card a step off the page, the grammar the
            # applied theme uses: flat tinted panel means aside, raised card means
            # artifact. The step is taken in the ground's own hue, never toward grey.
            _g = theme["ground"].lstrip("#")
            _rgb = [int(_g[_k : _k + 2], 16) for _k in (0, 2, 4)]
            _dark = sum(_rgb) < 384
            _step = 12 if _dark else -10
            _card_bg = "#" + "".join(f"{max(0, min(255, _v + _step)):02x}" for _v in _rgb)
            _edge = "#" + "".join(f"{max(0, min(255, _v + (26 if _dark else -22))):02x}" for _v in _rgb)
            _shadow = "0 1px 3px -1px rgba(0,0,0,.35), 0 5px 14px -6px rgba(0,0,0,.28)"
            _card_open = (
                f'<div style="background:{_card_bg};border:1px solid {_edge};border-radius:4px;'
                f'padding:12px 14px;box-shadow:{_shadow};overflow:hidden">'
            )
            _card_close = "</div>"
        if prose:
            _measure = "42rem" if surface == "notebook" else "34em"
            _centre = "margin:0 auto 14px auto" if surface == "notebook" else "margin:0 0 14px 0"
            _out.append(
                f"<div style=\"font-family:'IBM Plex Serif',serif;font-size:17px;line-height:1.6;"
                f'color:{theme["ink"]};max-width:{_measure};{_centre}">{_html.escape(_PROSE)}</div>'
            )
        _out.append(_card_open)
        _out.append(
            f"<pre style=\"font-family:'IosevkaLigated Nerd Font Mono',monospace;font-size:{code_px}px;"
            f'line-height:1.5;margin:0;white-space:pre;overflow:hidden;color:{theme["punct"]}">'
        )
        _find_ids = set(snippet["ident_ids"]) if find_current is not None else set()
        for _i, _s in enumerate(snippet["spans"]):
            _r, _c = _s["sr"] - 1, _s["sc"]
            _pr, _pc = _cursor.get("r", 0), _cursor.get("c", 0)
            while _pr < _r:
                _out.append("\n")
                _pr, _pc = _pr + 1, 0
            if _c > _pc:
                _out.append(_lines[_r][_pc:_c])
            _style = f"color:{theme[_s['role']]}"
            if _s["role"] == "comment":
                _style += ";font-style:italic"
            if _i in _find_ids:
                _fill = theme["find_current"] if _i == find_current else theme["find_other"]
                _style += f";background:{_fill};border-radius:2px"
            _tid = f' data-tid="{_i}"' if task else ""
            _out.append(f'<span style="{_style}"{_tid}>{_html.escape(_s["text"])}</span>')
            _cursor = {"r": _s["er"] - 1, "c": _s["ec"]}
        _out.append("</pre>")
        _out.append(_card_close)
        if surface == "panel":
            # The diff card, because a Claude Code turn is mostly diffs and their colours
            # are part of what he reads all day. Both backgrounds are DERIVED, not searched:
            # the theme already carries a cool role colour and a warm one, and mixing each
            # into the ground keeps added/removed on the cool/warm polarity that survives
            # colour-vision deficiency while adding no dimension to a nine-dimensional
            # space that is already the binding constraint on convergence. Line text stays
            # the code ink -- a diff recolours the field, never the code.
            def _mix(_hex, _t):
                _a = theme["ground"].lstrip("#")
                _b = _hex.lstrip("#")
                return "#" + "".join(
                    f"{round(int(_a[_k : _k + 2], 16) * (1 - _t) + int(_b[_k : _k + 2], 16) * _t):02x}"
                    for _k in (0, 2, 4)
                )

            _add_bg, _del_bg = _mix(theme["function"], 0.16), _mix(theme["string"], 0.16)
            _sign = theme["comment"]
            _diff = [
                ("-", "    ferrous_voussoir_mark = stipple_plinth(ferrous_bellows_table)", _del_bg),
                ("+", "    ferrous_voussoir_mark = stipple_plinth(ferrous_bellows_table, 12)", _add_bg),
                (" ", "    with sift_gantry(opaline_voussoir_walk) as vernal_cistern_gate:", None),
                ("+", "        prime_mullion_stub = 128", _add_bg),
            ]
            _rows = []
            for _mark, _text, _bg in _diff:
                _style = f"display:block;padding:0 6px;color:{theme['punct']}"
                if _bg:
                    _style += f";background:{_bg}"
                _rows.append(
                    f'<span style="{_style}"><span style="color:{_sign}">{_mark}</span>{_html.escape(_text)}</span>'
                )
            _out.append(
                f"{_card_open}<div style=\"font-family:'IBM Plex Serif',serif;font-size:13px;"
                f'color:{theme["comment"]};margin:0 0 6px 0">edited _codegen.py</div>'
                f"<pre style=\"font-family:'IosevkaLigated Nerd Font Mono',monospace;"
                f"font-size:{code_px}px;line-height:1.5;margin:0;white-space:pre;"
                f'overflow:hidden">' + "".join(_rows) + f"</pre>{_card_close}"
            )
        if surface == "panel" and prose:
            # An assistant turn continues after the code: the second serif block is what
            # makes this the chat surface rather than a card on a page.
            _out.append(
                f"<div style=\"font-family:'IBM Plex Serif',serif;font-size:17px;line-height:1.6;"
                f'color:{theme["ink"]};max-width:34em;margin:12px 0 0 0">'
                f"{_html.escape(_PROSE_TAIL)}</div>"
            )
        if surface == "notebook":
            # A notebook cell is code plus its output, so the output block is part of the
            # stimulus: mono, one step of ink below the code, on the page rather than the card.
            _out.append(
                f"<pre style=\"font-family:'IosevkaLigated Nerd Font Mono',monospace;"
                f"font-size:{code_px}px;line-height:1.5;margin:8px 0 0 0;"
                f'color:{theme["comment"]};white-space:pre;overflow:hidden">'
                f"{_html.escape(_OUTPUT_TAIL)}</pre>"
            )
        return "".join(_out)

    DUEL_WIDTH = _codegen.DUEL_WIDTH
    return DUEL_WIDTH, SURFACES, render_card, snippet_for


@app.cell(hide_code=True)
def _(LOG, json, mo):
    _existing = [json.loads(_line) for _line in LOG.read_text().splitlines() if _line.strip()] if LOG.exists() else []
    get_responses, set_responses = mo.state(_existing)
    # The first trial of a sitting (and of every run) is gated behind a begin button;
    # inside a run the previous click anchors the clock, so render time is the baseline.
    SESSION_START_N = len(_existing)
    return SESSION_START_N, get_responses, set_responses


@app.cell(hide_code=True)
def _(DUEL_WIDTH, POOL, np, prior_mean, qmc, random, realize):
    # The preference model: a Gaussian process over theme space with a Bradley-Terry
    # likelihood on duels, fit by Laplace approximation — Chu & Ghahramani's preferential
    # GP, QUEST+'s generate-the-most-informative-trial loop on top. Reaction time enters
    # the likelihood drift-diffusion-style: decision time falls as the utility gap grows,
    # so a fast click steepens that duel's slope and a slow one flattens it toward a tie.
    # Length-scales are ARD: one per axis, estimated from the data rather than fixed, so
    # axes his choices ignore get long scales and stop costing sample efficiency. Nine
    # dimensions at ~100 duels is the binding constraint on how fast this converges, and
    # ARD is the cheapest honest way to shrink the effective dimension.
    _LS0 = np.array([0.35] * 9 + [0.9])
    _SF2 = 4.0

    def _kmat(A, B, ls=None):
        _l = _LS0 if ls is None else ls
        _d2 = (((A[:, None, :] - B[None, :, :]) / _l) ** 2).sum(-1)
        _r = np.sqrt(_d2 + 1e-12)
        return _SF2 * (1 + np.sqrt(5) * _r + 5 * _r**2 / 3) * np.exp(-np.sqrt(5) * _r)

    def _ard_scales(X, duels, lam):
        """Per-axis length-scales from a ridge-regularized linear Bradley-Terry fit.

        The principled route is maximizing the Laplace log-marginal-likelihood over ten
        log-length-scales, which costs a hundred-odd GP refits per trial and would make
        the instrument wait on itself. A linear BT model on the winner-minus-loser axis
        differences is the same question asked cheaply -- which axes move his choices --
        and its coefficient magnitudes plug straight in as relevances. Empirical-Bayes
        shortcut, deliberately: the fit runs in milliseconds and the GP keeps the
        nonlinearity.
        """
        # Shrinkage toward isotropy, because relevance is not identifiable early: with 60
        # duels the estimated ranking of nine axes was measured to be noise (0 of 4
        # simulated runs recovered the truly active axes, against reliable recovery at
        # 400). Blending toward the isotropic default with weight n/160 keeps a thin log
        # from distorting the kernel and converges on full ARD as duels accumulate.
        if len(duels) < 12:
            return _LS0.copy()
        _w_ard = min(1.0, len(duels) / 160.0)
        _D = np.array([(X[_w] - X[_l]) * _lm for (_w, _l), _lm in zip(duels, lam, strict=True)])
        _w = np.zeros(_D.shape[1])
        for _ in range(60):
            _p = 1.0 / (1.0 + np.exp(-(_D @ _w)))
            _g = _D.T @ (1.0 - _p) - 2.0 * _w
            _H = -(_D.T * (_p * (1 - _p))) @ _D - 2.0 * np.eye(_D.shape[1])
            _step = np.linalg.solve(_H, -_g)
            _w = _w + _step
            if np.abs(_step).max() < 1e-10:
                break
        _rel = np.abs(_w) / max(float(np.abs(_w).max()), 1e-9)
        _ls = 0.30 / np.sqrt(np.clip(_rel, 0.10, 1.0))
        _ls = np.clip(_ls, 0.25, 1.4)
        _ls = (1.0 - _w_ard) * _LS0 + _w_ard * _ls
        _ls[9] = 0.9
        return _ls

    def _coords(theta, polarity):
        return np.concatenate([np.asarray(theta, dtype=float), [1.0 if polarity == "night" else 0.0]])

    def duels_from(responses):
        """(X, duel index pairs, per-duel slopes, prior mean at X) from the log's duels."""
        _pts, _index = [], {}
        _duels, _rts, _paused, _sides = [], [], [], []
        for _r in responses:
            if _r.get("mode") != "duel" or _r.get("choice") not in (0, 1):
                continue
            _ids = []
            for _th in (_r["theta_a"], _r["theta_b"]):
                _key = (tuple(round(float(_v), 6) for _v in _th), _r["polarity"])
                if _key not in _index:
                    _index[_key] = len(_pts)
                    _pts.append(_coords(_th, _r["polarity"]))
                _ids.append(_index[_key])
            _win = _ids[_r["choice"]]
            _lose = _ids[1 - _r["choice"]]
            _duels.append((_win, _lose))
            _rts.append(float(_r.get("rt_ms", 2500.0)))
            _paused.append(bool(_r.get("paused")))
            # Which SIDE the winner was displayed on. Measured 2026-09-03 over 79 duels:
            # he picks the right-hand card 61% of the time (z = -1.91 against no bias).
            # Unmodelled, that lands on the utility as noise; as a fitted term it is
            # subtracted out. Reconstructible from the log, so no past duel is wasted.
            _shown = (1 - _r["choice"]) if _r.get("swap") else _r["choice"]
            _sides.append(1.0 if _shown == 0 else -1.0)
        if not _pts:
            return None
        _X = np.array(_pts)
        _paused = np.array(_paused)
        _clean = np.array(_rts)[~_paused]
        _rt_med = float(np.median(_clean)) if len(_clean) >= 8 else 2500.0
        _lam = np.clip(np.sqrt(_rt_med / np.maximum(np.array(_rts), 200.0)), 0.6, 1.8)
        # A paused trial's time says nothing about the utility gap: its choice still counts,
        # at the neutral slope, neither sharpened nor flattened by the clock.
        _lam[_paused] = 1.0
        _m = np.array([prior_mean(_x[:9], "night" if _x[9] > 0.5 else "day") for _x in _X])
        return _X, _duels, _lam, _m, np.array(_sides)

    def fit_laplace(X, duels, lam, m, sides=None, ls=None):
        """Laplace posterior over utilities, alternating with the position-bias term.

        delta is one number shared by every duel: the log-odds advantage of the card on
        the left. f and delta are identifiable because side is randomized independently
        of theme, and they are fitted by alternation -- f given delta by Newton, then
        delta given f by its own one-dimensional Newton -- which converges in two or
        three rounds at this scale.
        """
        _n = len(X)
        _K = _kmat(X, X, ls) + 1e-6 * np.eye(_n)
        _Ki = np.linalg.inv(_K)
        _f = m.copy()
        _W = np.zeros((_n, _n))  # replaced each Newton step; kept for the final _cov
        _sd = np.zeros(len(duels)) if sides is None else np.asarray(sides, dtype=float)
        _delta = 0.0
        for _round in range(3):
            # One BLAS product per Newton step instead of a Python loop over duels. Each
            # duel contributes q_k (e_win - e_lose)(e_win - e_lose)^T to the Hessian, which
            # is exactly D^T diag(q) D for the difference matrix D -- so the whole update is
            # two matrix products. Measured on the live log: the loop cost 108 ms per fit,
            # an np.add.at scatter cost 166 ms (add.at is unbuffered and slow), and this
            # costs 128 ms -- SLOWER than the loop at today's 121 duels, because building D
            # dominates at this size. Kept anyway: the loop pays one interpreter trip per
            # duel per Newton step, so it degrades linearly in log length where this is one
            # BLAS call, and 20 ms is noise against a 350 ms trial. Revisit only if a fit
            # ever dominates again. Identical arithmetic either way -- the recovery tests
            # reproduce every number.
            _D = np.zeros((len(duels), _n))
            for _k, (_w, _l) in enumerate(duels):
                _D[_k, _w] += 1.0
                _D[_k, _l] -= 1.0
            _lm_v = np.asarray(lam, dtype=float)
            _Dl = _D * _lm_v[:, None]
            for _ in range(60):
                _z = _Dl @ _f + _delta * _sd
                _p = 1.0 / (1.0 + np.exp(-_z))
                _g = _Dl.T @ (1.0 - _p)
                _q = _lm_v * _lm_v * _p * (1.0 - _p)
                _W = (_D * _q[:, None]).T @ _D
                _step = np.linalg.solve(_Ki + _W, _g - _Ki @ (_f - m))
                _f = _f + _step
                if np.abs(_step).max() < 1e-8:
                    break
            if sides is None or len(duels) < 12:
                break
            _gap = _Dl @ _f
            for _ in range(40):
                _p = 1.0 / (1.0 + np.exp(-(_gap + _delta * _sd)))
                _gd = float(_sd @ (1.0 - _p)) - 4.0 * _delta
                _hd = -float((_sd * _sd) @ (_p * (1 - _p))) - 4.0
                _d_step = -_gd / _hd
                _delta = float(np.clip(_delta + _d_step, -2.0, 2.0))
                if abs(_d_step) < 1e-10:
                    break
        _cov = np.linalg.inv(_Ki + _W)
        return _f, _cov, _Ki, _delta

    def predict(X, f, m, cov, Ki, Xs, ms, ls=None):
        _ks = _kmat(Xs, X, ls)
        _mu = ms + _ks @ (Ki @ (f - m))
        _A = Ki - Ki @ cov @ Ki
        _var = np.maximum(_SF2 - np.einsum("ij,jk,ik->i", _ks, _A, _ks), 1e-9)
        return _mu, _var, _ks, _A

    def posterior_joint(fit, thetas, polarity):
        """Mean and FULL covariance over candidates -- what P(best) needs.

        Marginal variances cannot answer "which of these is the best theme": candidates
        near each other in theme space share almost all their uncertainty, and ignoring
        that correlation would scatter the probability of being best across a cluster of
        effectively identical pages.
        """
        _Xs = np.array([_coords(_t, polarity) for _t in thetas])
        _ms = np.array([prior_mean(_t, polarity) for _t in thetas])
        _ls = fit.get("ls")
        _ks = _kmat(_Xs, fit["X"], _ls)
        _mu = _ms + _ks @ (fit["Ki"] @ (fit["f"] - fit["m"]))
        _A = fit["Ki"] - fit["Ki"] @ fit["cov"] @ fit["Ki"]
        _cov = _kmat(_Xs, _Xs, _ls) - _ks @ _A @ _ks.T
        _cov = 0.5 * (_cov + _cov.T) + 1e-8 * np.eye(len(thetas))
        return _mu, _cov

    def _h2(p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -(p * np.log(p) + (1 - p) * np.log(1 - p))

    _GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(9)
    _GH_W = _GH_W / _GH_W.sum()

    def _posterior_over(fit, thetas, polarity):
        _X, _duels, _lam, _m = fit["X"], fit["duels"], fit["lam"], fit["m"]
        _Xs = np.array([_coords(_t, polarity) for _t in thetas])
        _ms = np.array([prior_mean(_t, polarity) for _t in thetas])
        return predict(_X, fit["f"], _m, fit["cov"], fit["Ki"], _Xs, _ms, fit.get("ls"))

    _FIT_MEMO = {}

    def fitted(responses):
        # Keyed by how many duels have been answered: the fit is a pure function of the
        # log, three cells ask for the same one, and it is the cubic-cost step. Only the
        # newest entry is kept -- an older fit is never asked for again.
        _key = sum(1 for _r in responses if _r.get("mode") == "duel" and _r.get("choice") in (0, 1))
        if _key in _FIT_MEMO:
            return _FIT_MEMO[_key]
        _d = duels_from(responses)
        if _d is None:
            return None
        _X, _duels, _lam, _m, _sides = _d
        _ls = _ard_scales(_X, _duels, _lam)
        _f, _cov, _Ki, _delta = fit_laplace(_X, _duels, _lam, _m, _sides, _ls)
        _out = {
            "X": _X,
            "duels": _duels,
            "lam": _lam,
            "m": _m,
            "f": _f,
            "cov": _cov,
            "Ki": _Ki,
            "ls": _ls,
            "delta": _delta,
            "sides": _sides,
        }
        _FIT_MEMO.clear()
        _FIT_MEMO[_key] = _out
        return _out

    def mu_at(fit, thetas, polarity):
        """Posterior-mean utility at arbitrary thetas — the analysis cell's window in."""
        return _posterior_over(fit, thetas, polarity)[0]

    # ---- candidate generation: global reach PLUS bred refinement --------------------
    #
    # The pool was 512 points drawn once with a fixed seed, and the only refinement was 48
    # jittered children of the single argmax champion. Measured against a synthetic
    # two-mode utility (see the escape test in the commit that added this), that design
    # has good REACH -- 512 uniform points cover nine dimensions well enough for Thompson
    # sampling to discover a distant better mode -- and poor RESOLUTION: nothing can sit
    # between pool points except near one champion, at a fixed step size.
    #
    # The first attempt at fixing it replaced the pool with bred children and lost the
    # reach, scoring *worse* in simulation. So candidates are now reach and refinement
    # together, every trial:
    #
    #   standing    the full pool plus a SMALL fresh scrambled-Sobol block (64), advanced
    #               by trial number. The pool is a codebook: revisiting the same points
    #               concentrates information there and sharpens the posterior, where a
    #               fully churning candidate set spreads every duel over ground never
    #               seen again -- measured, a 512-per-trial immigrant flood scored worse
    #               than no immigrants at all. Sixty-four is the measured sweet spot: a
    #               trickle of genuinely new ground each trial, never enough to drown the
    #               codebook, and enough that no region stays permanently unvisited.
    #   elites      the best already-evaluated themes, chosen for spread as well as for
    #               posterior mean, so refinement is not confined to one basin.
    #   mutation    Gaussian children of each elite, per-axis sigma proportional to the
    #               ARD length-scale: fine steps where utility actually turns, coarse
    #               where the model has learned that nothing rides.
    #   crossover   uniform per-axis recombination between elite pairs. Worth having
    #               because the axes are semi-separable (ground, accent set, comment
    #               recession, find-highlight): a good ground and a good accent set
    #               recombine into a plausible page, the building-block case where
    #               crossover earns its keep rather than adding noise.
    #
    # Infeasible children are dropped by the floors rather than penalized, so the whole
    # candidate set is legible-by-construction.
    def _sobol_block(n_log2, offset_blocks):
        """A power-of-two block from one fixed scrambled Sobol sequence.

        Deterministic in the block index, so trial n always draws the same immigrants and
        successive trials continue the sequence instead of resampling the same clumps.
        random() rather than random_base2(): the latter also demands that the TOTAL drawn
        be a power of two, which a fast-forwarded engine cannot satisfy. n itself is a
        power of two, which is what the balance property needs.
        """
        _n = 2**n_log2
        _eng = qmc.Sobol(d=9, scramble=True, seed=0xC0FFEE)
        _skip = (offset_blocks * _n) % 65536
        if _skip:
            _eng.fast_forward(_skip)
        return _eng.random(_n)

    def candidates(fit, polarity, nprng, n_trial=0, n_elite=10, n_mut=20, n_cross=48, imm_log2=6):
        """(candidates, index where the standing global stratum ends) for this trial."""
        _out, _seen = [], set()

        def _add(_t, _theme=None):
            _t = np.clip(np.asarray(_t, dtype=float), 0.0, 1.0)
            _key = tuple(np.round(_t, 4))
            if _key in _seen:
                return
            _th = _theme if _theme is not None else realize(_t, polarity)
            if _th is None:
                return
            _seen.add(_key)
            _out.append((_t, _th))

        for _t, _theme in POOL[polarity]:
            _add(_t, _theme)
        for _imm in _sobol_block(imm_log2, n_trial):
            _add(_imm)
        _n_standing = len(_out)
        if fit is None:
            return _out, _n_standing
        _want = 1.0 if polarity == "night" else 0.0
        _arch = [_x[:9] for _x in fit["X"] if abs(_x[9] - _want) < 0.5]
        _seed_set = _arch + [_c[0] for _c in _out]
        _mu = _posterior_over(fit, _seed_set, polarity)[0]
        _ls = fit.get("ls")
        _top = np.argsort(-_mu)[: 6 * n_elite]
        # Elites for spread as well as for mean: the best few, then the most different
        # among the rest of the leaders, so refinement is not confined to one basin.
        # Deliberately NOT Thompson-sampled elites: tried, and measured clearly worse
        # (reach 3/12 runs, t = -2.6). Refining around a high-variance region spends the
        # mutation budget on noise and displaces elites that are actually good; explore
        # belongs in the standing stratum, refine belongs where the mean is high.
        _keep = [int(_i) for _i in _top[: n_elite // 2]]
        _w = 1.0 / (_LS0[:9] if _ls is None else _ls[:9])
        _P = np.array([np.asarray(_seed_set[int(_i)]) * _w for _i in _top])
        _top_list = list(_top)
        while len(_keep) < n_elite and len(_keep) < len(_top):
            _chosen = [_top_list.index(_i) for _i in _keep if _i in _top_list]
            if not _chosen:
                _chosen = [0]
            _d = np.min(np.linalg.norm(_P[:, None, :] - _P[None, _chosen, :], axis=-1), axis=1)
            _d[_chosen] = -1.0
            _keep.append(int(_top[int(np.argmax(_d))]))
        _elites = [np.asarray(_seed_set[_i]) for _i in _keep]
        _sig = 0.25 * (_LS0[:9] if _ls is None else _ls[:9])
        for _e in _elites:
            _add(_e)
            for _child in np.clip(_e[None, :] + nprng.normal(0, _sig, (n_mut, 9)), 0, 1):
                _add(_child)
        if len(_elites) >= 2:
            for _ in range(n_cross):
                _i, _j = nprng.choice(len(_elites), 2, replace=False)
                _mask = nprng.random(9) < 0.5
                _add(np.where(_mask, _elites[_i], _elites[_j]))
        return _out, _n_standing

    def best_set(fit, polarity, thetas, samples=2048, mass=0.5, seed=0, radius=0.9):
        """Which theme is best, or which SET is -- as a distribution over argmaxes.

        Three things have to be right for this to answer the question honestly.

        Sample the JOINT posterior, because candidates near each other share almost all
        their uncertainty and marginals would scatter the probability of being best across
        a cluster of effectively identical pages.

        Then GROUP before counting. A candidate set of eight hundred contains many pages
        that differ by less than he could ever see, and each sibling steals argmax mass
        from the others: measured on the real log, the leader held 1.6% while the report
        claimed a plateau -- a number that says nothing about whether one theme leads. Mass
        belongs to a perceptually distinct group, not to a coordinate.

        And read the verdict off CUMULATIVE mass, not an absolute cutoff. The credible set
        is the smallest group of groups holding `mass` of the argmax probability: one group
        over half of it is a winner; a handful sharing it is a real plateau; and when even
        the top group is thin, the honest answer is that the log cannot yet tell -- which
        is a state this reports rather than dressing up as a plateau.
        """
        _mu, _cov = posterior_joint(fit, thetas, polarity)
        try:
            _L = np.linalg.cholesky(_cov)
        except np.linalg.LinAlgError:
            _w, _V = np.linalg.eigh(_cov)
            _L = _V * np.sqrt(np.maximum(_w, 1e-12))
        _Z = np.random.default_rng(seed).standard_normal((len(thetas), samples))
        _F = _mu[:, None] + _L @ _Z
        _p = np.bincount(np.argmax(_F, axis=0), minlength=len(thetas)) / float(samples)

        # Group into perceptually distinct themes: greedy, best-first, in length-scale
        # scaled theta space, so a group is "themes his eyes and this model cannot
        # separate" rather than an arbitrary grid cell.
        _w_ax = 1.0 / (_LS0[:9] if fit.get("ls") is None else fit["ls"][:9])
        _P = np.array([np.asarray(_t) * _w_ax for _t in thetas])
        _order = np.argsort(-_p)
        _reps, _group_of = [], np.full(len(thetas), -1)
        for _i in _order:
            if _reps:
                _d = np.linalg.norm(_P[_reps] - _P[_i], axis=1)
                _j = int(np.argmin(_d))
                if _d[_j] <= radius:
                    _group_of[_i] = _j
                    continue
            _group_of[_i] = len(_reps)
            _reps.append(int(_i))
        _gp = np.zeros(len(_reps))
        for _i in range(len(thetas)):
            _gp[_group_of[_i]] += _p[_i]
        _gorder = np.argsort(-_gp)
        _keep, _acc = [], 0.0
        for _g in _gorder:
            _keep.append(int(_g))
            _acc += _gp[_g]
            if _acc >= mass:
                break
        _lead = float(_gp[_gorder[0]])
        _verdict = "single" if _lead > 0.5 else ("plateau" if _lead > 0.12 else "undecided")
        return {
            "p_best": _p,
            "order": _order,
            "groups": _reps,
            "group_p": _gp,
            "group_order": _gorder,
            "credible": [_reps[_g] for _g in _keep],
            "credible_p": [float(_gp[_g]) for _g in _keep],
            "lead": _lead,
            "mu": _mu,
            "verdict": _verdict,
        }

    def spread_out(thetas, idx, k, ls=None):
        """k maximally different members of a set -- greedy max-min in scaled theta space.

        A plateau is only useful if its members actually look different; picking the top-k
        by probability would return k variations of one page.
        """
        if not idx:
            return []
        _w = 1.0 / (_LS0[:9] if ls is None else ls[:9])
        _P = np.array([np.asarray(thetas[_i]) * _w for _i in idx])
        _pick = [0]
        while len(_pick) < min(k, len(idx)):
            _d = np.min(np.linalg.norm(_P[:, None, :] - _P[None, _pick, :], axis=-1), axis=1)
            _d[_pick] = -1.0
            _pick.append(int(np.argmax(_d)))
        return [idx[_i] for _i in _pick]

    def schedule_mode(n, n_duels):
        """Twenty-four-trial polarity blocks, each a run of sixteen duels, then four
        comprehension probes, then four find hunts — same-kind trials batched so one
        instruction serves a whole run and no click is spent re-reading. All-duel until the
        model has something to probe."""
        _pol = ("day", "night")[(n // 24) % 2]
        if n_duels < 6:
            return _pol, "duel"
        _slot = n % 24
        if _slot < 16:
            return _pol, "duel"
        if _slot < 20:
            return _pol, "comprehension"
        return _pol, "search"

    def run_info(n, n_duels):
        """(polarity, mode, position within the run, run length) for trial n."""
        _pol, _mode = schedule_mode(n, n_duels)
        if n_duels < 6:
            return _pol, _mode, min(n_duels, 5), 6
        _slot = n % 24
        if _slot < 16:
            return _pol, _mode, _slot, 16
        if _slot < 20:
            return _pol, _mode, _slot - 16, 4
        return _pol, _mode, _slot - 20, 4

    # Deterministic given the log, so a memo keyed by trial number is a pure cache: three
    # cells ask for the same trial and pay for one fit.
    _TRIAL_MEMO = {}

    def trial_for(n, responses):
        """The nth trial, generated to maximize expected information about the utility.

        Duels: candidates are bred fresh (see candidates() -- elites, mutation, crossover,
        Sobol immigrants), one arm is a Thompson sample's argmax over them (explore where
        the optimum might be), the other the challenger with maximal expected information
        gain about the duel's outcome — plus a 7% share of uniform feasible pairs against
        model misspecification and, once a champion exists, a 5% share of
        champion-vs-worst anchors that double as
        engagement breathers and sanity checks. Comprehension probes ride the Thompson
        argmax; find hunts hold the champion's page and sweep the find axes uniformly."""
        if n in _TRIAL_MEMO:
            return _TRIAL_MEMO[n]
        _hist = responses[:n]
        _n_duels = sum(1 for _r in _hist if _r.get("mode") == "duel")
        _pol, _mode = schedule_mode(n, _n_duels)
        _rng = random.Random(n * 2654435761 % (2**31))
        _nprng = np.random.default_rng(n * 7919 + 13)
        _pool = POOL[_pol]
        _fit = fitted(_hist) if _n_duels >= 4 else None

        def _pick_pool(k):
            _idx = _rng.sample(range(len(_pool)), k)
            return [_pool[_i] for _i in _idx]

        _kind = "probe"
        if _mode == "duel":
            if _fit is None or _rng.random() < 0.07:
                (_ta, _tha), (_tb, _thb) = _pick_pool(2)
            else:
                _bred, _n_std = candidates(_fit, _pol, _nprng, n_trial=n)
                _cand = [_b[0] for _b in _bred]
                _cthemes = [_b[1] for _b in _bred]
                _mu, _var, _ks, _A = _posterior_over(_fit, _cand, _pol)
                if _rng.random() < 0.054:
                    _kind = "anchor"
                    _i1, _i2 = int(np.argmax(_mu)), int(np.argmin(_mu))
                else:
                    _kind = "eig"
                    _samp = _mu + np.sqrt(_var) * _nprng.standard_normal(len(_mu))
                    # Stratified Thompson: the explore/exploit split is DECLARED, not left
                    # to however many candidates each stratum happened to contribute.
                    # Measured: adding local children silently pulled the sampled argmax
                    # toward the incumbent's basin and cost reach (paired diff -0.14 on the
                    # two-mode test). Drawing the champion arm from the global stratum half
                    # the time restores it without giving up refinement.
                    _lo, _hi = (_n_std, len(_cand)) if (_rng.random() < 0.5 and _n_std < len(_cand)) else (0, _n_std)
                    _i1 = _lo + int(np.argmax(_samp[_lo:_hi]))
                    _cross = _kmat(
                        np.array([_coords(_t, _pol) for _t in _cand]),
                        np.array([_coords(_cand[_i1], _pol)]),
                        _fit.get("ls"),
                    )[:, 0] - np.einsum("ij,jk,k->i", _ks, _A, _ks[_i1])
                    _mud = _mu - _mu[_i1]
                    _s2 = np.maximum(_var + _var[_i1] - 2 * _cross, 1e-9)
                    _pbar = 1.0 / (1.0 + np.exp(-_mud / np.sqrt(1 + np.pi * _s2 / 8)))
                    _cond = (
                        _h2(1.0 / (1.0 + np.exp(-(_mud[:, None] + np.sqrt(_s2)[:, None] * _GH_X[None, :])))) @ _GH_W
                    )
                    _eig = _h2(_pbar) - _cond
                    _eig[_i1] = -1.0
                    _i2 = int(np.argmax(_eig))
                _ta, _tha = _cand[_i1], _cthemes[_i1]
                _tb, _thb = _cand[_i2], _cthemes[_i2]
            _snip = n * 7919 + 17
            _trial = {
                "mode": "duel",
                # Both arms share surface and page: a duel varies the theme, nothing else.
                # A duel is judged full screen, so the sample must BE a page -- a fourteen
                # line block adrift in half a screen tells him nothing about how a screen
                # of this theme reads. Long enough to fill the half, and smaller type,
                # which is also what a full screen at this pixel density looks like in the
                # editor itself. Both stay logged as stimulus parameters.
                "snippet_width": DUEL_WIDTH,
                "snippet_lines": 28,
                "surface": ("editor", "panel", "notebook")[n % 3],
                "kind": _kind,
                "polarity": _pol,
                "theta_a": [round(float(_v), 6) for _v in _ta],
                "theta_b": [round(float(_v), 6) for _v in _tb],
                "theme_a": _tha,
                "theme_b": _thb,
                "snippet": _snip,
                "code_px": 12 if _rng.random() < 0.5 else 13,
                "swap": _rng.random() < 0.5,
                "find_current": None,  # filled by the widget cell from the snippet
            }
        elif _mode == "comprehension":
            if _fit is not None and _rng.random() > 0.25:
                _bred = candidates(_fit, _pol, _nprng, n_trial=n)[0]
                _mu, _var, _ks, _A = _posterior_over(_fit, [_b[0] for _b in _bred], _pol)
                _samp = _mu + np.sqrt(_var) * _nprng.standard_normal(len(_mu))
                _ta, _tha = _bred[int(np.argmax(_samp))]
            else:
                _ta, _tha = _pool[_rng.randrange(len(_pool))]
            # Comprehension probes require a CALL-site target (Titus spotted this): a name
            # at its `def` sits at a line start, at a predictable indent, one or two to a
            # page, and is found far faster than the same name inside an expression. Mixing
            # the two kinds puts a step in the task's difficulty, and reaction time then
            # measures which kind was drawn rather than how the theme reads -- 12 of 60
            # probe pages were handing out the easy kind.
            _snip = n * 7919 + 17
            _trial = {
                "mode": "comprehension",
                "surface": "editor",
                "target_kind": "call",
                # A page, not a snippet: fourteen lines centred on an 8K screen is an island
                # spanning a quarter of the field, and a probe needs distractors to reject --
                # accuracy was saturated at 100% over twenty probes, and a 28-line page
                # offers ~97 identifiers to reject instead of ~28.
                "snippet_lines": 28,
                "kind": "task",
                "polarity": _pol,
                "theta_a": [round(float(_v), 6) for _v in _ta],
                "theme_a": _tha,
                "snippet": _snip,
                # One page on the screen, so it is read at the size his editor actually uses --
                # a duel halves the screen and takes the smaller end of the same range.
                "code_px": 15 if _rng.random() < 0.5 else 16,
            }
        else:  # search
            if _fit is not None:
                _bred = candidates(_fit, _pol, _nprng, n_trial=n)[0]
                _mu = _posterior_over(_fit, [_b[0] for _b in _bred], _pol)[0]
                _base = np.array(_bred[int(np.argmax(_mu))][0])
            else:
                _base = np.array(_pool[_rng.randrange(len(_pool))][0])
            # Sweep the find axes over their whole range on an otherwise-fixed page: the
            # regression of time-to-find on predicted salience needs coverage, not comfort.
            _bt = _base.copy()
            _bt[7], _bt[8] = _rng.random(), _rng.random()
            _tha = realize(_bt, _pol)
            if _tha is None:
                _idx = _rng.randrange(len(_pool))
                _bt, _tha = np.array(_pool[_idx][0]), _pool[_idx][1]
            _snip = n * 7919 + 17
            _trial = {
                "mode": "search",
                "surface": "editor",
                "snippet_lines": 28,
                "kind": "task",
                "polarity": _pol,
                "theta_a": [round(float(_v), 6) for _v in _bt],
                "theme_a": _tha,
                "snippet": _snip,
                "code_px": 15 if _rng.random() < 0.5 else 16,
            }
        _TRIAL_MEMO[n] = _trial
        return _trial

    return best_set, candidates, fitted, mu_at, posterior_joint, run_info, schedule_mode, spread_out, trial_for


@app.cell(hide_code=True)
def _(get_responses, mo):
    # The trial number doubles as a staleness indicator: if it disagrees with the stimulus
    # below, the surface lagged and the guard is dropping clicks. Instructions live in the
    # instrument's own bar, where the eye already is.
    _n = len(get_responses())
    mo.hstack([mo.md(f"**Trial {_n + 1}**")], justify="center")
    return


@app.cell(hide_code=True)
def _(SESSION_START_N, get_responses, mo, random, render_card, run_info, schedule_mode, snippet_for, trial_for):
    _n = len(get_responses())
    _t = trial_for(_n, get_responses())
    _nd = sum(1 for _r in get_responses() if _r.get("mode") == "duel")
    _pol, _mode, _pos, _len = run_info(_n, _nd)
    # Gate (begin button) at the first trial of a sitting and at every run boundary — the
    # moments where a new instruction must be read; inside a run the previous click is
    # the anchor and the clock starts at render.
    _nd_prev = _nd - (1 if _n > 0 and get_responses()[-1].get("mode") == "duel" else 0)
    _gate = _n == SESSION_START_N or (_n > 0 and schedule_mode(_n - 1, _nd_prev) != (_pol, _mode))
    _rng = random.Random(_n * 48271 % (2**31))
    _snip = snippet_for(_t["snippet"], _t.get("snippet_width"), _t.get("target_kind"), _t.get("snippet_lines"))
    _neutral = {"day": "#d8d2cf", "night": "#14161c"}[_t["polarity"]]
    # A single-card trial has ONE ground, so band, page and card are one continuous field:
    # a neutral band around the card would put a third colour between the sample and the
    # page that was just painted to match it. A duel keeps the neutral, since its band
    # surrounds two different grounds and must favour neither.
    _strip = _neutral if _t["mode"] == "duel" else _t["theme_a"]["ground"]
    # A duel's surround must not favour either arm, so it stays the polarity's neutral; a
    # single-card trial paints the page with the theme under test.
    _page_bg = _neutral if _t["mode"] == "duel" else _t["theme_a"]["ground"]
    # Whether the page runs marimo's dark theme is decided by the ground it took, not by
    # the polarity label: a light-ish night candidate should still get light-theme prose.
    _pg = _page_bg.lstrip("#")
    _page_dark = sum(int(_pg[_k : _k + 2], 16) for _k in (0, 2, 4)) < 384
    _ptxt = {"day": "#3a3532", "night": "#b8bcc6"}[_t["polarity"]]

    import anywidget
    import traitlets

    class _ThemeTrial(anywidget.AnyWidget):
        # The clock's baseline is the latest reveal; the click stamps the end; both ride the
        # synced traits into the record. First click only — later clicks and clicks on an
        # orphaned stale widget record nothing (the guard double-checks the trial number).
        # Gated trials (first of a sitting, first of a run) start behind an opaque cover
        # with the run's instruction and a begin button; the rest reveal at render, since
        # the click that produced them is the anchor. Pausing re-covers the stimulus (an
        # exposed one lets a decision form off the clock) and swallows clicks; revealing
        # again re-baselines. Tab-hide and 25 s of idling auto-pause. A trial paused after
        # its first reveal carries paused=true: its time is read as a near-tie, never as
        # evidence.
        _esm = """
        function render({ model, el }) {
          let t0 = -1;               // the clock's baseline: the latest reveal
          let revealed = false;      // every trial starts hidden
          let pausedNow = false;
          let pauses = 0;            // pauses AFTER the first reveal only
          el.style.cssText = "display:block;width:100%";
          // The WHOLE page takes the surround, not just the band. Titus judges these in
          // full screen, and in the vision instrument's own rule adaptation state is part
          // of the measurement: a dark candidate read inside a light page is measured in
          // the wrong adaptation state, and the surround dominates the field in full
          // screen. For a duel the surround is the polarity's fixed neutral -- the two
          // candidates have DIFFERENT grounds and painting the page with either would
          // advantage it -- while a single-card trial takes the candidate's own ground,
          // which is what a theme owning the screen actually looks like.
          // The page joins the polarity under test. Titus judges these in full screen,
          // where the surround is most of what the eye adapts to, and adaptation state is
          // part of the measurement by the vision instrument's own rule -- a dark
          // candidate read inside a light page is measured in the wrong state.
          //
          // Two earlier attempts were wrong in instructive ways. Painting body with
          // guessed container selectors left marimo's own content column white over the
          // dark field; walking up from this widget and clearing ancestor backgrounds
          // fixed the field but not the PROSE, which lives in sibling cells and stayed
          // dark-on-dark. The framework already has the switch: marimo keys its whole
          // theme off a `dark` class on the root element, so flipping that gets every
          // container, every piece of prose and every default ink coherently, and only
          // the exact ground still has to be painted on top.
          const surround = model.get("page_bg");
          document.documentElement.classList.toggle("dark", !!model.get("page_dark"));
          let pageStyle = document.getElementById("theme-trial-surround");
          if (!pageStyle) {
            pageStyle = document.createElement("style");
            pageStyle.id = "theme-trial-surround";
            document.head.appendChild(pageStyle);
          }
          // The exact ground goes through marimo's OWN custom property rather than over
          // the top of it: its page container is .bg-background reading
          // --background: light-dark(#fff, #181c1a), so setting that variable paints the
          // real field the real colour, and cards keep a step off it. Overriding a
          // framework with !important on guessed selectors is the smell that its hook has
          // not been found yet; this is the hook.
          const step = model.get("page_dark") ? 14 : -12;
          const hex = surround.replace("#", "");
          const card = "#" + [0, 2, 4].map((k) => {
            const v = parseInt(hex.slice(k, k + 2), 16) + step;
            return Math.max(0, Math.min(255, v)).toString(16).padStart(2, "0");
          }).join("");
          pageStyle.textContent =
            ":root, html { --background: " + surround + " !important;" +
            " --card: " + card + " !important; --popover: " + card + " !important; }" +
            "html, body { background: " + surround + " !important; }" +
            "body { transition: background 140ms linear; }";
          const wrap = document.createElement("div");
          // Full-bleed: marimo's prose column is ~700 px, too narrow for two code pages
          // at true editor sizes; the band breaks out to the viewport, capped at 1400 px.
          // Inline while gated (so the page around it stays reachable), viewport-owning
          // once a duel is revealed: he judges these in full screen and half a screen of
          // unrelated page would be half the adapting field. Pausing returns it inline.
          const inlineCss =
            `background:${model.get("strip_bg")};padding:18px;` +
            `border-radius:10px;display:flex;flex-direction:column;gap:14px;` +
            `position:relative;left:50%;transform:translateX(-50%);` +
            `width:min(96vw, 1400px);box-sizing:border-box;color:${model.get("ink")}`;
          // EVERY trial takes the screen once revealed, not only duels: he judges in full
          // screen, and a comparison that owns the field while a probe shares it with the
          // page would be measured in two different conditions.
          const fullCss =
            `background:${model.get("strip_bg")};padding:0;display:flex;` +
            // Above every piece of marimo's own chrome: its logo sat over the instruction
            // chip and its scrollbar showed at the edge. A trial that takes the screen
            // takes all of it -- stacking below the host's furniture and then padding
            // around it would be a hack that breaks whenever the host moves something.
            `flex-direction:column;gap:0;position:fixed;inset:0;z-index:2147483000;` +
            `box-sizing:border-box;color:${model.get("ink")}`;
          wrap.style.cssText = inlineCss;
          const setFrame = (full) => {
            // Reparented to <body> rather than trusting a big z-index: marimo's logo sat
            // over the instruction bar even at z-index 2147483000, because z-index only
            // orders siblings within a stacking context and the widget's container is
            // inside one of marimo's. Moving the frame to the root context is the fix that
            // holds however the host rearranges its own furniture; it returns to its slot
            // when the frame goes inline, so nothing leaks.
            if (full && wrap.parentElement !== document.body) {
              document.body.appendChild(wrap);
            } else if (!full && wrap.parentElement !== el) {
              el.appendChild(wrap);
            }
            wrap.style.cssText = full ? fullCss : inlineCss;
            top.style.padding = full ? "14px 20px 10px 20px" : "0";
            stage.style.flex = full ? "1 1 auto" : "0 0 auto";
            stage.style.minHeight = full ? "0" : "";
          };
          // The instruction bar: what kind of run (chip), what to do (question, large),
          // where you are in the run (progress). One glance, then act.
          const top = document.createElement("div");
          top.style.cssText = "display:flex;align-items:center;gap:16px";
          const chip = document.createElement("div");
          chip.textContent = model.get("chip");
          chip.style.cssText = "font-family:'IBM Plex Serif',serif;font-size:12px;" +
            "letter-spacing:.14em;text-transform:uppercase;opacity:.75;white-space:nowrap;" +
            "border:1px solid currentColor;border-radius:999px;padding:3px 12px";
          const prompt = document.createElement("div");
          prompt.innerHTML = model.get("prompt_html");
          prompt.style.cssText = "flex:1 1 0;font-family:'IBM Plex Serif',serif;" +
            "font-size:19px;line-height:1.3";
          const progress = document.createElement("div");
          progress.textContent = model.get("progress");
          progress.style.cssText = "font-family:'IBM Plex Serif',serif;font-size:13px;" +
            "opacity:.6;white-space:nowrap;font-variant-numeric:tabular-nums";
          top.appendChild(chip);
          const btnStyle = "font-family:'IBM Plex Serif',serif;background:transparent;" +
            "color:inherit;border:1px solid currentColor;border-radius:8px;cursor:pointer";
          const pauseBtn = document.createElement("button");
          pauseBtn.textContent = "pause";
          pauseBtn.title = "hide the trial; the clock re-baselines when you reveal it again";
          pauseBtn.style.cssText = btnStyle + ";font-size:13px;opacity:.55;padding:2px 10px;" +
            "visibility:hidden";
          top.appendChild(prompt);
          top.appendChild(progress);
          top.appendChild(pauseBtn);
          // The stimulus row keeps its box in the layout at all times; the cover is an
          // opaque overlay on exactly that box, so reveal/pause never move the page.
          const stage = document.createElement("div");
          // Grows in the full-screen frame so the halves reach the bottom of the viewport;
          // inline it keeps its content height.
          stage.style.cssText = "position:relative;display:flex;flex-direction:column";
          // A duel splits the VIEWPORT rather than laying two cards on a shared page.
          // Each half is full-bleed in its own ground, so each candidate is judged in its
          // own adaptation state -- the same reason the page takes the ground on a
          // single-card trial. A neutral surround would put every card on a mismatched
          // field, and painting the shared page with either candidate's ground would
          // advantage that one; splitting is the only arrangement that is both matched and
          // symmetric. No gap and no radius between the halves: a gutter would reintroduce
          // a third colour between the two things being compared.
          const isDuel = model.get("mode") === "duel";
          const row = document.createElement("div");
          // Edge to edge, because he judges in full screen and a centred pair gave back
          // adaptation area to a neutral surround for no gain. Each half owns its ground
          // with no gutter between them -- the point of splitting.
          row.style.cssText = isDuel
            ? "display:flex;gap:0;align-items:stretch;width:100%;visibility:hidden;flex:1 1 auto"
            : "display:flex;gap:16px;justify-content:center;align-items:center;" +
              "width:100%;visibility:hidden;flex:1 1 auto";
          const cover = document.createElement("div");
          cover.style.cssText = `position:absolute;inset:0;display:flex;flex-direction:column;` +
            `align-items:center;justify-content:center;gap:16px;border-radius:10px;` +
            `background:${model.get("strip_bg")};border:1px dashed currentColor;` +
            `font-family:'IBM Plex Serif',serif;font-size:15px;box-sizing:border-box`;
          const coverText = document.createElement("div");
          coverText.style.cssText = "opacity:.75;font-size:17px;max-width:38em;text-align:center;" +
            "line-height:1.5";
          const goBtn = document.createElement("button");
          goBtn.style.cssText = btnStyle + ";font-size:16px;padding:8px 26px;letter-spacing:.02em";
          cover.appendChild(coverText);
          cover.appendChild(goBtn);
          const setCover = (text, label) => {
            coverText.textContent = text;
            goBtn.textContent = label;
            cover.style.display = "flex";
            row.style.visibility = "hidden";
            pauseBtn.style.visibility = "hidden";
            setFrame(false);
          };
          let idleTimer = null;
          const armIdle = () => {
            if (idleTimer) clearTimeout(idleTimer);
            idleTimer = setTimeout(() => doPause("paused after 25 s without a click"), 25000);
          };
          const reveal = () => {
            setFrame(true);
            cover.style.display = "none";
            row.style.visibility = "visible";
            pauseBtn.style.visibility = "visible";
            revealed = true;
            pausedNow = false;
            t0 = performance.now();   // baseline re-initialized on EVERY reveal
            armIdle();
          };
          const doPause = (why) => {
            if (!revealed || pausedNow) return;
            pausedNow = true;
            pauses += 1;
            if (idleTimer) clearTimeout(idleTimer);
            setCover((why || "paused") + " \u2014 the stimulus is hidden; " +
              "the clock re-baselines when you resume", "resume");
          };
          goBtn.onclick = reveal;
          if (model.get("gate")) {
            setCover(model.get("gate_text"), "begin");
          } else {
            reveal();
          }
          pauseBtn.onclick = () => doPause("paused");
          const onVis = () => { if (document.hidden) doPause("paused while the tab was hidden"); };
          document.addEventListener("visibilitychange", onVis);
          const pick = (tid) => {
            if (!revealed || pausedNow) return;
            if (idleTimer) clearTimeout(idleTimer);
            model.set("clicks", model.get("clicks") + 1);
            model.set("choice", tid);
            model.set("pauses", pauses);
            model.set("t_render", t0);
            model.set("t_click", performance.now());
            model.save_changes();
          };
          model.get("cards").forEach((c, i) => {
            const card = document.createElement("div");
            // The surface's own blocks (prose, code card, output) are SIBLINGS, so they go
            // inside one block-level child: a flex parent would otherwise lay them out as
            // a row and clip the code mid-line (measured -- it looked exactly as broken as
            // it sounds).
            const inner = document.createElement("div");
            // Capped so the pair straddles the centre rather than each block sprawling to
            // its own outer edge.
            inner.style.cssText = isDuel
              ? "width:100%;max-width:min(720px, 44vw);min-width:0"
              : "width:100%;max-width:100%;min-width:0";
            inner.innerHTML = c.html;
            card.appendChild(inner);
            card.style.cssText = isDuel
              // Grounds stay full-bleed (adaptation), but the CONTENT of each half hugs the
              // seam: left half right-aligned, right half left-aligned, so both code blocks
              // sit inside the middle half of the screen. On an 8K panel the outer edges
              // are viewed at an angle steep enough to skew the judgement, and code being
              // left-bound put the left candidate out there (Titus). Symmetric about the
              // centre, so neither candidate gains — the fairness property is preserved
              // while the optics stop biasing the answer.
              ? `background:${c.ground};padding:26px 34px;flex:1 1 0;min-width:0;` +
                `overflow:hidden;display:flex;align-items:center;` +
                `justify-content:${i === 0 ? "flex-end" : "flex-start"}`
              // A single-card trial centres on the screen, on the same ground the page
              // took, with no radius: card, band and page are one continuous field.
              : `background:${c.ground};padding:28px 32px;max-width:min(1100px, 92vw);` +
                `min-width:0;overflow:hidden;display:flex;align-items:center;` +
                `justify-content:center`;
            if (isDuel) {
              card.style.cursor = "pointer";
              card.onclick = () => pick(i);
            } else {
              card.onclick = (ev) => {
                const s = ev.target.closest("[data-tid]");
                if (s) pick(parseInt(s.dataset.tid));
              };
            }
            row.appendChild(card);
          });
          stage.appendChild(row);
          stage.appendChild(cover);
          wrap.appendChild(top);
          wrap.appendChild(stage);
          el.replaceChildren(wrap);
          return () => {
            if (idleTimer) clearTimeout(idleTimer);
            document.removeEventListener("visibilitychange", onVis);
          };
        }
        export default { render };
        """
        mode = traitlets.Unicode("duel").tag(sync=True)
        strip_bg = traitlets.Unicode("#888888").tag(sync=True)
        page_bg = traitlets.Unicode("#888888").tag(sync=True)
        page_dark = traitlets.Bool(False).tag(sync=True)
        ink = traitlets.Unicode("#808080").tag(sync=True)
        prompt_html = traitlets.Unicode("").tag(sync=True)
        chip = traitlets.Unicode("").tag(sync=True)
        progress = traitlets.Unicode("").tag(sync=True)
        gate = traitlets.Bool(False).tag(sync=True)
        gate_text = traitlets.Unicode("").tag(sync=True)
        cards = traitlets.List([]).tag(sync=True)
        choice = traitlets.Int(-1).tag(sync=True)
        clicks = traitlets.Int(0).tag(sync=True)
        pauses = traitlets.Int(0).tag(sync=True)
        t_render = traitlets.Float(-1.0).tag(sync=True)
        t_click = traitlets.Float(-1.0).tag(sync=True)

    _mono = "font-family:'IosevkaLigated Nerd Font Mono',monospace;font-size:18px"
    _chip = {"duel": "duel", "comprehension": "spot", "search": "find"}[_t["mode"]] + f" · {_pol} page"
    _progress = f"{_pos + 1} of {_len}"
    _gate_text = {
        "duel": (
            f"A run of {_len} duels on the {_pol} page: two pages render the same code — "
            "click the one you would rather read. Trust the first pull; a slow choice reads as a tie."
        ),
        "comprehension": (
            f"A run of {_len} probes on the {_pol} page: the bar names a function — "
            "click that name in the code as fast as you can find it."
        ),
        "search": (
            f"A run of {_len} find hunts on the {_pol} page: several matches are highlighted — "
            "click the current one, the strongest highlight, as fast as you can find it."
        ),
    }[_t["mode"]]
    if _t["mode"] == "duel":
        _cur = _rng.choice(_snip["ident_ids"]) if _snip["ident_ids"] else None
        _surface = _t.get("surface", "editor")
        _cards = [
            {
                "html": render_card(_t["theme_a"], _snip, _t["code_px"], find_current=_cur, surface=_surface),
                "ground": _t["theme_a"]["ground"],
            },
            {
                "html": render_card(_t["theme_b"], _snip, _t["code_px"], find_current=_cur, surface=_surface),
                "ground": _t["theme_b"]["ground"],
            },
        ]
        if _t["swap"]:
            _cards = _cards[::-1]
        _prompt = 'Which page would you rather read? <span style="opacity:.55">Click it.</span>'
    elif _t["mode"] == "comprehension":
        _target = _rng.choice(_snip["fn_ids"])
        _name = _snip["spans"][_target]["text"]
        _surface = _t.get("surface", "editor")
        _cards = [
            {
                "html": render_card(_t["theme_a"], _snip, _t["code_px"], task=True, prose=False),
                "ground": _t["theme_a"]["ground"],
            }
        ]
        _prompt = f'Click <code style="{_mono}">{_name}</code>'

    else:
        _cur = _rng.choice(_snip["ident_ids"])
        _surface = _t.get("surface", "editor")
        _cards = [
            {
                "html": render_card(_t["theme_a"], _snip, _t["code_px"], find_current=_cur, task=True, prose=False),
                "ground": _t["theme_a"]["ground"],
            }
        ]
        _prompt = 'Click the <b>current</b> match <span style="opacity:.55">— the strongest highlight.</span>'

    trial_widget = mo.ui.anywidget(
        _ThemeTrial(
            mode=_t["mode"],
            strip_bg=_strip,
            page_bg=_page_bg,
            page_dark=bool(_page_dark),
            ink=_ptxt,
            prompt_html=_prompt,
            chip=_chip,
            progress=_progress,
            gate=bool(_gate),
            gate_text=_gate_text,
            cards=_cards,
        )
    )
    trial_widget
    return (trial_widget,)


@app.cell(hide_code=True)
def _(LOG, datetime, get_responses, json, random, set_responses, snippet_for, timezone, trial_for, trial_widget):
    # Recording watches the widget's synced traits; the guard converts a stale surface's
    # click into a dropped click instead of a mis-record, and the trial is recomputed from
    # the log at event time — never read from a rendering's closure.
    _n = len(get_responses())

    def _record(v, n=_n):
        if n != len(get_responses()):
            return
        _t = trial_for(n, get_responses())
        _rng = random.Random(n * 48271 % (2**31))
        _snip = snippet_for(_t["snippet"], _t.get("snippet_width"), _t.get("target_kind"), _t.get("snippet_lines"))
        _entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n": n,
            "mode": _t["mode"],
            "kind": _t["kind"],
            "polarity": _t["polarity"],
            "snippet": _snip["id"],
            "snippet_hash": _snip.get("hash"),
            "snippet_kind": _snip.get("kind"),
            "snippet_fresh": bool(_snip.get("fresh", True)),
            "target_kind": _snip.get("target_kind"),
            "surface": _t.get("surface", "editor"),
            # Recomputed rather than shared: this is a different cell, and the widget
            # cell's underscore names are local to it.
            "page_bg": (
                {"day": "#d8d2cf", "night": "#14161c"}[_t["polarity"]]
                if _t["mode"] == "duel"
                else _t["theme_a"]["ground"]
            ),
            "code_px": _t["code_px"],
            "theta_a": _t["theta_a"],
            "theme_a": _t["theme_a"],
            # rt_ms runs from the LAST reveal (render, or resume after a pause); a trial
            # that was ever paused is flagged so its time is read as a near-tie downstream.
            "rt_ms": round(v["t_click"] - v["t_render"], 1),
            "t_render": round(v["t_render"], 1),
            "t_click": round(v["t_click"], 1),
            "paused": v.get("pauses", 0) > 0,
        }
        if _t["mode"] == "duel":
            _cur = _rng.choice(_snip["ident_ids"]) if _snip["ident_ids"] else None
            _shown = v["choice"]  # 0 = left card
            _choice = (1 - _shown) if _t["swap"] else _shown  # 0 = theme_a
            _entry.update(
                theta_b=_t["theta_b"],
                theme_b=_t["theme_b"],
                swap=_t["swap"],
                find_current=_cur,
                choice=_choice,
            )
        elif _t["mode"] == "comprehension":
            _target = _rng.choice(_snip["fn_ids"])
            _name = _snip["spans"][_target]["text"]
            _accept = [_i for _i, _s in enumerate(_snip["spans"]) if _s["role"] == "function" and _s["text"] == _name]
            _entry.update(target=_target, target_text=_name, clicked=v["choice"], correct=v["choice"] in _accept)
        else:
            _cur = _rng.choice(_snip["ident_ids"])
            _entry.update(
                target=_cur,
                clicked=v["choice"],
                correct=v["choice"] == _cur,
                salience=_t["theme_a"]["salience"],
                find_sal_theta=_t["theta_a"][8],
            )
        # Append-only, one record per line: concurrent sessions interleave, never overwrite.
        with LOG.open("a") as _f:
            _f.write(json.dumps(_entry) + "\n")
        set_responses([*get_responses(), _entry])

    _v = trial_widget.value
    if _v.get("clicks") == 1 and _v.get("choice", -1) >= 0 and _v.get("t_click", -1) > 0:
        _record(_v)
    return


@app.cell(hide_code=True)
def _(
    AXES,
    DE_MIN,
    POOL,
    THRESH_DETAIL,
    VISION_N,
    best_set,
    candidates,
    fitted,
    get_responses,
    mo,
    mu_at,
    np,
    pd,
    render_card,
    snippet_for,
    spread_out,
):
    # A stable page for the champion preview: the same generated code every time, so
    # what changes between renders is the theme and nothing else.
    _preview_snip = snippet_for(0)
    _log = get_responses()
    if not _log:
        _out = mo.md("*No responses yet — the analysis fills in as you answer.*")
    else:
        _frame = pd.DataFrame(_log)
        _n_duel = int((_frame["mode"] == "duel").sum())
        _fit = fitted(_log) if _n_duel >= 5 else None
        _blocks = [
            mo.hstack(
                [
                    mo.stat(f"{len(_frame):,}", label="responses", bordered=True),
                    mo.stat(str(_n_duel), label="duels", bordered=True),
                    mo.stat(
                        str(int((_frame["mode"] == "comprehension").sum())),
                        label="comprehension probes",
                        bordered=True,
                    ),
                    mo.stat(str(int((_frame["mode"] == "search").sum())), label="find hunts", bordered=True),
                    mo.stat(
                        f"{DE_MIN['day']:.1f} / {DE_MIN['night']:.1f}",
                        label=f"ΔE floors (day/night), {VISION_N} vision trials",
                        bordered=True,
                    ),
                ],
                justify="start",
                gap=1,
            )
        ]
        if _fit is not None:
            for _pol in ("day", "night"):
                # The verdict is computed over BRED candidates, not the frozen pool: the
                # answer should be the best theme the search can reach, not the best of 512
                # points fixed before the first click.
                _bred = candidates(_fit, _pol, np.random.default_rng(4242), n_trial=0)[0]
                _thetas = [_b[0] for _b in _bred]
                _themes = [_b[1] for _b in _bred]
                _mu = mu_at(_fit, _thetas, _pol)
                _ci = int(np.argmax(_mu))
                _champ_theta, _champ = _thetas[_ci], _themes[_ci]
                _beats = float(np.mean(1.0 / (1.0 + np.exp(-(_mu[_ci] - _mu)))))
                # Is there ONE best theme or a plateau of equals? P(best) over the joint
                # posterior answers it as a distribution rather than a ranking: mass
                # concentrated on one page means a winner, mass spread means any member of
                # the credible set is a defensible choice -- and the ones shown are picked
                # for spread, since a plateau is only useful if its members look different.
                _bs = best_set(_fit, _pol, _thetas, seed=17)
                _cred = _bs["credible"]
                _reps = spread_out(_thetas, _cred, 4, _fit.get("ls"))
                _lead_pct = 100 * _bs["lead"]
                if _bs["verdict"] == "single":
                    _verdict = (
                        f"**one theme leads** — it holds {_lead_pct:.0f}% of the probability of "
                        f"being the best theme, so this is the one to apply"
                    )
                elif _bs["verdict"] == "plateau":
                    _verdict = (
                        f"**a plateau of {len(_cred)} distinct themes** — the leader holds "
                        f"{_lead_pct:.0f}%, and these together hold half the probability of being "
                        f"best. They are equally good by measurement, not merely acceptable: every "
                        f"one has already cleared the legibility floors, so pick by eye"
                    )
                else:
                    _verdict = (
                        f"**not yet decided** — the strongest theme holds only {_lead_pct:.0f}% of "
                        f"the probability of being best, which is what a thin log looks like rather "
                        f"than a real plateau. {len(_cred)} themes share half the mass; more duels "
                        f"on this polarity will separate them"
                    )
                _blocks.append(
                    mo.md(
                        f"### The {_pol} verdict\n\n{_verdict}. Shown below: the leader, then the "
                        f"most *different* members of the set holding half the probability mass — "
                        f"near-identical themes are grouped first, so what you see are choices "
                        f"rather than variations of one."
                    )
                )
                _blocks.append(
                    mo.hstack(
                        [
                            mo.vstack(
                                [
                                    mo.md(f"**{100 * _bs['p_best'][_i]:.0f}%** · utility {_mu[_i]:.2f}"),
                                    mo.Html(
                                        f'<div style="background:{_themes[_i]["ground"]};border-radius:8px;'
                                        f'padding:12px;width:330px">'
                                        + render_card(_themes[_i], _preview_snip, 13, prose=False)
                                        + "</div>"
                                    ),
                                ],
                                gap=0.3,
                            )
                            for _i in _reps
                        ],
                        justify="start",
                        gap=1,
                        wrap=True,
                    )
                )
                _sweep = []
                for _ax in range(9):
                    _lo_t = np.array(_champ_theta, dtype=float)
                    _hi_t = _lo_t.copy()
                    _lo_t[_ax], _hi_t[_ax] = 0.15, 0.85
                    _mm = mu_at(_fit, [_lo_t, _hi_t], _pol)
                    _sweep.append(
                        {
                            "axis": AXES[_ax],
                            "low (0.15)": round(float(_mm[0] - _mu[_ci]), 2),
                            "high (0.85)": round(float(_mm[1] - _mu[_ci]), 2),
                        }
                    )
                _blocks += [
                    mo.md(
                        f"**Current best {_pol} page** — beats a random feasible theme with "
                        f"p ≈ {_beats:.2f}; utility marginals below are the posterior-mean change "
                        f"from the champion when one axis is pushed to its walls (negative = the "
                        f"champion's setting is better):"
                    ),
                    mo.Html(
                        f'<div style="background:{_champ["ground"]};border-radius:10px;padding:20px;max-width:620px">'
                        + render_card(
                            _champ,
                            _preview_snip,
                            16,
                            find_current=(_preview_snip["ident_ids"] or [None])[0],
                        )
                        + "</div>"
                    ),
                    mo.ui.table(pd.DataFrame(_sweep), selection=None),
                    mo.md(
                        "Champion override snippet (paste into settings.jsonc under the matching "
                        "theme block; find keys carry their alpha):"
                    ),
                    mo.md(
                        "```jsonc\n"
                        + "{\n"
                        + f"  // {_pol} · ground {_champ['ground']}\n"
                        + f'  "editor.background": "{_champ["ground"]}",\n'
                        + f'  "editor.findMatchBackground": "{_champ["find_fill"]}d9",\n'
                        + f'  "editor.findMatchHighlightBackground": "{_champ["find_fill"]}73",\n'
                        + '  "textMateRules": {\n'
                        + f'    "keyword": "{_champ["keyword"]}", "function": "{_champ["function"]}",\n'
                        + f'    "string|number": "{_champ["string"]}", "comment (italic)": "{_champ["comment"]}",\n'
                        + f'    "variables/ink": "{_champ["ink"]}", "punctuation": "{_champ["punct"]}"\n'
                        + "  }\n"
                        + "}\n```"
                    ),
                ]
        _tasks = _frame[_frame["mode"] == "comprehension"]
        if len(_tasks) >= 6:
            # Correct AND never-paused: a paused trial's clock measures the break, not the
            # eyes. Rows predating the pause affordance lack the field: they count unpaused.
            _np1 = ~_tasks.get("paused", pd.Series(False, index=_tasks.index)).fillna(False).astype(bool)
            _ok = _tasks[(_tasks["correct"] == True) & _np1]  # noqa: E712
            _blocks.append(
                mo.md(
                    f"**Comprehension**: {len(_tasks)} probes, {100 * _tasks['correct'].mean():.0f}% correct; "
                    f"median time-to-click {_ok['rt_ms'].median():.0f} ms "
                    f"(fastest quartile {_ok['rt_ms'].quantile(0.25):.0f} ms — the gap is what theming can win)."
                )
            )
        _hunts = _frame[_frame["mode"] == "search"]
        if len(_hunts) >= 6:
            _np2 = ~_hunts.get("paused", pd.Series(False, index=_hunts.index)).fillna(False).astype(bool)
            _hok = _hunts[(_hunts["correct"] == True) & _np2]  # noqa: E712
            if len(_hok) >= 4:
                _z = np.polyfit(_hok["salience"], np.log(_hok["rt_ms"]), 1)
                _blocks.append(
                    mo.md(
                        f"**Find hunts**: {len(_hunts)} trials, {100 * _hunts['correct'].mean():.0f}% correct; "
                        f"log time-to-find slope over salience {_z[0]:+.3f} per ΔE "
                        f"(negative = louder is genuinely faster; near zero = salience past this point buys nothing "
                        f"and beauty should take the wheel)."
                    )
                )
        if THRESH_DETAIL.get("day"):
            _blocks.append(
                mo.md(
                    "Constraint provenance — your fitted 75%-correct thresholds in CAM16-UCS ΔE (day / night): "
                    + ", ".join(
                        f"{_ax} {THRESH_DETAIL['day'][_ax]:.1f} / {THRESH_DETAIL['night'][_ax]:.1f}"
                        for _ax in THRESH_DETAIL["day"]
                    )
                    + " — the pairwise floor is 2× the minimum "
                    + f"({2 * DE_MIN['day']:.1f} day, {2 * DE_MIN['night']:.1f} night)."
                )
            )
        _out = mo.vstack(_blocks, gap=0.8)
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reading the numbers, and what happens to them

    The utility is latent and relative — only differences mean anything, so the readout is
    given as *probabilities*: how often the current champion would beat a random feasible
    theme in a duel you haven't run. Marginals near zero on an axis mean your taste is flat
    there (let the constraints or the prior decide); a large one-sided marginal means the
    axis matters and the champion sits where you put it. Early sittings will look noisy —
    a preferential GP needs roughly forty duels before the Thompson arm stops wandering,
    and the 7% uniform probes *should* occasionally look strange: they are the insurance
    premium against a model that only ever asks questions it already believes.

    **One theme or several?** The verdict above is a distribution, not a ranking: sampling
    the joint posterior gives each candidate its probability of being *the* best, and the
    answer is read from how that mass sits. Two details make that honest. Near-identical
    candidates are **grouped** before counting, because eight hundred candidates contain
    many pages that differ by less than you could see and each sibling would otherwise steal
    argmax mass from the others. And the reading is of *cumulative* mass, not a fixed
    cutoff: one group holding over half of it is a winner; a handful sharing it is a real
    plateau, and the members shown are then chosen to be as *different* from each other as
    the set allows; and when even the strongest group is thin, the report says **not yet
    decided** rather than dressing a thin log up as a plateau. Nothing on that shelf is a
    compromise either way — every candidate has already cleared the legibility floors, so a
    plateau means genuinely equal, not merely acceptable.

    Two properties of the machinery were measured rather than assumed, and the tests live in
    `_model_tests.py` beside this file. The position of a card matters to you — over the
    first 79 duels the right-hand card won 61% of the time — so a side-advantage term is
    fitted and subtracted out instead of being left to land on the themes as noise. And the
    nine axes are not equally alive: their length-scales are learned, which shrinks the
    effective dimension the search has to cover, with the estimate held near isotropic until
    enough duels exist to identify relevance at all.

    Reaction time is doing quiet work throughout: a fast duel click steepens that duel's
    likelihood (drift-diffusion reading — big utility gaps decide quickly), a slow one
    flattens it toward a tie, so deliberating over a near-tie neither punishes nor rewards
    either side. That channel is only as clean as its baseline. The first trial of a sitting
    and the first of every run start hidden behind a **begin** button, because those are the
    moments you read an instruction; inside a run the click that produced a trial is its
    anchor, so the clock starts at render and no button stands between you and the next
    page. A **pause** button, the tab losing visibility, or 25 s without a click re-covers the
    stimulus — an exposed one lets a decision form off the clock — and resuming re-baselines;
    a trial paused after its first reveal is flagged in the log: its choice still counts, at
    the neutral slope, and it is excluded from the comprehension and find-hunt timing
    statistics.

    Comprehension probes and find hunts measure time directly; they are the
    glyph-scale ground truth that the 2× threshold safety margin (from the 104-px vision
    fit) is standing in for until this instrument accumulates its own.

    Hard floors are never traded: every page shown clears WCAG 4.5:1 and APCA 60 on body
    tokens, and every pair of colored roles clears twice your measured CAM16-UCS threshold
    for its ground. Literals are one family by measurement, not taste: Horizon's own day
    string and number oranges sit within your threshold of each other. Plain variable reads
    render as ink by standing preference (figure-ground: definitions, literals, and control
    words carry the color).

    The winner's destination: the find-highlight pair lands in `editor.findMatchBackground`
    and `editor.findMatchHighlightBackground` (settings.jsonc already overrides those keys),
    the token colors in `editor.tokenColorCustomizations` per theme, the ground in the
    workbench block — via the champion snippet above, once its posterior stops moving
    between sittings. Trials accumulate in `aesthetics-responses.jsonl`, committed like any
    measurement; the trial generator is deterministic given that log, so any session
    resumes exactly where the last one stopped. Findings that outlive a sitting get written
    into this closing prose, next to the live numbers.
    """)
    return


if __name__ == "__main__":
    app.run()
