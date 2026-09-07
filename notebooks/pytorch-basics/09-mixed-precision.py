# /// script
# [tool.marimo.runtime]
# on_cell_change = "autorun"
# ///

# The repository default is lazy, which marks a cell stale rather than running it when
# something upstream changes -- correct for a notebook holding a model on the GPU, and
# fatal for a slider, whose whole point is that the picture moves while you drag. Script
# metadata is merged over the project config at the highest precedence, so a notebook
# opts in on its own. `auto_instantiate` cannot be set here (marimo strips it from script
# metadata), so opening this file still runs nothing.
#
# The two training cells are the only things here expensive enough to want a guard, and
# each has one: mo.stop on a run button, so autorun re-runs them only after a click. The
# gates lift for a script run, so the headless check exercises those code paths too.

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
    *PyTorch basics, 9 of 9 — before this: [Save & Load Model](08-save-load-run.py)*

    # Mixed Precision

    Every tensor in the eight notebooks before this one was `float32`: four bytes per
    number, about seven decimal digits, a range up to $3.4 \times 10^{38}$. The tensor cores
    on the GPU in this machine multiply 16-bit matrices several times faster than 32-bit
    ones — the exact factor is measured below — and a 16-bit activation takes half the
    memory. So why not train in 16 bits?

    Because two of the three things a training loop does with a number are safe in 16
    bits and the third is not. Multiplying and adding activations survives the loss of
    precision. Adding a tiny update to a weight does not: the update rounds away and the
    weight never moves. *Mixed* precision is the arrangement that puts each operation in
    the format it can afford — compute in 16 bits, keep the weights in 32. This notebook
    builds that arrangement up from the bits, then applies it to the training loop of
    [notebook 07](07-optimization-loop.py), which changes by four lines.

    ## Prerequisite Code

    The model, data and loaders are notebook 07's, with one addition: the hidden width is a
    parameter of `NeuralNetwork`, because the last section needs a wider model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Today's target** — watch one forward pass change dtype under `autocast`, see a
    > weight update disappear in `bfloat16`, and understand why the FP16 scaler skips
    > a nonfinite-gradient step. The five-mode performance experiment is optional.
    >
    > **Depth line** — deeper than the tutorials. `torch.autocast` and `torch.amp.GradScaler`
    > are what every training framework's mixed-precision switch is built on, and every
    > distributed recipe from here on assumes them. The depth that matters is being able to
    > say what `autocast` does to one matrix multiply, why `bfloat16` needs no loss scaler
    > and `float16` does, and where the master weights live.
    >
    > **Stop-line** — explain why weights and their gradients stay `float32`, why matmuls
    > can run in 16 bits, and what the scaler protects. Capture one assertion you could
    > test in an Accelerate integration and any open question — then close it. Benchmark
    > tuning is not a prerequisite for starting that PR.
    >
    > **Capture** — `scripts/q "your question"` appends it to Friday's file for Marc.
    """)
    return


@app.cell
def _():
    import time

    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from torchvision.transforms import v2

    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    )

    class NeuralNetwork(nn.Module):
        def __init__(self, width=512):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28 * 28, width),
                nn.ReLU(),
                nn.Linear(width, width),
                nn.ReLU(),
                nn.Linear(width, 10),
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    device
    return (
        DataLoader,
        NeuralNetwork,
        device,
        nn,
        test_data,
        time,
        torch,
        training_data,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Three ways to spend 16 bits

    A floating-point number is three fields: a **sign** bit, an **exponent** that sets the
    magnitude (how far the binary point slides), and a **mantissa** that sets the digits
    (how finely the numbers between two powers of two are spaced). More exponent bits buy
    *range*; more mantissa bits buy *precision*; a format has to choose.

    `float32` spends 8 bits on exponent and 23 on mantissa. The two 16-bit formats split the
    savings differently. `float16` (IEEE half precision) keeps most of the digits — 10
    mantissa bits — and pays with a 5-bit exponent, so its range collapses to about
    $6 \times 10^{-5}$ … $65\,504$. `bfloat16` (Google's *brain float*) is simply `float32`
    with the last 16 bits cut off: the same 8-bit exponent, so the same range, and only 7
    mantissa bits, so about two decimal digits. The fourth row, **TF32**, is not a storage
    format at all — it is what an NVIDIA tensor core reads out of a `float32` input when it
    is allowed to: the full exponent and the top 10 mantissa bits, the rest dropped.
    `torch.finfo` reports each format's limits; the picture under it is the layout.
    """)
    return


@app.cell
def _(torch):
    formats = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    {name: torch.finfo(dtype) for name, dtype in formats.items()}
    return (formats,)


@app.cell(hide_code=True)
def _():
    import altair as alt
    import pandas as pd
    from _palette import FURNITURE, tint
    from _viz import BASE, INK_DARK, INK_LIGHT, OKABE_ITO

    # One hue per format, kept constant through every exhibit here: TF32 is float32's
    # sibling (same exponent, shorter mantissa), so it takes the lighter blue.
    FORMAT_COLORS = {
        "float32": BASE,
        "TF32": OKABE_ITO["sky"],
        "bfloat16": OKABE_ITO["green"],
        "float16": OKABE_ITO["vermillion"],
    }
    return (
        FORMAT_COLORS,
        FURNITURE,
        INK_DARK,
        INK_LIGHT,
        OKABE_ITO,
        alt,
        pd,
        tint,
    )


@app.cell(hide_code=True)
def _(FURNITURE, mo):
    def furnish(chart):
        """Axis ink, tick labels and gridlines from the measured graph furniture, per theme.

        The canvas stays transparent so the chart sits on the page's own color; only the
        marks and the furniture are painted, and the furniture follows the polarity the
        notebook is being read in.
        """
        polarity = "night" if mo.app_meta().theme == "dark" else "day"
        f = FURNITURE[polarity]
        return (
            chart.configure(background="transparent")
            .configure_view(strokeWidth=0)
            .configure_axis(
                labelColor=f["label"],
                titleColor=f["ink"],
                gridColor=f["grid"],
                domainColor=f["axis"],
                tickColor=f["axis"],
                labelFontSize=12,
                titleFontSize=12,
            )
            .configure_legend(labelColor=f["label"], titleColor=f["ink"])
            .configure_title(color=f["ink"])
        )

    return (furnish,)


@app.cell(hide_code=True)
def _(FORMAT_COLORS, INK_DARK, INK_LIGHT, alt, furnish, mo, pd, tint):
    _layouts = [
        # name, exponent bits, mantissa bits, bits the tensor core drops (TF32 only)
        ("float32", 8, 23, 0),
        ("TF32", 8, 10, 13),
        ("bfloat16", 8, 7, 0),
        ("float16", 5, 10, 0),
    ]
    _fields, _colors = [], {}
    for _name, _exp, _man, _dropped in _layouts:
        _start = 0
        for _role, _width in (("sign", 1), ("exponent", _exp), ("mantissa", _man), ("dropped", _dropped)):
            if _width == 0:
                continue
            _key = f"{_name} {_role}"
            _hue = FORMAT_COLORS[_name]
            # Role is carried by lightness of the format's own hue: sign is ink, the
            # exponent the hue itself, the mantissa a tint of it, dropped bits nearly paper.
            _colors[_key] = {
                "sign": INK_DARK,
                "exponent": _hue,
                "mantissa": tint(_hue, 0.55),
                "dropped": tint(_hue, 0.88),
            }[_role]
            _fields.append(
                {
                    "format": _name,
                    "role": _role,
                    "key": _key,
                    "start": _start,
                    "end": _start + _width,
                    "mid": _start + _width / 2,
                    "label": {"sign": "±", "dropped": f"{_width} bits the tensor core drops"}.get(
                        _role, f"{_role} · {_width}"
                    ),
                    "ink": INK_LIGHT if _role in ("sign", "exponent") else INK_DARK,
                }
            )
            _start += _width
    _frame = pd.DataFrame(_fields)
    _y = alt.Y("format:N", sort=[name for name, *_ in _layouts], title=None)
    _picture = (
        alt.Chart(_frame)
        .mark_rect(stroke="transparent", strokeWidth=2)
        .encode(
            x=alt.X(
                "start:Q",
                title="bit, most significant first",
                axis=alt.Axis(values=[0, 1, 9, 16, 32], grid=False),
            ),
            x2="end:Q",
            y=_y,
            color=alt.Color("key:N", scale=alt.Scale(domain=list(_colors), range=list(_colors.values())), legend=None),
            tooltip=[
                "format:N",
                "role:N",
                alt.Tooltip("start:Q", title="first bit"),
                alt.Tooltip("end:Q", title="up to"),
            ],
        )
        + alt.Chart(_frame)
        .mark_text(fontSize=12, fontWeight=500)
        .encode(x="mid:Q", y=_y, text="label:N", color=alt.Color("ink:N", scale=None))
    ).properties(width=640, height=4 * 34)
    mo.vstack(
        [
            furnish(_picture),
            mo.md(
                "<small>The four layouts, most significant bit at the left. `bfloat16` is the top half of "
                "`float32`; `float16` trades three exponent bits for three more of mantissa; TF32 reads "
                "19 of a `float32`'s 32 bits.</small>"
            ),
        ],
        align="center",
        gap=0.2,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Counting

    The mantissa decides how far a format can count in whole numbers before it skips one:
    an integer needs as many significant bits as its length, and the mantissa has one more
    bit than it stores (a leading 1 that is implied, never written). Predict the three
    numbers before running the cell: `float16` has 10 stored bits, `bfloat16` has 7,
    `float32` has 23.
    """)
    return


@app.cell
def _(formats, torch):
    def first_uncountable(dtype, upto=2**25):
        integers = torch.arange(upto, dtype=torch.float64)
        skipped = integers[integers.to(dtype).to(torch.float64) != integers]
        return int(skipped[0])

    {name: first_uncountable(dtype) for name, dtype in formats.items()}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $2^{11} + 1 = 2049$, $2^{8} + 1 = 257$, $2^{24} + 1 = 16\,777\,217$. A `bfloat16` cannot
    tell 257 from 256; in its neighbourhood the representable numbers are two apart. That
    is not a corner case — it is the format's resolution at every magnitude: about one part
    in 256, everywhere from $10^{-38}$ to $10^{38}$.

    ### The ruler

    The picture below is the spacing between one representable number and the next — the
    *unit in the last place*, or ulp — against the number's magnitude, for each format. Each
    line is a staircase because the spacing doubles at every power of two, and the height of
    a line is the format's precision: the `bfloat16` staircase sits eight times higher than
    `float16`'s, `float16`'s eight thousand times higher than `float32`'s.

    Width is range. `float16` ends abruptly at 65 504 — one step further is infinity — and
    below $6 \times 10^{-5}$ it enters the *subnormal* numbers, where the spacing stops
    shrinking down to about $6 \times 10^{-8}$; smaller values can round to zero. `bfloat16` and `float32` continue
    far past both edges of this picture.

    Slide the probe. The cell casts one number into each format and reports what came back;
    the marks on the chart follow it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    probe_exponent = mo.ui.slider(-8, 6, step=0.25, value=-5, label="probe: 10 to the power", show_value=True)
    probe_exponent
    return (probe_exponent,)


@app.cell
def _(formats, probe_exponent, torch):
    probe = torch.tensor(10.0**probe_exponent.value, dtype=torch.float64)
    {name: probe.to(dtype).item() for name, dtype in formats.items()}
    return (probe,)


@app.cell(hide_code=True)
def _(
    FORMAT_COLORS,
    alt,
    formats,
    furnish,
    mo,
    pd,
    probe,
    probe_exponent,
    torch,
):
    def _spacing(values, dtype):
        """Each value once stored in dtype, and its distance to the next number dtype has."""
        stored = values.to(dtype)
        up = torch.nextafter(stored, torch.full_like(stored, float("inf")))
        return stored.to(torch.float64), up.to(torch.float64) - stored.to(torch.float64)

    _grid = torch.logspace(-8.5, 6.5, 400, dtype=torch.float64)
    _rows = []
    for _name, _dtype in formats.items():
        _stored, _ulp = _spacing(_grid, _dtype)
        for _x, _s, _u in zip(_grid.tolist(), _stored.tolist(), _ulp.tolist(), strict=True):
            if _s != 0 and _s != float("inf"):
                _rows.append({"format": _name, "magnitude": _x, "spacing": _u})
    _frame = pd.DataFrame(_rows)

    def _verdict(name, stored):
        if stored == float("inf"):
            return "overflow → inf"
        if stored == 0:
            return "underflow → 0"
        if abs(stored) < torch.finfo(formats[name]).smallest_normal:
            return "subnormal"
        return "exact" if stored == probe.item() else "rounded"

    _probe_rows, _marks = [], []
    for _name, _dtype in formats.items():
        _stored, _ulp = (t.item() for t in _spacing(probe.reshape(1), _dtype))
        _finite = 0 < abs(_stored) < float("inf")
        _probe_rows.append(
            {
                "format": _name,
                "stored as": _stored,
                "relative error": f"{abs(_stored - probe.item()) / probe.item():.2e}" if _finite else "—",
                "spacing to next": f"{_ulp:.3g}" if _finite else "—",
                "verdict": _verdict(_name, _stored),
            }
        )
        if _finite:
            _marks.append({"format": _name, "magnitude": probe.item(), "spacing": _ulp})

    _scale = alt.Scale(domain=list(formats), range=[FORMAT_COLORS[name] for name in formats])
    _f16 = torch.finfo(torch.float16)
    _edges = pd.DataFrame(
        [
            {"x": _f16.max, "label": "float16 max 65 504"},
            {"x": _f16.smallest_normal, "label": "float16 subnormals begin"},
        ]
    )
    _chart = (
        alt.Chart(_frame)
        .mark_line(interpolate="step-after", strokeWidth=2)
        .encode(
            x=alt.X("magnitude:Q", scale=alt.Scale(type="log"), title="magnitude of the number"),
            y=alt.Y("spacing:Q", scale=alt.Scale(type="log"), title="spacing to the next representable number"),
            color=alt.Color("format:N", scale=_scale, legend=alt.Legend(title=None, orient="top-left")),
            tooltip=["format:N", alt.Tooltip("magnitude:Q", format=".3g"), alt.Tooltip("spacing:Q", format=".3g")],
        )
        + alt.Chart(_edges).mark_rule(color=FORMAT_COLORS["float16"], strokeDash=[4, 4], opacity=0.7).encode(x="x:Q")
        + alt.Chart(_edges)
        .mark_text(align="right", dx=-4, dy=-4, fontSize=11, color=FORMAT_COLORS["float16"])
        .encode(x="x:Q", y=alt.value(12), text="label:N")
        + alt.Chart(pd.DataFrame(_marks))
        .mark_point(size=110, filled=True, stroke="white", strokeWidth=1.5)
        .encode(x="magnitude:Q", y="spacing:Q", color=alt.Color("format:N", scale=_scale, legend=None))
    ).properties(width=620, height=300)

    mo.vstack(
        [
            furnish(_chart),
            mo.md(
                f"<small>Spacing between adjacent representable numbers, by magnitude; the filled marks are the "
                f"probe, $10^{{{probe_exponent.value:g}}}$. `float16` is drawn only where it has a finite, nonzero "
                f"value.</small>"
            ),
            mo.ui.table(_probe_rows, selection=None),
        ],
        align="center",
        gap=0.6,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Three ways to lose a number

    Each format fails in its own place. `float16` fails at the edges of the ruler: a value
    past 65 504 becomes infinity (*overflow*), and a gradient of $10^{-8}$ becomes exactly
    zero (*underflow*) — both are ordinary sizes for activations and gradients in a deep
    network. `bfloat16` never runs off either edge, but it fails in the middle: adding a
    small number to a larger one rounds the sum back to the larger one, because the small
    number is less than half a step on the ruler at that magnitude. Predict what a thousand
    additions of 0.001 come to in each format before running the cell.
    """)
    return


@app.cell
def _(formats, torch):
    def add_thousandths(dtype, times=1000):
        total = torch.tensor(0.0, dtype=dtype)
        for _ in range(times):
            total = total + torch.tensor(0.001, dtype=dtype)
        return total.item()

    {
        "70 000 in float16": torch.tensor(70_000.0).to(torch.float16).item(),
        "1e-8 in float16": torch.tensor(1e-8).to(torch.float16).item(),
        "1e-8 in bfloat16": torch.tensor(1e-8).to(torch.bfloat16).item(),
        "0.001 added 1000 times": {name: add_thousandths(dtype) for name, dtype in formats.items()},
    }
    return


@app.cell(hide_code=True)
def _(FORMAT_COLORS, alt, formats, furnish, mo, pd, torch):
    _rows = []
    for _name, _dtype in formats.items():
        _total = torch.tensor(0.0, dtype=_dtype)
        for _step in range(1, 1001):
            _total = _total + torch.tensor(0.001, dtype=_dtype)
            if _step % 5 == 0:
                _rows.append({"format": _name, "additions": _step, "total": _total.item()})
    _frame = pd.DataFrame(_rows)
    _ideal = pd.DataFrame({"additions": [0, 1000], "total": [0, 1.0]})
    _chart = (
        alt.Chart(_ideal).mark_line(color="gray", strokeDash=[3, 3], opacity=0.6).encode(x="additions:Q", y="total:Q")
        + alt.Chart(_frame)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("additions:Q", title="additions of 0.001"),
            y=alt.Y("total:Q", title="running total"),
            color=alt.Color(
                "format:N",
                scale=alt.Scale(domain=list(formats), range=[FORMAT_COLORS[name] for name in formats]),
                legend=alt.Legend(title=None, orient="top-left"),
            ),
            tooltip=["format:N", "additions:Q", alt.Tooltip("total:Q", format=".4f")],
        )
    ).properties(width=620, height=240)
    mo.vstack(
        [
            furnish(_chart),
            mo.md(
                "<small>The running total of a thousand additions of 0.001, per format; the dashed line is the exact "
                "answer. `bfloat16` stalls at 0.5, where its spacing is 0.0039 and 0.001 is less than half a step; "
                "`float16` bends past 0.5 for the same reason, more gently.</small>"
            ),
        ],
        align="center",
        gap=0.2,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A weight update is exactly the third failure: $w \leftarrow w - \eta\, g$ adds a small
    number to a larger one. With notebook 07's learning rate of 0.001 and a gradient of
    0.01, the update is $10^{-5}$ on a weight of size 0.03 — where `bfloat16`'s spacing is
    $2^{-13} \approx 1.2 \times 10^{-4}$. The update is a twelfth of a step. It rounds away,
    and the weight is exactly what it was. That single fact is why the weights in a
    mixed-precision loop are kept in `float32`; the section on master weights below counts
    how many updates of the real model it would swallow.

    ## Where the speed comes from

    Since the Volta generation, an NVIDIA GPU carries **tensor cores**: units that
    multiply small matrix tiles in one instruction, and only in the reduced formats. The
    ordinary `float32` path runs on the general-purpose cores and does one multiply-add per
    lane per cycle. The cell below times a square matrix multiply in four modes and reports
    the achieved rate. The two `float32` rows differ only in one global switch:
    `torch.set_float32_matmul_precision("high")` lets the tensor cores read `float32`
    inputs as TF32 — 10 mantissa bits instead of 23 — and return a `float32` result.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    matmul_size = mo.ui.slider(steps=[1024, 2048, 4096, 8192], value=4096, label="matrix side", show_value=True)
    matmul_size
    return (matmul_size,)


@app.cell
def _(device, matmul_size, time, torch):
    def timed(operation, repeats=20):
        for _ in range(3):
            operation()  # warm-up: the first call of each kernel pays for loading it
        torch.accelerator.synchronize()
        started = time.perf_counter()
        for _ in range(repeats):
            operation()
        torch.accelerator.synchronize()
        return (time.perf_counter() - started) / repeats

    n = matmul_size.value
    a = torch.randn(n, n, device=device, generator=torch.Generator(device).manual_seed(0))
    b = torch.randn(n, n, device=device, generator=torch.Generator(device).manual_seed(1))
    reference = a.to(torch.float64) @ b.to(torch.float64)

    matmul_modes = {
        "float32": ("highest", torch.float32),
        "TF32": ("high", torch.float32),
        "bfloat16": ("highest", torch.bfloat16),
        "float16": ("highest", torch.float16),
    }
    matmul_results = []
    for mode, (precision, dtype) in matmul_modes.items():
        torch.set_float32_matmul_precision(precision)
        x, y = a.to(dtype), b.to(dtype)
        seconds = timed(lambda x=x, y=y: x @ y)
        error = ((x @ y).to(torch.float64) - reference).norm() / reference.norm()
        matmul_results.append(
            {"mode": mode, "ms": seconds * 1e3, "TFLOP/s": 2 * n**3 / seconds / 1e12, "relative error": error.item()}
        )
    torch.set_float32_matmul_precision("highest")  # the switch is global; leave it as found
    return (matmul_results,)


@app.cell(hide_code=True)
def _(FORMAT_COLORS, alt, furnish, matmul_results, matmul_size, mo, pd):
    _frame = pd.DataFrame(matmul_results)
    _scale = alt.Scale(domain=list(FORMAT_COLORS), range=list(FORMAT_COLORS.values()))
    _bars = (
        alt.Chart(_frame)
        .mark_bar()
        .encode(
            y=alt.Y("mode:N", sort=list(FORMAT_COLORS), title=None),
            x=alt.X("TFLOP/s:Q", title="achieved TFLOP/s"),
            color=alt.Color("mode:N", scale=_scale, legend=None),
            tooltip=["mode:N", alt.Tooltip("ms:Q", format=".2f"), alt.Tooltip("TFLOP/s:Q", format=".0f")],
        )
        + alt.Chart(_frame)
        .mark_text(align="left", dx=4, fontSize=12)
        .encode(y=alt.Y("mode:N", sort=list(FORMAT_COLORS)), x="TFLOP/s:Q", text=alt.Text("TFLOP/s:Q", format=".0f"))
    ).properties(width=520, height=4 * 30)
    _table = [
        {
            "mode": r["mode"],
            "ms per matmul": f"{r['ms']:.2f}",
            "TFLOP/s": f"{r['TFLOP/s']:.0f}",
            "vs float32": f"{r['TFLOP/s'] / _frame['TFLOP/s'][0]:.1f}×",
            "relative error vs float64": f"{r['relative error']:.1e}",
        }
        for r in matmul_results
    ]
    mo.vstack(
        [
            furnish(_bars),
            mo.md(
                f"<small>One {matmul_size.value} × {matmul_size.value} matrix multiply, mean of 20 after warm-up, "
                f"on this machine's GPU. Error is the Frobenius norm of the difference from a float64 result, "
                f"relative to that result.</small>"
            ),
            mo.ui.table(_table, selection=None),
        ],
        align="center",
        gap=0.6,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read the error column against the speed column. TF32 buys its speed with a thousandfold
    loss of matmul accuracy ($10^{-4}$ against $10^{-7}$), which is why PyTorch does not turn
    it on for you: `torch.get_float32_matmul_precision()` is `"highest"` by default. The
    default is not uniform, though — `torch.backends.cudnn.allow_tf32` is `True`, so a
    *convolution* on `float32` inputs already runs in TF32 while a matrix multiply does not.
    A `float32` model is, on an NVIDIA card, already slightly mixed.

    The two 16-bit rows are the reason this notebook exists. `bfloat16`'s error is an order
    of magnitude worse than `float16`'s (seven mantissa bits against ten) at the same speed,
    and both are far worse than TF32. Training tolerates it: a matmul error of a few parts
    in a thousand is noise beneath the gradient noise of a minibatch. What training does
    *not* tolerate is the weight update rounding away, and that is the operation the next
    section keeps out of 16 bits.

    ## Mixed: compute in 16, keep in 32

    `torch.autocast` applies device-specific dtype rules to eligible operations inside
    its context. For the CUDA operations used here, the important cases are:

    - **Down to 16 bits** — the operations that are expensive and tolerant: `matmul`,
      `linear`, `conv*`, `bmm`, and their relatives. Their `float32` inputs are cast on the
      way in; their outputs come out 16-bit.
    - **Up to `float32`** — the operations that are cheap and fragile: `softmax`,
      `log_softmax`, `cross_entropy`, `layer_norm`, `exp`, `log`, `pow`, `sum`, and the
      other reductions and losses, where a 16-bit intermediate would overflow or lose the
      small terms. Their 16-bit inputs are cast *up* on the way in.
    - **Selected multi-input operations** promote to the widest input dtype. Unlisted
      operations follow their own dtype rules; they are not automatically promoted.

    In-place operations, calls with `out=`, and an explicit `dtype=` bypass autocasting.
    The [operation reference](https://docs.pytorch.org/docs/stable/amp.html#autocast-op-reference)
    identifies the eligible operations for each device.

    Nothing about the model changes. Its parameters stay `float32` — autocast casts a copy
    of each weight the first time an operation asks for it and caches that copy until the
    context exits, so each weight is cast once per forward pass, not once per use. The
    backward pass is not inside the context and does not need to be: autograd recorded which
    dtype each operation ran in and differentiates in that dtype, and each gradient lands
    in the dtype of the parameter it belongs to. The cell below puts notebook 07's model
    through one training step under autocast and records the dtype of every tensor it
    produces.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            "**The default is the older format.** `torch.autocast(device_type='cuda')` with no `dtype` argument "
            "casts to `float16`, not `bfloat16` — a default set when Volta and Turing cards had `float16` tensor "
            "cores and nothing else. On any card from Ampere on, pass `dtype=torch.bfloat16` explicitly; the "
            "section on the loss scaler shows what `float16` costs when you forget."
        ),
        kind="info",
    )
    return


@app.cell
def _(NeuralNetwork, device, nn, torch):
    torch.manual_seed(0)
    model = NeuralNetwork().to(device)
    images = torch.rand(64, 1, 28, 28, device=device)
    labels = torch.randint(0, 10, (64,), device=device)

    # Forward hooks record what each layer hands to the next; the model itself is untouched.
    dtypes = {"input images": images.dtype}
    for index, layer in enumerate(model.linear_relu_stack):
        layer.register_forward_hook(
            lambda module, inputs, output, i=index: dtypes.__setitem__(
                f"linear_relu_stack[{i}] {module}", output.dtype
            )
        )

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits = model(images)
        loss = nn.functional.cross_entropy(logits, labels)
    loss.backward()

    dtypes["loss"] = loss.dtype
    dtypes["linear_relu_stack[0].weight"] = model.linear_relu_stack[0].weight.dtype
    dtypes["linear_relu_stack[0].weight.grad"] = model.linear_relu_stack[0].weight.grad.dtype
    dtypes["torch.get_autocast_dtype(device) — the default when dtype= is omitted"] = torch.get_autocast_dtype(device)
    return dtypes, model


@app.cell(hide_code=True)
def _(dtypes, mo):
    mo.ui.table(
        [{"tensor": name, "dtype": str(dtype).removeprefix("torch.")} for name, dtype in dtypes.items()],
        selection=None,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `float32` images enter the first `Linear` and `bfloat16` comes out; `ReLU` passes
    what it is given; the logits are `bfloat16`; `cross_entropy` promotes them and the loss
    is `float32`. The weight was `float32` before, is `float32` after, and its gradient is
    `float32` too — the matmul's `bfloat16` gradient was cast up to match. Every 16-bit
    tensor in that table is transient: it lives for one step and dies. Everything that
    persists across steps is 32-bit. That is the whole design, and *master weights* is its
    name.

    ### Master weights

    "Master" because the `float32` copy is the one that accumulates; the `bfloat16` copy
    the matmul sees is derived from it each step and thrown away. The cell takes the
    gradients the backward pass above just produced and asks, for three learning rates:
    if the update $-\eta\, g$ were applied to a 16-bit copy of the weight instead, how many
    parameters would not move at all?
    """)
    return


@app.cell
def _(model, torch):
    def share_swallowed(dtype, learning_rate):
        swallowed = total = 0
        for parameter in model.parameters():
            moving = parameter.grad != 0  # a zero gradient moves nothing in any format; count only the rest
            weight, update = parameter.detach()[moving], learning_rate * parameter.grad[moving]
            stored = weight.to(dtype)
            swallowed += ((stored.float() - update).to(dtype) == stored).sum().item()
            total += moving.sum().item()
        return swallowed / total

    swallowed_shares = [
        {
            "learning rate": learning_rate,
            "swallowed in bfloat16": f"{share_swallowed(torch.bfloat16, learning_rate):.1%}",
            "swallowed in float16": f"{share_swallowed(torch.float16, learning_rate):.1%}",
            "swallowed in float32": f"{share_swallowed(torch.float32, learning_rate):.1%}",
        }
        for learning_rate in (0.001, 0.01, 0.1)
    ]

    return (swallowed_shares,)


@app.cell(hide_code=True)
def _(mo, swallowed_shares):
    mo.ui.table(swallowed_shares, selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The table counts nonzero updates that disappear for this gradient snapshot. It starts
    from weights stored in the selected dtype, subtracts the original gradient's update
    in `float32`, then rounds back to the storage dtype. This isolates update rounding;
    it does not simulate an entire low-precision training run. Future gradients can change.
    The `float32` column is the control: keeping the accumulating weights in `float32`
    preserves much smaller updates than storing those weights in `bfloat16`.

    ### Why `float16` usually uses a scaler and `bfloat16` usually does not

    `float16`'s smallest positive normal value is about $6 \times 10^{-5}$; subnormals
    extend to about $6 \times 10^{-8}$. A value near $10^{-6}$ is representable, while
    sufficiently smaller values round to zero. Tiny gradients can therefore disappear
    during the FP16 backward pass even though parameter gradients are stored in FP32.
    Autocast selects operation dtypes; loss scaling addresses this separate underflow
    problem. Multiply the loss by a factor $S$ before `backward()` to enlarge the
    gradients, then unscale them before the optimizer uses them.
    That is all `torch.amp.GradScaler` does, plus one piece of adaptivity: $S$ starts at
    $2^{16}$, and whenever a scaled gradient overflows to `inf` the scaler *skips that
    optimizer step* and halves $S$; after 2 000 consecutive clean steps it doubles $S$ again.
    `bfloat16` has the same exponent width as `float32`, so its much wider range normally
    makes loss scaling unnecessary. Neither dtype nor scaling guarantees stable training.

    The cell runs five `float16` steps and poisons the third gradient with an `inf` by
    hand. Watch the scale and the weights.
    """)
    return


@app.cell
def _(NeuralNetwork, device, nn, torch):
    # A fresh model keeps reruns independent of the earlier dtype/weight exhibits.
    torch.manual_seed(0)
    _scaler_model = NeuralNetwork().to(device)
    # The defaults: init_scale=65536, growth_factor=2, backoff_factor=0.5, growth_interval=2000
    scaler = torch.amp.GradScaler(device)
    optimizer = torch.optim.SGD(_scaler_model.parameters(), lr=0.1)
    images_16 = torch.rand(64, 1, 28, 28, device=device)
    labels_16 = torch.randint(0, 10, (64,), device=device)
    first_weight = _scaler_model.linear_relu_stack[0].weight

    scaler_trace = []
    for step in range(5):
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.float16):
            loss_16 = nn.functional.cross_entropy(_scaler_model(images_16), labels_16)
        scaler.scale(loss_16).backward()  # gradients arrive multiplied by the scale
        if step == 2:
            first_weight.grad[0, 0] = float("inf")  # what an overflow in the backward pass looks like
        before = first_weight.detach().clone()
        scaler.step(optimizer)  # unscales, checks for inf/nan, steps only if clean
        scaler.update()  # halves the scale after a skipped step; doubles after 2000 clean ones
        scaler_trace.append(
            {
                "step": step,
                "scale after update": scaler.get_scale(),
                "weights moved": bool((first_weight != before).any()),
                "note": "gradient poisoned with inf" if step == 2 else "",
            }
        )

    return (scaler_trace,)


@app.cell(hide_code=True)
def _(mo, scaler_trace):
    mo.ui.table(scaler_trace, selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Step 2 — the poisoned one — leaves the weights exactly where they were and halves the
    scale from 65 536 to 32 768; the steps around it move the weights and leave the scale
    alone. A real overflow is handled the same way: the batch is lost, the scale drops, and
    training continues. BF16 autocast normally needs no enabled scaler;
    `torch.amp.GradScaler(device, enabled=False)` is the idiom that lets one loop serve both,
    and the loop below uses it.

    ## Notebook 07's loop, made mixed

    Here is `train_loop` from notebook 07 with the recipe applied. Four lines change, each
    marked: the forward pass and the loss move inside `autocast`; `loss.backward()` becomes
    `scaler.scale(loss).backward()`; `optimizer.step()` becomes `scaler.step(optimizer)`; and
    `scaler.update()` follows. With `autocast_dtype=None` and a disabled scaler every one of
    those lines is a no-op and this *is* notebook 07's loop — so the same function trains
    the `float32` control. Two smaller edits carry no precision meaning: the batch moves to
    the accelerator, which 07's loop did not do, and the loss is recorded every twenty
    batches for the chart instead of printed every hundred.
    """)
    return


@app.cell
def _(device, torch):
    def train_loop(dataloader, model, loss_fn, optimizer, scaler, autocast_dtype=None):
        model.train()
        history = []
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            # changed: forward pass and loss run under autocast (disabled, they stay float32)
            with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                pred = model(X)
                loss = loss_fn(pred, y)

            scaler.scale(loss).backward()  # changed from loss.backward(): the loss is multiplied by the scale
            scaler.step(optimizer)  # changed from optimizer.step(): unscale, check for inf, step if clean
            scaler.update()  # changed: new line, adapts the scale
            optimizer.zero_grad()

            if batch % 20 == 0:
                history.append({"batches": batch, "loss": loss.item()})
        return history

    def test_loop(dataloader, model, loss_fn, autocast_dtype=None):
        model.eval()
        size = len(dataloader.dataset)
        test_loss, correct = 0.0, 0
        with (
            torch.no_grad(),
            torch.autocast(device_type=device, dtype=autocast_dtype, enabled=autocast_dtype is not None),
        ):
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item() * len(y)
                correct += (pred.argmax(1) == y).to(torch.float32).sum().item()
        return {"loss": test_loss / size, "accuracy": correct / size}

    return test_loop, train_loop


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The cell below trains three fresh copies of the model for one epoch each — the `float32`
    control, `bfloat16` autocast, and `float16` autocast with the scaler — from the same
    seed and in the same batch order, at the learning rate 07's interactive section settled
    on. Predict how far apart the three loss curves will be before pressing the button.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    start_recipe = mo.ui.run_button(label="Train the three loops", kind="success")
    start_recipe
    return (start_recipe,)


@app.cell
def _(
    DataLoader,
    NeuralNetwork,
    device,
    mo,
    nn,
    start_recipe,
    test_data,
    test_loop,
    time,
    torch,
    train_loop,
    training_data,
):
    mo.stop(
        mo.running_in_notebook() and not start_recipe.value,
        mo.md("Press **Train the three loops**. Nothing runs until you do."),
    )

    recipe_modes = {"float32": None, "bfloat16 autocast": torch.bfloat16, "float16 autocast + scaler": torch.float16}
    recipe_runs = {}
    with mo.status.progress_bar(total=len(recipe_modes), title="training", completion_title="done") as _bar:
        for _name, _autocast_dtype in recipe_modes.items():
            torch.manual_seed(0)
            _model = NeuralNetwork().to(device)
            _loader = DataLoader(
                training_data, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(0)
            )
            _scaler = torch.amp.GradScaler(device, enabled=_autocast_dtype is torch.float16)
            _optimizer = torch.optim.SGD(_model.parameters(), lr=0.1)
            torch.accelerator.synchronize()
            _started = time.perf_counter()
            _history = train_loop(_loader, _model, nn.CrossEntropyLoss(), _optimizer, _scaler, _autocast_dtype)
            torch.accelerator.synchronize()
            _seconds = time.perf_counter() - _started
            _test = test_loop(DataLoader(test_data, batch_size=64), _model, nn.CrossEntropyLoss(), _autocast_dtype)
            recipe_runs[_name] = {
                "history": _history,
                "seconds": _seconds,
                "scale": _scaler.get_scale() if _scaler.is_enabled() else None,
                **_test,
            }
            _bar.update(subtitle=_name)
    return (recipe_runs,)


@app.cell(hide_code=True)
def _(FORMAT_COLORS, alt, furnish, mo, pd, recipe_runs):
    _colors = {
        "float32": FORMAT_COLORS["float32"],
        "bfloat16 autocast": FORMAT_COLORS["bfloat16"],
        "float16 autocast + scaler": FORMAT_COLORS["float16"],
    }
    _frame = pd.DataFrame([{"mode": name, **point} for name, run in recipe_runs.items() for point in run["history"]])
    _chart = (
        alt.Chart(_frame)
        .mark_line(opacity=0.9)
        .encode(
            x=alt.X("batches:Q", title="batches seen"),
            y=alt.Y("loss:Q", title="training loss", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "mode:N",
                scale=alt.Scale(domain=list(_colors), range=list(_colors.values())),
                legend=alt.Legend(title=None, orient="top-right"),
            ),
            tooltip=["mode:N", "batches:Q", alt.Tooltip("loss:Q", format=".3f")],
            # The curves coincide; widths let the lower ones show as halos under the top one.
            strokeWidth=alt.StrokeWidth(
                "mode:N", scale=alt.Scale(domain=list(_colors), range=[5, 2.5, 1.2]), legend=None
            ),
        )
        .properties(width=620, height=260)
    )
    _table = [
        {
            "mode": name,
            "test accuracy": f"{run['accuracy']:.1%}",
            "test loss": f"{run['loss']:.3f}",
            "epoch, seconds": f"{run['seconds']:.1f}",
            "final loss scale": f"{run['scale']:.0f}" if run["scale"] is not None else "—",
        }
        for name, run in recipe_runs.items()
    ]
    mo.vstack(
        [
            furnish(_chart),
            mo.md(
                "<small>Training loss every twenty batches, one epoch of 938 batches of 64, the same seed and batch "
                "order for all three; `float32` is drawn widest so the others sit on it. Time is the epoch on this "
                "machine's GPU, data loading included.</small>"
            ),
            mo.ui.table(_table, selection=None),
        ],
        align="center",
        gap=0.6,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The three curves lie on top of each other and the three accuracies agree to a fraction
    of a percent: mixed precision changed what the loop computes with, not what it learns.
    The `float16` scale is still at its starting value — nothing in one epoch of this model
    overflowed and 938 steps is short of the 2 000 that would double it.

    The time column is the disappointment, and it is the honest one. The matmul benchmark
    promised a large factor and the loop shows nothing of the kind — this loop is bound by
    the loader and by the launch of many small kernels, not by arithmetic. The last section
    takes the loader out of the picture and asks where the factor went.

    ## The loop, in five precisions

    The cell trains for one epoch five times, with the model width and batch size you
    choose, using the same `train_loop` — fed with batches that already sit on the GPU as
    slices of one tensor, so nothing but the arithmetic is being timed — and reports the
    time per step, the peak memory and the test accuracy of each. Two modes are added to
    the three above:

    - **TF32** — `float32` after `set_float32_matmul_precision("high")`.
    - **pure bfloat16** — the model itself cast to `bfloat16`, no `float32` copy anywhere.
      Not mixed precision at all; included because it is the fastest and smallest, and
      because of what happens to it at notebook 07's learning rate.

    Worth trying, in this order:

    - **Width 512, batch 256** — close to 07's shape. Autocast is *slower* than `float32`
      here. The matmuls are tiny; the step is dominated by launching kernels and moving the
      weights (read for the forward, written as gradients, read and written by the
      optimizer), and casting adds kernels while saving no bytes on any of those.
    - **Width 8192, batch 4096.** Now each step does sixteen times more arithmetic per byte
      of weight moved, the tensor cores are the bottleneck, and `bfloat16` autocast runs
      between two and three times faster than `float32` — the matmul benchmark's ratio,
      recovered. The two 16-bit modes are indistinguishable in speed here.
    - **Learning rate 0.001, any shape.** Pure `bfloat16` stops learning: chance accuracy,
      loss flat. The other four are identical to two decimal places. Master weights are not
      a refinement; at this learning rate they are the difference between a model and none.
    - **Peak memory, wide model.** Autocast *raises* it at batch 256 and lowers it at 4096.
      The cached 16-bit weight copies cost memory; the halved activations save it; which
      wins is a property of the shape. The memory savings mixed precision is famous for
      come from activations, and an MLP with a small batch barely has any.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    width_pick = mo.ui.slider(steps=[512, 2048, 8192], value=512, label="hidden width", show_value=True)
    batch_pick = mo.ui.slider(steps=[256, 1024, 4096], value=256, label="batch size", show_value=True)
    rate_pick = mo.ui.dropdown({"0.1": 0.1, "0.001 — notebook 07's": 0.001}, value="0.1", label="learning rate")
    start_comparison = mo.ui.run_button(label="Train five ways", kind="success")
    mo.hstack(
        [width_pick, batch_pick, rate_pick, start_comparison],
        justify="start",
        align="center",
        gap=2,
        wrap=True,
    )
    return batch_pick, rate_pick, start_comparison, width_pick


@app.cell
def _(device, test_data, torch, training_data):
    # The raw uint8 pixels, scaled the way ToDtype(scale=True) scales them, moved to the
    # accelerator once: 188 MB for the training set. A batch is then an index into it.
    train_images = (training_data.data.to(torch.float32) / 255).unsqueeze(1).to(device)
    train_labels = training_data.targets.to(device)
    test_images = (test_data.data.to(torch.float32) / 255).unsqueeze(1).to(device)
    test_labels = test_data.targets.to(device)

    class ResidentBatches:
        """What train_loop and test_loop need from a DataLoader, over tensors on the device."""

        def __init__(self, images, labels, batch_size, dtype=torch.float32, seed=0):
            order = torch.randperm(len(images), device=device, generator=torch.Generator(device).manual_seed(seed))
            self.batches = [(images[i].to(dtype), labels[i]) for i in order.split(batch_size)]
            self.dataset = images

        def __iter__(self):
            return iter(self.batches)

        def __len__(self):
            return len(self.batches)

    return (
        ResidentBatches,
        test_images,
        test_labels,
        train_images,
        train_labels,
    )


@app.cell
def _(
    NeuralNetwork,
    ResidentBatches,
    device,
    nn,
    test_images,
    test_labels,
    test_loop,
    time,
    torch,
    train_images,
    train_labels,
    train_loop,
):
    PRECISION_MODES = ("float32", "TF32", "bfloat16 autocast", "float16 autocast + scaler", "pure bfloat16")

    def train_one_epoch(mode, width, batch_size, learning_rate, seed=0):
        torch.manual_seed(seed)
        model = NeuralNetwork(width).to(device)
        weight_dtype = torch.float32
        if mode == "pure bfloat16":
            model, weight_dtype = model.to(torch.bfloat16), torch.bfloat16
        torch.set_float32_matmul_precision("high" if mode == "TF32" else "highest")
        autocast_dtype = {"bfloat16 autocast": torch.bfloat16, "float16 autocast + scaler": torch.float16}.get(mode)
        scaler = torch.amp.GradScaler(device, enabled=mode == "float16 autocast + scaler")
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        batches = ResidentBatches(train_images, train_labels, batch_size, weight_dtype, seed)
        # The first call of each kernel pays for loading it: three unclocked warm-up batches.
        train_loop(batches.batches[:3], model, loss_fn, optimizer, scaler, autocast_dtype)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        torch.accelerator.synchronize()
        started = time.perf_counter()
        history = train_loop(batches, model, loss_fn, optimizer, scaler, autocast_dtype)
        torch.accelerator.synchronize()
        seconds_per_step = (time.perf_counter() - started) / len(batches)
        peak_mib = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else float("nan")
        torch.set_float32_matmul_precision("highest")

        test = test_loop(
            ResidentBatches(test_images, test_labels, 10_000, weight_dtype), model, loss_fn, autocast_dtype
        )
        parameters = sum(p.numel() for p in model.parameters())
        return {
            "mode": mode,
            "ms per step": seconds_per_step * 1e3,
            "TFLOP/s": 6 * parameters * batch_size / seconds_per_step / 1e12,
            "peak MiB": peak_mib,
            "test accuracy": test["accuracy"],
            "final loss": history[-1]["loss"],
            "parameters": parameters,
        }

    return PRECISION_MODES, train_one_epoch


@app.cell
def _(
    PRECISION_MODES,
    batch_pick,
    mo,
    rate_pick,
    start_comparison,
    train_one_epoch,
    width_pick,
):
    mo.stop(
        mo.running_in_notebook() and not start_comparison.value,
        mo.md("Set the shape above, then press **Train five ways**. Nothing runs until you do."),
    )

    comparison = []
    with mo.status.progress_bar(total=len(PRECISION_MODES), title="training", completion_title="done") as _bar:
        for _mode in PRECISION_MODES:
            comparison.append(train_one_epoch(_mode, width_pick.value, batch_pick.value, rate_pick.value))
            _bar.update(subtitle=_mode)
    return (comparison,)


@app.cell(hide_code=True)
def _(
    FORMAT_COLORS,
    OKABE_ITO,
    PRECISION_MODES,
    alt,
    batch_pick,
    comparison,
    furnish,
    mo,
    pd,
    width_pick,
):
    _colors = {
        "float32": FORMAT_COLORS["float32"],
        "TF32": FORMAT_COLORS["TF32"],
        "bfloat16 autocast": FORMAT_COLORS["bfloat16"],
        "float16 autocast + scaler": FORMAT_COLORS["float16"],
        "pure bfloat16": OKABE_ITO["orange"],
    }
    _frame = pd.DataFrame(comparison)
    _scale = alt.Scale(domain=list(_colors), range=list(_colors.values()))

    def _bars(field, title, fmt):
        return (
            alt.Chart(_frame)
            .mark_bar()
            .encode(
                y=alt.Y("mode:N", sort=list(PRECISION_MODES), title=None),
                x=alt.X(f"{field}:Q", title=title),
                color=alt.Color("mode:N", scale=_scale, legend=None),
                tooltip=["mode:N", alt.Tooltip(f"{field}:Q", format=fmt)],
            )
            + alt.Chart(_frame)
            .mark_text(align="left", dx=4, fontSize=12)
            .encode(
                y=alt.Y("mode:N", sort=list(PRECISION_MODES)),
                x=f"{field}:Q",
                text=alt.Text(f"{field}:Q", format=fmt),
            )
        ).properties(width=300, height=5 * 28)

    _table = [
        {
            "mode": r["mode"],
            "ms / step": f"{r['ms per step']:.2f}",
            "vs float32": f"{_frame['ms per step'][0] / r['ms per step']:.2f}×",
            "TFLOP/s": f"{r['TFLOP/s']:.0f}",
            "peak MiB": f"{r['peak MiB']:.0f}",
            "test accuracy": f"{r['test accuracy']:.1%}",
            "final loss": f"{r['final loss']:.3f}",
        }
        for r in comparison
    ]
    _steps = -(-60_000 // batch_pick.value)
    mo.vstack(
        [
            furnish(
                alt.hconcat(
                    _bars("ms per step", "ms per optimizer step", ".2f"),
                    _bars("peak MiB", "peak memory, MiB", ".0f"),
                    spacing=40,
                )
            ),
            mo.md(
                f"<small>One epoch — {_steps} steps of {batch_pick.value} — with hidden width {width_pick.value}, "
                f"{comparison[0]['parameters']:,} parameters, on this machine's GPU, after three warm-up steps. Peak "
                f"memory is everything allocated on the device during the epoch, the rest of the notebook included. "
                f"TFLOP/s counts 6 · parameters · batch per step (forward and backward).</small>"
            ),
            mo.ui.table(_table, selection=None),
        ],
        align="center",
        gap=0.6,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Where to go next

    - **The official surface.** [Automatic Mixed Precision
      package](https://docs.pytorch.org/docs/stable/amp.html) is the reference — its
      *Autocast Op Reference* is the fixed list of which operations go down to 16 bits and
      which come up to `float32`; [Automatic Mixed Precision
      examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html) has the
      `GradScaler` idioms for gradient clipping, accumulation, and multiple losses.
    - **The two papers.** [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
      (Micikevicius et al., 2017) introduced master weights and loss scaling for `float16`;
      [A Study of BFLOAT16 for Deep Learning Training](https://arxiv.org/abs/1905.12322)
      (Kalamkar et al., 2019) is the case that a `float32` exponent makes the scaler
      unnecessary.
    - **TF32, the forward-looking spelling.** [CUDA
      semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices)
      documents `set_float32_matmul_precision` and the newer per-backend
      `torch.backends.cuda.matmul.fp32_precision`, which is where this control is heading.
    - **Where the loop's time really went.** The width-and-batch experiment is a first
      encounter with the *roofline*: arithmetic per byte moved decides whether faster
      arithmetic helps at all. Horace He's [Making Deep Learning Go
      Brrrr](https://horace.io/brrr_intro.html) is the one essay to read on it.
    - **Eight bits.** The GPU in this machine also has `float8` tensor cores;
      [torchao](https://github.com/pytorch/ao)'s `Float8Linear` and `torch._scaled_mm` are
      where PyTorch exposes them, with per-tensor scaling in place of a global loss scale.
    - **The same four lines, behind a flag.** Hugging Face's `accelerate` wraps exactly this
      recipe: `Accelerator(mixed_precision="bf16")` runs the prepared model's forward under
      `autocast` and casts its outputs back to `float32`, and `accelerator.backward` carries
      the `GradScaler` when the mode is `"fp16"`. Its [mixed precision
      guide](https://huggingface.co/docs/accelerate/usage_guides/mixed_precision) is the
      operational front end for what this notebook did by hand.
    - **In the distributed setting.** Once a model shards across GPUs, mixed precision
      becomes a *policy*: which dtype the shards are stored in, which the compute runs in,
      which the gradients are reduced in — [FSDP's
      MixedPrecision](https://docs.pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.MixedPrecision)
      names the three separately. That is where this repository goes next.
    """)
    return


if __name__ == "__main__":
    app.run()
