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

    # Mixed precision: spend accuracy where it buys learning

    A training step does two different kinds of work. It estimates a direction from a
    minibatch, then adds a small correction to weights that may have survived thousands
    of earlier steps. Those operations need not have the same numerical precision.
    An approximate matrix multiply may give a useful direction. A weight update that
    rounds away contributes nothing, however useful that direction was.

    Mixed precision exploits this difference. We use less expensive arithmetic for
    selected operations while retaining enough precision where information must survive.
    It is a numerical design, not a promise that every 16-bit computation is harmless.
    The question throughout this notebook is: **where could information disappear,
    and which part of the recipe protects it?**

    You will build the answer from representable numbers, follow one real forward and
    backward pass, and reconstruct the training loop. The larger benchmarks come last.
    You can understand the mechanism without waiting for an epoch to finish.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A route through the notebook

    First, predict what rounding will do to a number. Then follow the dtypes through
    `Linear → ReLU → Linear → loss → backward`. Finally, explain two protections:
    FP32 weight storage preserves small updates; loss scaling helps small gradients
    survive an FP16 backward pass. They solve different problems.

    The essential stopping point is **Read the loop as a numerical policy**. The
    training buttons and matrix benchmark are optional experiments, not admission
    tests for understanding. Take whichever question interests you into them.

    **Terms.** FP32 means `float32`, FP16 means `float16`, and BF16 means `bfloat16`.
    Automatic mixed precision (AMP) combines operation-specific dtype selection with
    gradient scaling when needed. CUDA examples here target the NVIDIA GPU; policies
    and hardware support differ on other devices.

    ### The familiar model

    We reuse notebook 07's FashionMNIST multilayer perceptron (MLP). Only its hidden
    width is parameterized for the optional performance experiment. Images and model
    parameters begin in FP32; class labels remain integers. Precision selection is
    not a reason to cast labels or the whole model to a floating-point format.
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
    ## Range and resolution are different resources

    A floating-point number stores a sign, an exponent and a fraction. For a normal
    binary number its value is $(-1)^s\,2^e(1.f)$. The leading 1 is implicit: the
    fraction field supplies the bits after it. More exponent bits give a wider
    **range**. More fraction bits give finer **spacing** within that range.

    | Format | Storage | Exponent bits | Stored fraction bits | What it gives up |
    | --- | --- | --- | --- | --- |
    | FP32 | 32 bits | 8 | 23 | Our higher-precision baseline |
    | FP16 | 16 bits | 5 | 10 | Much of FP32's range |
    | BF16 | 16 bits | 8 | 7 | More resolution, to retain a similar range |

    FP16's smallest positive *normal* value is about $6.10\times10^{-5}$; subnormals
    extend down to $5.96\times10^{-8}$. Its largest finite value is 65,504. BF16 has
    FP32's exponent width, but not all its exact endpoints or representable values.
    Converting FP32 to BF16 rounds to a coarser grid; it is not generally a bit chop.

    **TensorFloat-32 (TF32)** belongs in a different category. It is an NVIDIA matrix
    arithmetic mode with an 8-bit exponent and 10 fraction bits of input precision,
    with FP32 accumulation. The tensors can still be stored as FP32. The diagram's
    TF32 row illustrates arithmetic precision, not a tensor storage dtype.
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
        "FP32 high": OKABE_ITO["sky"],
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
            _hue = FORMAT_COLORS["FP32 high" if _name == "TF32" else _name]
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
                    "label": {"sign": "±", "dropped": f"{_width} omitted fraction bits"}.get(
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
                "`float32` layout; conversion rounds. `float16` trades three exponent bits for three more "
                "fraction bits; TF32 arithmetic uses "
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
    ### Predict the first missing integer

    With $p$ significant bits, every integer through $2^p$ is representable. Immediately
    above that boundary the spacing becomes 2, so $2^p+1$ is the first missing integer.
    Here $p$ includes the implicit leading bit: 11 for FP16, 8 for BF16, 24 for FP32.

    Predict each result, then inspect just the five integers around its boundary.
    A local example is enough; we do not need to allocate millions of numbers.
    """)
    return


@app.cell
def _(formats, torch):
    def first_uncountable(dtype):
        significant_bits = round(1 - torch.log2(torch.tensor(torch.finfo(dtype).eps)).item())
        boundary = 2**significant_bits
        integers = torch.arange(boundary - 2, boundary + 3, dtype=torch.float64)
        skipped = integers[integers.to(dtype).to(torch.float64) != integers]
        return int(skipped[0])

    {name: first_uncountable(dtype) for name, dtype in formats.items()}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The missing integers are 2,049, 257 and 16,777,217 for FP16, BF16 and FP32.
    BF16 rounds 257 to 256 under round-to-nearest, ties-to-even. It can represent 258;
    it has not run out of range, only of resolution at this magnitude.

    ### A ruler whose tick spacing changes

    The distance to the next representable number is a *unit in the last place* (ULP).
    For normal numbers, spacing doubles at each power of two. Relative resolution
    therefore stays roughly constant, not absolute resolution. At the same magnitude,
    BF16's spacing is eight times FP16's, and FP16's is 8,192 times FP32's.

    Near zero, subnormals keep a fixed absolute spacing and progressively lose relative
    precision. Some arithmetic paths flush subnormals to zero. A storage-format limit
    does not guarantee every hardware operation preserves values down to that limit.

    Move the probe and compare **stored value**, **relative error** and **spacing**.
    Try $10^{-8}$ and $10^5$: one tests FP16's lower end, the other its upper end.
    In the visible cell, `torch.nextafter` finds the stored number's next neighbor;
    their difference measures the spacing. The chart uses that same `neighbor_spacing`
    function, so the code you read determines the grid you see.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    probe_exponent = mo.ui.slider(-8, 6, step=0.25, value=-5, label="probe: 10 to the power", show_value=True)
    probe_exponent
    return (probe_exponent,)


@app.cell
def _(formats, probe_exponent, torch):
    def neighbor_spacing(values, dtype):
        stored = values.to(dtype)
        next_number = torch.nextafter(stored, torch.full_like(stored, float("inf")))
        return stored.to(torch.float64), next_number.to(torch.float64) - stored.to(torch.float64)

    probe = torch.tensor(10.0**probe_exponent.value, dtype=torch.float64)
    {name: probe.to(dtype).item() for name, dtype in formats.items()}
    return neighbor_spacing, probe


@app.cell(hide_code=True)
def _(FORMAT_COLORS, alt, formats, furnish, mo, neighbor_spacing, pd, probe, probe_exponent, torch):
    _grid = torch.logspace(-8.5, 6.5, 400, dtype=torch.float64)
    _rows = []
    for _name, _dtype in formats.items():
        _stored, _ulp = neighbor_spacing(_grid, _dtype)
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
        _stored, _ulp = (t.item() for t in neighbor_spacing(probe.reshape(1), _dtype))
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
    ### Three ways information disappears

    **Overflow:** 70,000 cannot be stored as finite FP16. **Underflow:** $10^{-8}$
    rounds to zero in FP16. **Update rounding:** a small addition can round back to
    the original value even when both operands are individually representable.

    BF16 avoids those two particular FP16 range failures; it can still overflow or
    underflow at more extreme magnitudes. Its coarser spacing makes update rounding
    especially easy to encounter.

    Predict the result of adding 0.001 a thousand times. Each addition is rounded back
    into the selected storage dtype. This is deliberately different from a tensor-core
    dot product that accumulates products in a wider format.
    """)
    return


@app.cell
def _(formats, torch):
    def trace_additions(dtype, increment, steps):
        total = torch.tensor(0.0, dtype=dtype)
        trace = [total.item()]
        for _ in range(steps):
            total = total + torch.tensor(increment, dtype=dtype)
            trace.append(total.item())
        return trace

    addition_increment = 0.001
    addition_count = 1000
    addition_traces = {
        name: trace_additions(dtype, addition_increment, addition_count) for name, dtype in formats.items()
    }

    {
        "70 000 in float16": torch.tensor(70_000.0).to(torch.float16).item(),
        "1e-8 in float16": torch.tensor(1e-8).to(torch.float16).item(),
        "1e-8 in bfloat16": torch.tensor(1e-8).to(torch.bfloat16).item(),
        "final running totals": {name: trace[-1] for name, trace in addition_traces.items()},
    }
    return addition_count, addition_increment, addition_traces


@app.cell(hide_code=True)
def _(FORMAT_COLORS, addition_count, addition_increment, addition_traces, alt, furnish, mo, pd):
    _rows = [
        {"format": name, "additions": step, "total": total}
        for name, trace in addition_traces.items()
        for step, total in enumerate(trace)
        if step % 5 == 0 or step == addition_count
    ]
    _frame = pd.DataFrame(_rows)
    _ideal = pd.DataFrame({"additions": [0, addition_count], "total": [0, addition_count * addition_increment]})
    _chart = (
        alt.Chart(_ideal).mark_line(color="gray", strokeDash=[3, 3], opacity=0.6).encode(x="additions:Q", y="total:Q")
        + alt.Chart(_frame)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("additions:Q", title=f"additions of {addition_increment:g}"),
            y=alt.Y("total:Q", title="running total"),
            color=alt.Color(
                "format:N",
                scale=alt.Scale(domain=list(addition_traces), range=[FORMAT_COLORS[name] for name in addition_traces]),
                legend=alt.Legend(title=None, orient="top-left"),
            ),
            tooltip=["format:N", "additions:Q", alt.Tooltip("total:Q", format=".4f")],
        )
    ).properties(width=620, height=240)
    mo.vstack(
        [
            furnish(_chart),
            mo.md(
                f"<small>The chart reads `addition_traces` from the visible experiment above; "
                f"the dashed line is the ideal total, {addition_count * addition_increment:g}. "
                "With the initial settings, BF16 stalls at 0.5: its spacing there is 0.00390625, "
                "and an increment of 0.001 is less than half a step. Change the visible inputs "
                "and rerun that cell to test a different case.</small>"
            ),
        ],
        align="center",
        gap=0.2,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The stalled sum is the important surprise: making 0.001 representable did not make
    repeated addition accurate. As the total grew, its spacing grew too. Eventually
    the new contribution could no longer move it to the next tick.

    A stochastic gradient descent (SGD) update has the same shape:
    $w\leftarrow w-\eta g$. With $\eta=0.001$ and $g=0.01$, the update is $10^{-5}$.
    Near a weight of 0.03, BF16's spacing is about $1.22\times10^{-4}$. That update can
    disappear when the result is stored in BF16, even if we computed the subtraction
    in FP32. Wider intermediate arithmetic and wider persistent storage are separate
    protections. We will measure this distinction on the model's actual gradients.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Follow one step, rather than one dtype

    The model's parameters stay FP32. Inside `torch.autocast`, eligible operations
    select a compute dtype according to device-specific rules. On CUDA, `linear`
    and matrix multiplies can use BF16 here; `cross_entropy` uses FP32. ReLU follows
    its input. An operation absent from the autocast list follows its own dtype rules.
    In-place calls, `out=` calls and explicit `dtype=` arguments are important exceptions.

    Leave the context before calling `backward()`. Autograd's backward operations
    follow the dtypes selected for the corresponding forward operations; leaving
    autocast does **not** turn the whole backward pass into FP32. Gradients eventually
    accumulate into the FP32 parameters' `.grad` buffers. Casting a rounded or zero
    intermediate gradient to FP32 cannot recover information already lost.

    Predict the table before running the cell: input images, each layer's output,
    loss, stored weight, and stored weight gradient. These are observations at Python
    boundaries. They do not reveal the internal accumulator precision of a GPU kernel.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            "**Choose the dtype explicitly.** CUDA autocast defaults to FP16; CPU autocast "
            "defaults to BF16. BF16 is a useful starting point on this GPU because it has "
            "much more range than FP16. It is not universally more accurate: FP16 has finer "
            "spacing within its smaller range. Match the model and supported hardware."
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
    _handles = []
    for index, layer in enumerate(model.linear_relu_stack):
        _handles.append(
            layer.register_forward_hook(
                lambda module, inputs, output, i=index: dtypes.__setitem__(
                    f"linear_relu_stack[{i}] {module}", output.dtype
                )
            )
        )

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits = model(images)
        loss = nn.functional.cross_entropy(logits, labels)
    loss.backward()
    for _handle in _handles:
        _handle.remove()

    dtypes["loss"] = loss.dtype
    dtypes["linear_relu_stack[0].weight"] = model.linear_relu_stack[0].weight.dtype
    dtypes["linear_relu_stack[0].weight.grad"] = model.linear_relu_stack[0].weight.grad.dtype
    dtypes["torch.get_autocast_dtype(device) — the default when dtype= is omitted"] = torch.get_autocast_dtype(device)
    # Export a fresh snapshot: marimo cannot track later in-place changes to model.grad.
    weight_gradient_snapshot = [
        (parameter.detach().clone(), parameter.grad.detach().clone()) for parameter in model.parameters()
    ]
    return dtypes, weight_gradient_snapshot


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
    The table separates **storage** from **execution**. FP32 images produce BF16 linear
    outputs. The loss returns to FP32. The stored parameters and their accumulated
    gradients remain FP32 in this ordinary AMP recipe. Autocast has not converted
    the model in place, and this table does not claim that all intermediate gradients
    were FP32.

    ### What “master weights” means here

    The weights that accumulate updates are the lasting record of learning. In this
    recipe, the ordinary FP32 model parameters already serve that role; there is no
    separate optimizer-owned master copy to find. Autocast can cache eligible lower-
    precision weight casts within its context. Other training systems may keep explicit
    copies or use different parameter, optimizer-state and communication dtypes.

    The next experiment asks a narrow counterfactual: keep this gradient snapshot,
    but store the updated weights in a different dtype. Among parameters with nonzero
    gradients, how many would not move? Predict which learning rate loses the most.
    The preceding cell clones the weights and gradients into `weight_gradient_snapshot`.
    This matters in a reactive notebook: marimo tracks dependencies between cells,
    not later in-place changes inside a model. Rerun the forward/backward cell to
    produce a new snapshot; rerunning the display does not train the model.
    """)
    return


@app.cell
def _(torch, weight_gradient_snapshot):
    def share_swallowed(dtype, learning_rate):
        swallowed = total = 0
        for weight_snapshot, gradient_snapshot in weight_gradient_snapshot:
            moving = gradient_snapshot != 0  # zero gradients move nothing; count only the rest
            weight, update = weight_snapshot[moving], learning_rate * gradient_snapshot[moving]
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
    The table isolates **rounding when an updated weight is stored**. It does not
    simulate an entire BF16 training run: the next gradient would depend on the new
    weights. Even FP32 can swallow a sufficiently small update; its grid is much finer.

    ### Loss scaling protects a different part of the journey

    An FP32 `.grad` buffer can receive zero because an intermediate FP16 gradient
    underflowed *before reaching it*. Scaling the loss changes the values traveling
    through backward. For a fixed scale $S$, the chain rule gives

    $$\nabla_w(SL)=S\nabla_w L,\qquad g=\frac{\nabla_w(SL)}{S}.$$

    In exact arithmetic, unscaling recovers the same gradient, so this is not a larger
    learning rate. In finite precision, scaling can move small gradients into a usable
    range. **Unscaling a gradient is not the same as undoing its rounding.** The useful
    work happened when the scaled intermediate survived instead of becoming zero.

    `GradScaler` also checks gradients for infinities and NaNs. If it finds either,
    it skips the optimizer update and lowers the scale. Repeated finite steps let it
    increase the scale. It does not retry the batch automatically, repair an already
    overflowed forward pass, or solve small updates lost in low-precision weight storage.

    BF16's wider range normally makes a scaler unnecessary. That is a practical
    advantage, not immunity from numerical failure. The next cell deliberately poisons
    the third step's gradient: predict which weights change and when the scale falls.
    """)
    return


@app.cell
def _(NeuralNetwork, device, nn, torch):
    torch.manual_seed(0)
    scaler_demo_model = NeuralNetwork().to(device)
    # The defaults: init_scale=65536, growth_factor=2, backoff_factor=0.5, growth_interval=2000
    scaler = torch.amp.GradScaler(device)
    optimizer = torch.optim.SGD(scaler_demo_model.parameters(), lr=0.1)
    images_16 = torch.rand(64, 1, 28, 28, device=device)
    labels_16 = torch.randint(0, 10, (64,), device=device)
    first_weight = scaler_demo_model.linear_relu_stack[0].weight

    scaler_trace = []
    for step in range(5):
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.float16):
            loss_16 = nn.functional.cross_entropy(scaler_demo_model(images_16), labels_16)
        scaler.scale(loss_16).backward()  # gradients arrive multiplied by the scale
        if step == 2:
            first_weight.grad[0, 0] = float("inf")  # what an overflow in the backward pass looks like
        before = first_weight.detach().clone()
        _scale_before = scaler.get_scale()
        scaler.step(optimizer)  # unscales, checks for inf/nan, steps only if clean
        scaler.update()  # halves the scale after a skipped step; doubles after 2000 clean ones
        scaler_trace.append(
            {
                "step (1-based)": step + 1,
                "scale before step": _scale_before,
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
    The deliberately poisoned third step should leave the inspected weights unchanged
    and reduce the scale. Read the before/after columns rather than assuming a specific
    starting scale survived the earlier steps: natural overflow could also trigger a
    skip. An unchanged weight alone is not proof of a skipped step; it can also have
    a zero or rounded-away update. The injected infinity and scale decrease make
    this experiment interpretable.

    ## Read the loop as a numerical policy

    The four changes to the familiar loop each have a job: `autocast` chooses eligible
    operation dtypes; `scale(loss).backward()` protects the FP16 gradient path;
    `step(optimizer)` unscales and conditionally updates; `update()` adapts the scale.
    Use a disabled scaler for BF16 or the FP32 control. Its methods then forward the
    ordinary work; they do not disable learning.

    Before moving on, explain why these tempting changes are different:

    - `model.bfloat16()` changes persistent parameter storage; BF16 autocast does not.
    - A `.float()` cast after an underflowed gradient gives FP32 zero, not the lost value.
    - A larger learning rate changes the intended update; loss scaling is undone before it.

    **One extension worth remembering:** unscale before gradient clipping. For gradient
    accumulation, keep the scale fixed across the effective batch, then unscale/step/update
    at its boundary. Otherwise you add gradients expressed in different units.
    The [AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)
    show those recipes. The loop below deliberately handles one optimizer and one batch
    per update so that the central mechanism remains visible.
    """)
    return


@app.cell
def _(device, torch):
    def train_loop(dataloader, model, loss_fn, optimizer, scaler, autocast_dtype=None):
        model.train()
        history = []
        for batch, (X, y) in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)
            X, y = X.to(device), y.to(device)
            # changed: forward pass and loss run under autocast (disabled, they stay float32)
            with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                pred = model(X)
                loss = loss_fn(pred, y)

            scaler.scale(loss).backward()  # changed from loss.backward(): the loss is multiplied by the scale
            scaler.step(optimizer)  # changed from optimizer.step(): unscale, check for inf, step if clean
            scaler.update()  # changed: new line, adapts the scale

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
    ### Optional: does the recipe still learn?

    Train fresh FP32, BF16-autocast and scaled-FP16 models from the same initialization
    and batch order. Compare the *trajectory*, final test loss and accuracy, not just
    whether each run completed. Similar curves support this recipe on this model;
    they do not establish that precision is irrelevant for all models.

    Predict a result that would make you investigate: nonfinite loss, a persistently
    diverging curve, or worse accuracy across repeated seeds. A single small difference
    is not yet evidence of a broken implementation. This is a learning experiment,
    not a statistically powered equivalence study.
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
            # Different widths keep nearly overlapping curves visible.
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
                "<small>Training loss every twenty batches: 937 batches of 64 and a final batch of 32, "
                "the same seed and batch "
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
    matmul_size = mo.ui.slider(steps=[1024, 2048, 4096, 8192], value=4096, label="matrix side", show_value=True)
    start_matmul = mo.ui.run_button(label="Measure matrix arithmetic")
    mo.vstack(
        [
            mo.md("## Optional: why faster arithmetic may not make a faster step"),
            mo.md(
                "Tensor cores accelerate suitable matrix operations. Compare time with error against an "
                "FP64 reference. The **FP32 high** row requests a reduced internal-precision policy; "
                "a dtype table alone cannot identify the selected GPU kernel. Convolutions have separate controls."
            ),
            matmul_size,
            start_matmul,
        ]
    )
    return matmul_size, start_matmul


@app.cell
def _(device, matmul_size, time, torch, mo, start_matmul):
    mo.stop(
        mo.running_in_notebook() and not start_matmul.value,
        mo.md("Press **Measure matrix arithmetic** to run this optional benchmark."),
    )
    mo.stop(
        device != "cuda", mo.md("This timing experiment targets CUDA; the numerical exhibits above also work on CPU.")
    )

    def timed(operation, repeats=20):
        for _ in range(3):
            operation()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(repeats):
            operation()
        torch.cuda.synchronize()
        return (time.perf_counter() - started) / repeats

    n = matmul_size.value
    a = torch.randn(n, n, device=device, generator=torch.Generator(device).manual_seed(0))
    b = torch.randn(n, n, device=device, generator=torch.Generator(device).manual_seed(1))
    reference = a.to(torch.float64) @ b.to(torch.float64)
    matmul_modes = {
        "float32": ("highest", torch.float32),
        "FP32 high": ("high", torch.float32),
        "bfloat16": ("highest", torch.bfloat16),
        "float16": ("highest", torch.float16),
    }
    matmul_results = []
    _previous_precision = torch.get_float32_matmul_precision()
    try:
        for mode, (precision, dtype) in matmul_modes.items():
            torch.set_float32_matmul_precision(precision)
            x, y = a.to(dtype), b.to(dtype)
            seconds = timed(lambda x=x, y=y: x @ y)
            error = ((x @ y).to(torch.float64) - reference).norm() / reference.norm()
            matmul_results.append(
                {
                    "mode": mode,
                    "ms": seconds * 1e3,
                    "TFLOP/s": 2 * n**3 / seconds / 1e12,
                    "relative error": error.item(),
                }
            )
    finally:
        torch.set_float32_matmul_precision(_previous_precision)
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
    Compare a matrix multiply with a whole training step. A kernel speedup applies only
    to the time spent in that kernel. Loading examples, launching kernels, moving data,
    casting tensors and applying optimizer updates still cost time. Similar final loss
    does not explain a timing difference; use a profiler before naming a bottleneck.

    ## Optional: change the shape, change the payoff

    The final experiment removes repeated host-to-device batch transfers by preparing
    resident batches. It still includes Python iteration, kernel launches and optimizer
    work. It is not a pure arithmetic benchmark. Three warm-up *training* steps change
    the weights before the timed epoch; every mode receives that same protocol.

    The **FP32 high** mode changes internal matmul precision while retaining FP32
    storage and output. PyTorch documents TF32 and, where available, BF16-based
    algorithms for this policy. The label states what we requested, not a profiled kernel.
    The **pure bfloat16** mode also stores the model in BF16, with no FP32 master
    parameters. It is a comparison of a different numerical policy, not ordinary AMP.

    Investigate one question at a time:

    - At small width/batch, does casting overhead outweigh faster matrix operations?
    - At larger shapes, is enough work in matrix multiplies for tensor cores to help?
    - At learning rate 0.001, does BF16 parameter storage lose useful updates? Use the
      earlier snapshot table to explain a result, not to predict inevitable failure.
    - Does saved activation memory outweigh cast copies and other allocations?

    Treat the numbers as measurements of this run. Peak memory includes other live
    tensors in this kernel; it is not an isolated model footprint. FP32 weights alone
    cost about $4P$ bytes for $P$ parameters, and their FP32 gradients another $4P$.
    Autocast does not halve those terms. Optimizer state, activations, cast copies and
    workspaces add their own costs. Our plain SGD has no Adam moment buffers.
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
def _(device, test_data, torch, training_data, mo, start_comparison):
    mo.stop(
        mo.running_in_notebook() and not start_comparison.value,
        mo.md("Resident GPU data is prepared only when the five-mode experiment is requested."),
    )
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
    PRECISION_MODES = ("float32", "FP32 high", "bfloat16 autocast", "float16 autocast + scaler", "pure bfloat16")

    def train_one_epoch(mode, width, batch_size, learning_rate, seed=0):
        torch.manual_seed(seed)
        model = NeuralNetwork(width).to(device)
        weight_dtype = torch.float32
        if mode == "pure bfloat16":
            model, weight_dtype = model.to(torch.bfloat16), torch.bfloat16
        _previous_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("high" if mode == "FP32 high" else "highest")
        try:
            autocast_dtype = {"bfloat16 autocast": torch.bfloat16, "float16 autocast + scaler": torch.float16}.get(
                mode
            )
            scaler = torch.amp.GradScaler(device, enabled=mode == "float16 autocast + scaler")
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            loss_fn = nn.CrossEntropyLoss()

            batches = ResidentBatches(train_images, train_labels, batch_size, weight_dtype, seed)
            # Three unclocked training steps warm up the kernels and update the model.
            train_loop(batches.batches[:3], model, loss_fn, optimizer, scaler, autocast_dtype)
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            torch.accelerator.synchronize()
            started = time.perf_counter()
            history = train_loop(batches, model, loss_fn, optimizer, scaler, autocast_dtype)
            torch.accelerator.synchronize()
            seconds_per_step = (time.perf_counter() - started) / len(batches)
            peak_mib = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else float("nan")

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
        finally:
            torch.set_float32_matmul_precision(_previous_precision)

    return PRECISION_MODES, train_one_epoch


@app.cell
def _(PRECISION_MODES, batch_pick, mo, rate_pick, start_comparison, train_one_epoch, width_pick):
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
def _(FORMAT_COLORS, OKABE_ITO, PRECISION_MODES, alt, batch_pick, comparison, furnish, mo, pd, width_pick):
    _colors = {
        "float32": FORMAT_COLORS["float32"],
        "FP32 high": FORMAT_COLORS["FP32 high"],
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
                f"TFLOP/s is an approximate 6 · parameters · batch per step, not a kernel instruction count.</small>"
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
    ## Carry the distinctions forward

    You now have a way to reason about a mixed-precision configuration without treating
    its name as an explanation. Ask which dtype stores parameters, which operations
    use reduced precision, where gradients accumulate, and what happens on a nonfinite
    step. In distributed training, add the dtype used to communicate gradients.

    **A useful integration-test question:** does the prepared training workflow preserve
    the intended update, within a justified tolerance, when precision or process count
    changes? Control initialization, examples per update, loss normalization and optimizer
    steps before interpreting a difference. A dtype assertion checks one boundary;
    a loss trajectory or parameter comparison checks more of the behavior. Neither alone
    proves convergence, and a tolerance chosen only to make a test pass explains nothing.

    ### Sources and deeper paths

    - [Micikevicius et al., *Mixed Precision Training*](https://arxiv.org/abs/1710.03740),
      submitted 2017, ICLR 2018: FP32 weight accumulation and loss scaling in an FP16
      training recipe. Our native AMP implementation need not have the paper's exact
      arrangement of explicit weight copies.
    - [Kalamkar et al., *A Study of BFLOAT16 for Deep Learning Training*](https://arxiv.org/abs/1905.12322),
      2019: empirical support for BF16 across several training workloads. Successful
      workloads are evidence, not a guarantee for every model.
    - [PyTorch AMP reference](https://docs.pytorch.org/docs/stable/amp.html) and
      [worked examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html):
      operation eligibility, scaling, clipping and accumulation. These policies are
      version- and device-dependent; check the installed version when behavior differs.
    - [Matmul precision policy](https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html):
      why FP32 storage does not fully specify internal arithmetic. Do not mix old and
      new backend-control APIs without checking their compatibility.
    - [Williams, Waterman and Patterson, *Roofline*](https://doi.org/10.1145/1498765.1498785),
      2009: relate arithmetic throughput to data movement. Use it to form a performance
      hypothesis, then measure which limit matters in the actual workload.
    - [Accelerator](https://huggingface.co/docs/accelerate/package_reference/accelerator)
      makes precision a configuration choice. Inspect the prepared forward, backward and
      optimizer together: the public output dtype alone need not expose autocast inside.
    - [Fully Sharded Data Parallel mixed-precision policy](https://docs.pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.MixedPrecision)
      separates parameter, reduction and buffer dtypes. Sharding adds more boundaries;
      it does not remove the distinctions you just learned.
    """)
    return


if __name__ == "__main__":
    app.run()
