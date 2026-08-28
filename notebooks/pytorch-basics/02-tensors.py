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
# Cells under an "Explore" heading are additions; everything else is the upstream
# tutorial as converted.

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    # '%matplotlib inline' command supported automatically in marimo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [Learn the Basics](intro.html) \|\| [Quickstart](quickstart_tutorial.html) \|\| **Tensors**
    \|\| [Datasets & DataLoaders](data_tutorial.html) \|\| [Transforms](transforms_tutorial.html)
    \|\| [Build Model](buildmodel_tutorial.html) \|\| [Autograd](autogradqs_tutorial.html) \|\|
    [Optimization](optimization_tutorial.html) \|\| [Save & Load Model](saveloadrun_tutorial.html)

    # Tensors

    Tensors are a specialized data structure that are very similar to arrays and matrices. In
    PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model’s
    parameters.

    Tensors are similar to [NumPy’s](https://numpy.org/) ndarrays, except that tensors can run on
    GPUs or other hardware accelerators. In fact, tensors and NumPy arrays can often share the same
    underlying memory, eliminating the need to copy data (see `bridge-to-np-label`). Tensors are
    also optimized for automatic differentiation (we'll see more about that later in the
    [Autograd](autogradqs_tutorial.html) section). If you’re familiar with ndarrays, you’ll be
    right at home with the Tensor API. If not, follow along!
    """)
    return


@app.cell
def _():
    import numpy as np
    import torch

    return np, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Initializing a Tensor

    Tensors can be initialized in various ways. Take a look at the following examples:

    **Directly from data**

    Tensors can be created directly from data. The data type is automatically inferred.
    """)
    return


@app.cell
def _(torch):
    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)
    return data, x_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **From a NumPy array**

    Tensors can be created from NumPy arrays (and vice versa - see `bridge-to-np-label`).
    """)
    return


@app.cell
def _(data, np, torch):
    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **From another tensor:**

    The new tensor retains the properties (shape, datatype) of the argument tensor, unless
    explicitly overridden.
    """)
    return


@app.cell
def _(torch, x_data):
    x_ones = torch.ones_like(x_data)  # retains the properties of x_data
    print(f"Ones Tensor: \n {x_ones} \n")

    x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data
    print(f"Random Tensor: \n {x_rand} \n")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **With random or constant values:**

    `shape` is a tuple of tensor dimensions. In the functions below, it determines the
    dimensionality of the output tensor.
    """)
    return


@app.cell
def _(torch):
    shape = (2, 3)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)

    print(f"Random Tensor: \n {rand_tensor} \n")
    print(f"Ones Tensor: \n {ones_tensor} \n")
    print(f"Zeros Tensor: \n {zeros_tensor}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    -----------------------------------------------------------------------------------------------
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Attributes of a Tensor

    Tensor attributes describe their shape, datatype, and the device on which they are stored.
    """)
    return


@app.cell
def _(torch):
    tensor = torch.rand(3, 4)

    print(f"Shape of tensor: {tensor.shape}")
    print(f"Datatype of tensor: {tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device}")
    return (tensor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Explore — `dtype` is three decisions, not one

    `torch.float32` is the default and the tutorial moves on. It is worth stopping,
    because the choice of dtype decides how much memory a model needs, how fast it
    trains, and what it can represent — and the two 16-bit formats below differ from each
    other more than either differs from `float32`.

    A floating point number is a sign, an exponent and a mantissa. The exponent sets the
    *range*, the mantissa sets the *precision*, and 16 bits has to be split between them:

    - **float16** spends its bits on mantissa. Finer steps than bfloat16, but the largest
      number it can hold is 65,504 and the smallest normal one is 6.1e-05. Gradients
      routinely fall below that and become zero — which is what a gradient scaler exists
      to prevent, by multiplying the loss up before the backward pass and dividing it out
      after.
    - **bfloat16** spends them on exponent. It has the *identical* range to float32, so
      nothing underflows that would not have underflowed anyway, and no scaler is needed.
      The price is precision: its steps are eight times coarser than float16's.

    Type a number and watch what each format does to it.
    """)
    return


@app.cell
def _(mo):
    stored_value = mo.ui.text("0.1", label="store this number as")
    stored_value
    return (stored_value,)


@app.cell
def _(mo, stored_value, torch):
    try:
        _exact = float(stored_value.value)
    except ValueError:
        _exact = 0.1

    _rows = []
    for _dtype in (torch.float64, torch.float32, torch.bfloat16, torch.float16):
        _info = torch.finfo(_dtype)
        _kept = torch.tensor(_exact, dtype=_dtype).item()
        _rows.append(
            {
                "dtype": str(_dtype).removeprefix("torch."),
                "bytes": torch.empty(0, dtype=_dtype).element_size(),
                "stored as": f"{_kept!r}",
                "error": f"{abs(_kept - _exact):.3e}",
                "step near 1.0 (eps)": f"{_info.eps:.3e}",
                "largest": f"{_info.max:.3e}",
                "smallest normal": f"{_info.tiny:.1e}",
            }
        )
    mo.vstack(
        [
            mo.ui.table(_rows, selection=None),
            mo.md(
                "A 669,706-parameter model needs "
                f"**{669706 * 4 / 1024**2:.1f} MB** in float32 and "
                f"**{669706 * 2 / 1024**2:.1f} MB** in either 16-bit format. That halving is "
                "why mixed precision exists; the two rows above are why it is *mixed* rather "
                "than simply 16-bit, since the optimizer keeps a float32 copy of the weights "
                "to accumulate into."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    -----------------------------------------------------------------------------------------------
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Operations on Tensors

    Over 1200 tensor operations, including arithmetic, linear algebra, matrix manipulation
    (transposing, indexing, slicing), sampling and more are comprehensively described
    [here](https://pytorch.org/docs/stable/torch.html).

    Each of these operations can be run on the CPU and
    [Accelerator](https://pytorch.org/docs/stable/torch.html#accelerators) such as CUDA, MPS, MTIA,
    or XPU. If you’re using Colab, allocate an accelerator by going to Runtime \> Change runtime
    type \> GPU.

    By default, tensors are created on the CPU. We need to explicitly move tensors to the
    accelerator using `.to` method (after checking for accelerator availability). Keep in mind that
    copying large tensors across devices can be expensive in terms of time and memory!
    """)
    return


@app.cell
def _(tensor, torch):
    # We move our tensor to the current accelerator if available
    if torch.accelerator.is_available():
        tensor_1 = tensor.to(torch.accelerator.current_accelerator())

    torch.accelerator.current_accelerator()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Explore — "can be expensive" as a number

    The paragraph above ends with a warning that copying large tensors across devices is
    expensive, and gives no figure. Measure it: the slider allocates a tensor of that size
    and times the copy to the accelerator, host to device, five times after a warm-up.

    Two rows, because host memory comes in two kinds. *Pageable* is what `torch.empty`
    gives you and the operating system may move it around, so the driver copies it into a
    staging buffer first. *Pinned* memory is locked in place, which is what
    `DataLoader(pin_memory=True)` allocates.

    Read the measured numbers rather than the folklore. On this machine pinning buys
    almost nothing in raw bandwidth — both land near the same GB/s — because the staging
    copy is not the bottleneck here. What pinning actually buys is that the transfer can
    be issued asynchronously, `non_blocking=True`, and overlap with computation already on
    the device. That overlap is the reason it appears in every input pipeline, and it does
    not show up in a benchmark that waits for the copy to finish, as this one does.
    """)
    return


@app.cell
def _(mo):
    transfer_size = mo.ui.slider(steps=[1, 4, 16, 64, 256], value=64, label="megabytes", show_value=True)
    transfer_size
    return (transfer_size,)


@app.cell
def _(mo, torch, transfer_size):
    import time

    mo.stop(
        not torch.accelerator.is_available(),
        mo.callout(mo.md("No accelerator available, so there is nothing to time."), kind="neutral"),
    )

    def _time_copy(megabytes, pinned, repeats=5):
        elements = megabytes * 1024 * 1024 // 4
        host = torch.empty(elements, dtype=torch.float32, pin_memory=pinned)
        device = torch.empty(elements, dtype=torch.float32, device=torch.accelerator.current_accelerator())
        for _ in range(2):
            device.copy_(host, non_blocking=pinned)
        # The copy is asynchronous, so the clock has to be stopped by the device rather
        # than by the return of the Python call.
        torch.accelerator.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            device.copy_(host, non_blocking=pinned)
        torch.accelerator.synchronize()
        return (time.perf_counter() - start) / repeats

    _rows = []
    for _pinned in (False, True):
        _seconds = _time_copy(transfer_size.value, _pinned)
        _rows.append(
            {
                "host memory": "pinned" if _pinned else "pageable",
                "milliseconds": round(_seconds * 1000, 3),
                "GB/s": round(transfer_size.value / 1024 / _seconds, 1),
            }
        )
    mo.vstack(
        [
            mo.ui.table(_rows, selection=None),
            mo.md(
                f"For scale: one batch of 64 FashionMNIST images is "
                f"{64 * 784 * 4 / 1024:.0f} KB, so the copy costs about "
                f"{_rows[0]['milliseconds'] * (64 * 784 * 4 / 1024**2) / transfer_size.value * 1000:.0f} "
                "microseconds — next to nothing. A gradient all-reduce across two GPUs moves "
                f"every parameter instead, and at {669706 * 4 / 1024**2:.1f} MB per copy that "
                "arithmetic is what decides whether a second card makes training faster."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Try out some of the operations from the list. If you're familiar with the NumPy API, you'll
    find the Tensor API a breeze to use.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Standard numpy-like indexing and slicing:**
    """)
    return


@app.cell
def _(torch):
    tensor_2 = torch.rand(4, 4)
    print(f"First row:    {tensor_2[0]}")
    print(f"First column: {tensor_2[:, 0]}")
    print(f"Last column:  {tensor_2[..., -1]}")
    tensor_2[:, 1] = 0
    print(tensor_2)
    return (tensor_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Explore — slicing, if you are arriving from Python lists

    Three lines above carry most of what indexing a tensor involves, and they go past
    quickly. The rules, once:

    **One pair of brackets, one entry per dimension, separated by commas.** A list of
    lists needs `rows[1][3]`, two lookups; a tensor takes `t[1, 3]`, one lookup that the
    shape and stride resolve directly. `t[1]` with the second entry left out means *every*
    column, so `t[1]` and `t[1, :]` are the same tensor.

    **A slice is `start:stop:step`, and `stop` is not included.** Leave a part out and it
    takes its default: `:` is everything, `2:` is from 2 to the end, `:3` is up to but not
    including 3. Negative numbers count from the end, so `t[-1]` is the last row.

    **An integer removes a dimension; a slice keeps it.** `t[0]` has shape `(8,)` and
    `t[0:1]` has shape `(1, 8)`, holding exactly the same numbers. This is the one to
    internalise: it is where a batch dimension quietly disappears and the error surfaces
    three lines later, in a matmul that expected two dimensions and got one.

    **`...` means "as many full dimensions as it takes"**, so `t[..., -1]` is the last
    column of a matrix and the last channel of a 4-dimensional batch, unchanged. And
    `None` inserts a new dimension of size 1, which is how you line two tensors up for
    broadcasting.

    Pick an expression: the highlighted cells are what it selects.
    """)
    return


@app.cell(hide_code=True)
def _():
    import altair as alt
    import pandas as pd

    return alt, pd


@app.cell(hide_code=True)
def _(alt, pd):
    def as_heatmap(matrix, title, cell=54):
        """Render a small 1D or 2D tensor as an annotated heatmap."""
        rows = matrix.tolist() if matrix.dim() == 2 else [matrix.tolist()]
        digits = ".2f" if matrix.dtype.is_floating_point else ".0f"
        cells = pd.DataFrame(
            [{"row": i, "col": j, "value": float(v)} for i, row in enumerate(rows) for j, v in enumerate(row)]
        )
        limit = max(abs(cells["value"].min()), abs(cells["value"].max()), 1e-9)
        base = alt.Chart(cells).encode(x=alt.X("col:O", axis=None), y=alt.Y("row:O", axis=None))
        squares = base.mark_rect(stroke="white", strokeWidth=2).encode(
            color=alt.Color(
                "value:Q",
                scale=alt.Scale(scheme="redblue", domain=[-limit, limit], reverse=True),
                legend=None,
            )
        )
        labels = base.mark_text(fontSize=12, fontWeight=500).encode(
            text=alt.Text("value:Q", format=digits),
            color=alt.condition(f"abs(datum.value) > {0.6 * limit}", alt.value("white"), alt.value("#111111")),
        )
        return (squares + labels).properties(width=cell * len(rows[0]), height=cell * len(rows), title=title)

    return (as_heatmap,)


@app.cell(hide_code=True)
def _(mo):
    slicing = mo.ui.dropdown(
        options={
            "t[0] — one row, and the row dimension is gone": lambda t: t[0],
            "t[0:1] — the same eight numbers, still a matrix": lambda t: t[0:1],
            "t[:, 0] — every row, column zero": lambda t: t[:, 0],
            "t[-1] — the last row": lambda t: t[-1],
            "t[1:5:2] — from 1, stopping before 5, every second": lambda t: t[1:5:2],
            "t[2:4, 3:6] — a block, sliced in both dimensions": lambda t: t[2:4, 3:6],
            "t[..., -1] — the last of whatever the final dimension is": lambda t: t[..., -1],
            "t[None] — insert a dimension of size 1 at the front": lambda t: t[None],
            "t[t % 7 == 0] — a boolean mask": lambda t: t[t % 7 == 0],
            "t[[0, 3, 5]] — pick rows by a list of indices": lambda t: t[[0, 3, 5]],
            "t[:, ::-1] — a negative step, which PyTorch refuses": lambda t: t[:, ::-1],
        },
        value="t[0] — one row, and the row dimension is gone",
        label="`t = torch.arange(48).reshape(6, 8)`, then",
        full_width=True,
    )
    slicing
    return (slicing,)


@app.cell(hide_code=True)
def _(alt, as_heatmap, mo, pd, slicing, torch):
    t_sliced = torch.arange(48).reshape(6, 8)
    try:
        _result = slicing.value(t_sliced)
    except (RuntimeError, ValueError) as error:
        _panel = mo.callout(
            mo.md(
                f"""
                `{type(error).__name__}: {error}`

                NumPy allows a negative step and PyTorch does not, because reversing needs
                a negative stride and PyTorch's storage model has no room for one. Use
                `torch.flip(t, dims=[1])`, which copies, and says so by copying.
                """
            ),
            kind="danger",
        )
        _picked = set()
    else:
        _picked = set(_result.flatten().tolist())
        _shares = _result.untyped_storage().data_ptr() == t_sliced.untyped_storage().data_ptr()
        _facts = mo.md(
            f"`{tuple(_result.shape)}` — {_result.dim()} "
            f"{'dimension' if _result.dim() == 1 else 'dimensions'}, {_result.numel()} elements, "
            f"and {'a **view**: it shares storage with `t`' if _shares else 'a **copy**: new storage'}."
        )
        _panel = mo.vstack(
            [_facts, as_heatmap(_result, "what you get back", cell=44) if _result.dim() <= 2 else mo.md("")]
        )

    _cells = pd.DataFrame(
        [
            {"row": i, "col": j, "value": v, "picked": v in _picked}
            for i, _row in enumerate(t_sliced.tolist())
            for j, v in enumerate(_row)
        ]
    )
    _position = {"x": alt.X("col:O", axis=None), "y": alt.Y("row:O", axis=None)}
    _grid = (
        alt.Chart(_cells)
        .mark_rect(stroke="white", strokeWidth=2)
        .encode(
            **_position,
            color=alt.Color(
                "picked:N",
                scale=alt.Scale(domain=[False, True], range=["#e9e9ee", "#4c78a8"]),
                legend=None,
            ),
        )
        + alt.Chart(_cells)
        .mark_text(fontSize=11)
        .encode(
            **_position,
            text=alt.Text("value:Q"),
            color=alt.condition("datum.picked", alt.value("white"), alt.value("#5a5a66")),
        )
    ).properties(width=8 * 46, height=6 * 46, title="t, with the selected elements filled in")

    mo.hstack([_grid, _panel], justify="start", align="center", gap=2, wrap=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The last two entries are the ones that cost memory. A slice is a view — the same
    storage read with different strides, which is why `tensor_2[:, 1] = 0` above changed
    `tensor_2` itself. A boolean mask or a list of indices cannot be expressed as a
    stride, so PyTorch has to gather the elements into new storage, and writing to the
    result changes nothing in the original. The section at the end of this notebook takes
    that apart.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Joining tensors** You can use `torch.cat` to concatenate a sequence of tensors along a given
    dimension. See also [torch.stack](https://pytorch.org/docs/stable/generated/torch.stack.html),
    another tensor joining operator that is subtly different from `torch.cat`.
    """)
    return


@app.cell
def _(tensor_2, torch):
    t1 = torch.cat([tensor_2, tensor_2, tensor_2], dim=1)
    print(t1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Explore — `cat` against `stack`, which the paragraph above calls "subtly different"

    It is not subtle once the shapes are side by side, and it is worth ten seconds now
    because it is the difference between a batch of 64 images and a single image with 64
    channels.

    **`cat` joins along a dimension that already exists.** The inputs need to agree on
    every *other* dimension, the chosen one adds up, and the result has the same number of
    dimensions it started with.

    **`stack` creates a new dimension.** The inputs must have identical shapes, and the
    result has one dimension more than they did. `dim` says where the new axis goes.

    Two 2x3 tensors, every option:
    """)
    return


@app.cell
def _(mo, torch):
    _left = torch.zeros(2, 3)
    _right = torch.ones(2, 3)
    _rows = [
        {
            "call": f"torch.{_name}([a, b], dim={_dim})",
            "result shape": str(tuple(getattr(torch, _name)([_left, _right], dim=_dim).shape)),
            "dimensions": getattr(torch, _name)([_left, _right], dim=_dim).dim(),
            "elements": getattr(torch, _name)([_left, _right], dim=_dim).numel(),
        }
        for _name, _dims in (("cat", (0, 1)), ("stack", (0, 1, 2)))
        for _dim in _dims
    ]
    mo.vstack(
        [
            mo.md("`a` and `b` are both `(2, 3)` — twelve elements between them, in every row below."),
            mo.ui.table(_rows, selection=None),
            mo.md(
                "Every result holds the same twelve numbers. `cat` arranges them in two "
                "dimensions and `stack` in three, which is the entire difference. The one to "
                "remember is `torch.stack(list_of_images)` — that is how a list of `(1, 28, 28)` "
                "samples becomes the `(64, 1, 28, 28)` batch a `DataLoader` hands you, and it is "
                "why every sample in a batch must have the same shape."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Arithmetic operations**
    """)
    return


@app.cell
def _(tensor_2, torch):
    # This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
    # ``tensor.T`` returns the transpose of a tensor
    y1 = tensor_2 @ tensor_2.T
    y2 = tensor_2.matmul(tensor_2.T)
    y3 = torch.rand_like(y1)
    torch.matmul(tensor_2, tensor_2.T, out=y3)
    z1 = tensor_2 * tensor_2
    z2 = tensor_2.mul(tensor_2)
    z3 = torch.rand_like(tensor_2)
    # This computes the element-wise product. z1, z2, z3 will have the same value
    torch.mul(tensor_2, tensor_2, out=z3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Explore — matrix multiplication you can drag

    Drag horizontally inside a cell of `A` to change it; every panel below recomputes.

    Two things are worth watching. `A @ A.T` stays symmetric whatever you do to `A`,
    because entry *(i, j)* is the dot product of rows *i* and *j* of `A`, and swapping *i*
    and *j* asks for the same dot product. And the two products differ in shape as well as
    in value: `A * A` is element-wise, same shape in and out, while `A @ A.T` contracts
    the three columns away and leaves 2x2. Nearly every shape error in a training script
    is that distinction, met without a picture.
    """)
    return


@app.cell
def _(mo):
    editable = mo.ui.matrix(
        [[1.0, 2.0, -1.0], [0.5, 0.0, 3.0]],
        min_value=-4.0,
        max_value=4.0,
        step=0.25,
        precision=2,
        label="A",
    )
    editable
    return (editable,)


@app.cell
def _(as_heatmap, editable, mo, torch):
    _a = torch.tensor(editable.value)
    mo.hstack(
        [
            as_heatmap(_a, "A"),
            as_heatmap(_a.T, "A.T"),
            as_heatmap(_a @ _a.T, "A @ A.T  (2x2)"),
            as_heatmap(_a * _a, "A * A  (2x3)"),
        ],
        justify="start",
        align="center",
        gap=1,
        wrap=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Explore — broadcasting, one dimension at a time

    The tutorial multiplies tensors of equal shape, which hides the rule that actually
    governs PyTorch arithmetic. Shapes are lined up from the *right*; a missing dimension
    counts as 1, and a dimension of 1 is stretched to meet its partner. Anything else is
    an error.

    Type two shapes. The table shows the alignment PyTorch performs, and the error text
    when there isn't one — worth reading once deliberately, because it is the error you
    will meet with a batch dimension in the wrong place.
    """)
    return


@app.cell
def _(mo):
    left_shape = mo.ui.text("3, 1, 4", label="left")
    right_shape = mo.ui.text("2, 4", label="right")
    mo.hstack([left_shape, right_shape], justify="start", gap=1)
    return left_shape, right_shape


@app.cell
def _(left_shape, mo, right_shape, torch):
    def _parse(text):
        return tuple(int(part) for part in text.replace("(", "").replace(")", "").split(",") if part.strip())

    try:
        _left, _right = _parse(left_shape.value), _parse(right_shape.value)
    except ValueError:
        _output = mo.callout(mo.md("Shapes are comma-separated integers, e.g. `3, 1, 4`."), kind="warn")
    else:
        _width = max(len(_left), len(_right))

        def _align(shape):
            # A dimension PyTorch supplies rather than one you wrote is marked, since that
            # is the half of the rule people forget.
            supplied = ["*1*"] * (_width - len(shape))
            return " | ".join(supplied + [str(d) for d in shape])

        _header = " | ".join(f"dim {i - _width}" for i in range(_width))
        _rows = [
            f"| left `{_left}` | {_align(_left)} |",
            f"| right `{_right}` | {_align(_right)} |",
        ]
        try:
            _result = tuple(torch.broadcast_shapes(_left, _right))
        except RuntimeError as error:
            _table = mo.md("\n".join([f"| | {_header} |", "|" + "---|" * (_width + 1), *_rows]))
            _output = mo.vstack([_table, mo.callout(mo.md(f"`RuntimeError: {error}`"), kind="danger")])
        else:
            _rows.append(f"| **result** | {' | '.join(f'**{d}**' for d in _result)} |")
            _output = mo.md("\n".join([f"| | {_header} |", "|" + "---|" * (_width + 1), *_rows]))
    _output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Single-element tensors** If you have a one-element tensor, for example by aggregating all
    values of a tensor into one value, you can convert it to a Python numerical value using
    `item()`:
    """)
    return


@app.cell
def _(tensor_2):
    agg = tensor_2.sum()
    agg_item = agg.item()
    print(agg_item, type(agg_item))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **In-place operations** Operations that store the result into the operand are called in-place.
    They are denoted by a `_` suffix. For example: `x.copy_(y)`, `x.t_()`, will change `x`.
    """)
    return


@app.cell
def _(tensor_2):
    print(f"{tensor_2} \n")
    tensor_2.add_(5)
    print(tensor_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > [!NOTE]
    > In-place operations save some memory, but can be problematic when computing derivatives
    > because of an immediate loss of history. Hence, their use is discouraged.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    -----------------------------------------------------------------------------------------------
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bridge with NumPy

    Tensors on the CPU and NumPy arrays can share their underlying memory locations, and changing
    one will change the other.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tensor to NumPy array
    """)
    return


@app.cell
def _(torch):
    t = torch.ones(5)
    print(f"t: {t}")
    n = t.numpy()
    print(f"n: {n}")
    return n, t


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A change in the tensor reflects in the NumPy array.
    """)
    return


@app.cell
def _(n, t):
    t.add_(1)
    print(f"t: {t}")
    print(f"n: {n}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # NumPy array to Tensor
    """)
    return


@app.cell
def _(np, torch):
    n_1 = np.ones(5)
    t_1 = torch.from_numpy(n_1)
    return n_1, t_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Changes in the NumPy array reflects in the tensor.
    """)
    return


@app.cell
def _(n_1, np, t_1):
    np.add(n_1, 1, out=n_1)
    print(f"t: {t_1}")
    print(f"n: {n_1}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Explore — what a tensor actually is

    "Tensors share their underlying memory with NumPy arrays" is the same fact as the one
    the section above demonstrates in one direction: a tensor is not its numbers. It is a
    *view* — a shape, a stride and an offset — onto one flat run of memory, and several
    tensors can describe the same run differently.

    Pick an operation. The strip is the storage, twelve numbers in the order they lie in
    memory, and it never changes. The grid is what the resulting tensor claims to be.

    - **stride** is how far to step, in elements, to move one position along each
      dimension. `x.T` does not move a single number: it swaps the two strides.
    - **contiguous** means the strides still walk the storage front to back. Transposing
      breaks that, and any operation needing a linear layout has to copy first.
    - **same storage** answers whether you got a view or a copy. Write into a view and
      the original changes with it.

    `x.T.view(2, 6)` is in the list on purpose: it is the error the `reshape` beneath it
    exists to avoid.
    """)
    return


@app.cell
def _(mo):
    operation = mo.ui.dropdown(
        options={
            "x.T": lambda x: x.T,
            "x.view(2, 6)": lambda x: x.view(2, 6),
            "x.T.view(2, 6)": lambda x: x.T.view(2, 6),
            "x.T.reshape(2, 6)": lambda x: x.T.reshape(2, 6),
            "x[:, 1:3]": lambda x: x[:, 1:3],
            "x[:1].expand(3, 4)": lambda x: x[:1].expand(3, 4),
            "x.contiguous()": lambda x: x.contiguous(),
            "x.T.contiguous()": lambda x: x.T.contiguous(),
        },
        value="x.T",
        label="`x = torch.arange(12).reshape(3, 4)`, then",
    )
    operation
    return (operation,)


@app.cell
def _(as_heatmap, mo, operation, torch):
    x_storage = torch.arange(12).reshape(3, 4)
    try:
        _result = operation.value(x_storage)
    except RuntimeError as error:
        _panel = mo.callout(
            mo.md(
                f"""
                `RuntimeError: {error}`

                `view` refuses to guess. It only ever reinterprets the strides it was
                given, and no stride pattern reads a transposed 3x4 as a 2x6, so it sends
                you to `reshape`, which is allowed to copy when it has to.
                """
            ),
            kind="danger",
        )
    else:
        _same_storage = _result.untyped_storage().data_ptr() == x_storage.untyped_storage().data_ptr()
        _panel = mo.vstack(
            [
                as_heatmap(_result, "the tensor you get", cell=48),
                mo.md(
                    "| shape | stride | storage offset | contiguous | same storage as `x` |\n"
                    "| --- | --- | --- | --- | --- |\n"
                    f"| `{tuple(_result.shape)}` | `{_result.stride()}` | {_result.storage_offset()} "
                    f"| {'yes' if _result.is_contiguous() else '**no**'} "
                    f"| {'yes — a view' if _same_storage else '**no — it copied**'} |"
                ),
            ]
        )

    mo.vstack(
        [
            as_heatmap(x_storage.flatten(), "storage: the same twelve numbers, always", cell=44),
            _panel,
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `x[:1].expand(3, 4)` is the one to sit with: stride `(0, 1)`, so stepping along rows
    steps zero elements and all three rows read the same four numbers. Broadcasting is
    implemented exactly this way — no memory is allocated for the repeated dimension,
    which is why broadcasting a large tensor against a small one costs nothing.

    The sharing is real, and it cuts: `e = x[:1].expand(3, 4)` then `e[0, 0] = 99` puts
    99 in all three rows, because there is only one 99 to put. Writing to the whole thing
    at once is refused outright — `e.add_(1)` raises *more than one element of the
    written-to tensor refers to a single memory location* — so PyTorch catches the
    ambiguous case and lets the surprising one through.
    """)
    return


if __name__ == "__main__":
    app.run()
