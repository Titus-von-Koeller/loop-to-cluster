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
# Sections under an "Explore" heading are additions. The upstream tutorial's prose and its
# sequence are untouched; its code cells were changed in one respect only, which is that
# they render their tensors instead of printing them.

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    Every tensor here is drawn rather than printed, so that what changes is visible. Under
    each picture is what the object *is*: its shape, its dtype, and its **stride** — the
    last of which turns out to be the whole story.

    The notebook builds toward one claim, and it is worth having in mind from the start: a
    tensor is not its numbers. It is a shape, a stride and an offset over one flat run of
    memory — and creating, indexing, joining, multiplying and sharing with NumPy are all
    consequences of that.
    """)
    return


@app.cell
def _():
    import altair as alt
    import numpy as np
    import pandas as pd
    import torch

    # Seeded so that a full run of the notebook always draws the same numbers. Note what
    # this does not buy: the generator is global state, invisible to the dependency graph,
    # and it is consumed in whatever order the graph runs. Re-run one random cell on its
    # own and it draws the *next* numbers in the sequence rather than its own again, so its
    # picture changes while every other picture stays put. Nothing here depends on a
    # particular value -- the lessons are about shape, stride and structure -- but it is
    # the reason a cell reading hidden state cannot be reactive.
    torch.manual_seed(0)
    return alt, np, pd, torch


@app.cell(hide_code=True)
def _(alt, mo, pd, torch):
    # One way of looking at a tensor, used by every cell in this notebook.
    #
    # Magnitude is carried by lightness and only by lightness, so the picture survives being
    # read by someone who cannot separate red from green, and survives being printed gray.
    # Hue carries sign and nothing else. There are no axes and no chart title: the numbers
    # are in the squares and the caption says what the object is.
    RAMP = ["#dbe7f7", "#a8c6ec", "#6b9ede", "#2a78d6", "#17457c"]
    POLARITY = ["#8f3413", "#d95926", "#eaa886", "#e8e8e6", "#93bae9", "#2a78d6", "#173f6e"]

    def show(tensor, title=None, cell=54, facts=True):
        """Render a small tensor as its own numbers, colored by magnitude."""
        values = torch.as_tensor(tensor).detach().cpu()
        grid = values.reshape(1, 1) if values.dim() == 0 else values if values.dim() == 2 else values.reshape(1, -1)
        numbers = [[float(v) for v in row] for row in grid.tolist()]
        signed = min(min(row) for row in numbers) < 0
        limit = max((max(abs(v) for v in row) for row in numbers), default=1.0) or 1.0
        digits = ".0f" if not values.dtype.is_floating_point else ".2f"

        frame = pd.DataFrame(
            [{"col": j, "row": i, "v": v} for i, row in enumerate(numbers) for j, v in enumerate(row)]
        )
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
                color=alt.condition(on_dark, alt.value("#ffffff"), alt.value("#15181d")),
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

    return (show,)


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
def _(show, torch):
    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)
    show(x_data, "x_data")
    return data, x_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **From a NumPy array**

    Tensors can be created from NumPy arrays (and vice versa - see `bridge-to-np-label`).
    """)
    return


@app.cell
def _(data, mo, np, show, torch):
    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)
    mo.hstack(
        [show(np_array, "np_array — a NumPy array"), show(x_np, "x_np — a tensor on the same memory")],
        justify="start",
        gap=2,
    )
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
def _(mo, show, torch, x_data):
    x_ones = torch.ones_like(x_data)  # retains the properties of x_data

    x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data

    mo.hstack(
        [show(x_data, "x_data"), show(x_ones, "ones_like(x_data)"), show(x_rand, "rand_like(x_data, dtype=float)")],
        justify="start",
        gap=2,
    )
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
def _(mo, show, torch):
    shape = (2, 3)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)

    mo.hstack(
        [
            show(rand_tensor, "torch.rand(shape)"),
            show(ones_tensor, "torch.ones(shape)"),
            show(zeros_tensor, "torch.zeros(shape)"),
        ],
        justify="start",
        gap=2,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Six numbers can be read as numbers. Four thousand cannot, and do not need to be:
    `torch.rand(64, 64)` is 4,096 of them, drawn instead of listed.

    That is not an analogy. An image *is* a tensor — one number per pixel, brightness as
    magnitude — so drawing a tensor as an image shows it as what it already is, and every
    image the later notebooks classify arrives as exactly this.
    """)
    return


@app.cell(hide_code=True)
def _(mo, torch):
    _noise = torch.rand(64, 64)
    _gradient = torch.linspace(0, 1, 64).expand(64, 64)
    mo.hstack(
        [
            mo.vstack(
                [
                    mo.image(_noise, width=150, vmin=0, vmax=1, rounded=True),
                    mo.md("<small>`torch.rand(64, 64)` — 4,096 numbers</small>"),
                ],
                align="center",
                gap=0.2,
            ),
            mo.vstack(
                [
                    mo.image(_gradient, width=150, vmin=0, vmax=1, rounded=True),
                    mo.md("<small>`torch.linspace(0, 1, 64).expand(64, 64)`</small>"),
                ],
                align="center",
                gap=0.2,
            ),
        ],
        justify="start",
        gap=2,
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
    # Attributes of a Tensor

    Tensor attributes describe their shape, datatype, and the device on which they are stored.
    """)
    return


@app.cell(hide_code=True)
def _(mo, show, torch):
    tensor = torch.rand(3, 4)

    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(str(tuple(tensor.shape)), label="shape", caption="3 rows, 4 columns", bordered=True),
                    mo.stat(
                        str(tensor.dtype).removeprefix("torch."),
                        label="dtype",
                        caption=f"{tensor.element_size()} bytes each",
                        bordered=True,
                    ),
                    mo.stat(str(tensor.device), label="device", caption="where the memory lives", bordered=True),
                    mo.stat(
                        str(tensor.stride()), label="stride", caption="steps to the next row, column", bordered=True
                    ),
                ],
                justify="start",
                gap=1,
                wrap=True,
            ),
            show(tensor, facts=False),
        ],
        gap=1,
    )
    return (tensor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Shape, dtype and device say what a tensor holds and where. **Stride** says how it is
    read, and it explains the other three: `(4, 1)` means that stepping to the next row
    moves four positions through memory, and stepping to the next column moves one. Hold on
    to it. It is printed under every picture from here on, and a later section is about
    nothing else.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Explore — `dtype` is three decisions, not one

    `torch.float32` is the default, and it is worth stopping on, because the choice of
    dtype decides how much memory a model needs, how fast it trains, and what it can
    represent — and the two 16-bit formats in the table differ from each other more than
    either differs from `float32`.

    A floating point number is a sign, an exponent and a mantissa. The exponent sets the
    *range*, the mantissa sets the *precision*, and 16 bits has to be split between them —
    which the two formats do differently, and that is the whole story.

    Type a number and watch what each format does to it. The columns say the rest.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    stored_value = mo.ui.text("0.1", label="store this number as")
    stored_value
    return (stored_value,)


@app.cell(hide_code=True)
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
                f"The classifier built in **Build Model** holds 669,706 parameters: "
                f"**{669706 * 4 / 1024**2:.1f} MB** in float32, **{669706 * 2 / 1024**2:.1f} MB** "
                "in either 16-bit format.\n\n"
                "Halving the parameters is the *least* of what mixed precision does, though. "
                "Under `torch.autocast` the parameters stay float32; what moves to 16-bit is the "
                "activations and the inputs to each matrix multiply — which is where both the "
                "memory of a large batch and the speedup live, since the tensor cores that make "
                "16-bit fast only accept 16-bit. That is what *mixed* names: two precisions in "
                "one step, chosen per operation."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "How the two 16-bit formats spend their bits": mo.md(r"""
            - **float16** spends its bits on mantissa. Finer steps than bfloat16, but the
              largest number it can hold is 65,504 and the smallest normal one is 6.1e-05.
              Gradients routinely fall below that and become zero — which is what a gradient
              scaler exists to prevent, by multiplying the loss up before the backward pass
              and dividing it out after.
            - **bfloat16** spends them on exponent — the same eight exponent bits as float32.
              So it starts underflowing in the same place float32 does (`tiny` is 1.18e-38
              for both), and a gradient that survives in float32 survives here, which is why
              bfloat16 needs no scaler. Its largest value differs from float32's only in the
              last mantissa digits, 3.39e38 against 3.40e38. The price is precision: its
              steps are eight times coarser than float16's, exactly eight, as the `eps`
              column shows.
            """)
        }
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Explore — the notation

    Indexing a tensor is four separate pieces of notation wearing one pair of brackets. The
    whole grammar is small, so here it is, and then the expressions read as arithmetic
    rather than as incantation.

    ### The brackets

    **`t[a, b, c]` — one pair of brackets, one entry per dimension, left to right.**

    A list of lists needs `rows[1][3]`: two lookups, because the outer list hands you an
    inner list which you then index again. A tensor takes `t[1, 3]`: a single lookup that
    the shape and stride resolve by arithmetic. There is no inner object to fetch.

    **Dimensions you leave off the end are untouched** — the missing entries are `:`. For a
    2-D `t`, `t[1]` and `t[1, :]` are the same tensor. And `t[:]` is the entire tensor: the
    one entry says "all of dimension 0", and the unwritten second says "all of dimension 1".

    ### The four things an entry can be

    What matters about each is not only what it selects, but what it does to *that
    dimension of the shape*:

    | entry | example | selects | effect on that dimension |
    | --- | --- | --- | --- |
    | an integer | `t[3]` | one position | **it disappears** |
    | a slice | `t[1:5]` | a range | stays, usually shorter |
    | `:` | `t[:]` | all of it | stays, unchanged |
    | `None` | `t[None]` | nothing at all | a **new** dimension of size 1 is inserted here |

    Two spellings sit on top of that. `...` stands for as many `:` as it takes to reach the
    entries you did write, so `t[..., -1]` binds to the *last* dimension whatever the rank.
    And a slice is `start:stop:step` with any part omissible: `stop` is excluded, `2:` runs
    to the end, `:3` stops before 3, `::2` takes every second, and negative numbers count
    from the end, so `t[-1]` is the last row.

    That is the entire syntax. Everything after this is a consequence of the table.

    The cell that builds `tensor_2` puts four of them to work. `tensor_2[0]` has an integer
    for its only entry, so dimension 0 disappears and four numbers come back.
    `tensor_2[:, 0]` keeps every row and drops the column dimension. `tensor_2[..., -1]`
    takes the *last* column, written so that it would still mean the last column on a
    four-dimensional batch. And `tensor_2[:, 1] = 0` puts an indexed expression on the
    *left* of an assignment, which writes into the tensor rather than reading from it —
    which is why `tensor_2` ends up with a column of zeros in it.
    """)
    return


@app.cell
def _(mo, show, torch):
    tensor_2 = torch.rand(4, 4)
    # Snapshotted before the write, because the three reads below are views: once
    # `tensor_2[:, 1] = 0` runs they show the zeros too, and the before/after would be
    # two pictures of the same tensor.
    _before = tensor_2.clone()
    _first_row = tensor_2[0].clone()
    _first_column = tensor_2[:, 0].clone()
    _last_column = tensor_2[..., -1].clone()
    tensor_2[:, 1] = 0

    mo.vstack(
        [
            mo.hstack(
                [
                    show(_before, "tensor_2"),
                    show(_first_row, "tensor_2[0]"),
                    show(_first_column, "tensor_2[:, 0]"),
                    show(_last_column, "tensor_2[..., -1]"),
                ],
                justify="start",
                align="center",
                gap=2,
                wrap=True,
            ),
            show(tensor_2, "tensor_2, after `tensor_2[:, 1] = 0`"),
        ],
        gap=1,
    )
    return (tensor_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The first consequence: the column that is not a column

    `tensor_2[:, 0]` reads aloud as "every row, column zero", so it feels like it should
    give you a column standing upright. It does not, and the table of entries says why: the
    integer `0` made dimension 1 disappear. What comes back has shape `(4,)` — four numbers
    in *one* dimension.

    A one-dimensional tensor has no orientation. Neither upright nor flat; the words only
    mean something once there are two dimensions to tell apart. `tensor_2[0]`, the first
    *row*, comes back with the identical shape `(4,)`. The row and the column are the same
    kind of object, which is why they are drawn the same way.

    Keep the dimension and you get the upright thing you pictured: `tensor_2[:, 0:1]` is a
    slice, so dimension 1 survives with length 1, and the shape is `(4, 1)`.

    Watch the strides, though. That is where the difference went.
    """)
    return


@app.cell(hide_code=True)
def _(mo, show, tensor_2):
    mo.hstack(
        [
            show(tensor_2[0], "tensor_2[0]"),
            show(tensor_2[0:1], "tensor_2[0:1]"),
            show(tensor_2[:, 0], "tensor_2[:, 0]"),
            show(tensor_2[:, 0:1], "tensor_2[:, 0:1]"),
        ],
        justify="start",
        align="center",
        gap=2,
        wrap=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Four tensors holding four numbers each, all taken from `tensor_2` as it now stands,
    zeroed column and all. The captions do the rest of the work.

    `tensor_2[0]` has stride `(1,)`: its four numbers lie side by side in memory.
    `tensor_2[:, 0]` has stride `(4,)`: it reads number 0, then 4, then 8, then 12, stepping
    a whole row each time — which is what "going down a column" *is* once the rows are laid
    end to end. The column-ness did not vanish when the dimension did. It moved into the
    stride.

    And the two that kept their second dimension differ from their flattened partners only
    there: `(1, 4)` against `(4,)`, `(4, 1)` against `(4,)`, the same four numbers in the
    same memory either way. This is the distinction worth carrying, because it is where a
    batch dimension quietly disappears and the error surfaces three lines later in a matrix
    multiply that wanted two dimensions and got one.

    Now pick an expression. The highlighted cells are what it selects.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    slicing = mo.ui.dropdown(
        options={
            "t[0] — one row, and the row dimension is gone": lambda t: t[0],
            "t[0:1] — the same eight numbers, still a matrix": lambda t: t[0:1],
            "t[:] — every entry omitted: the whole tensor, as a view": lambda t: t[:],
            "t[:, 0] — every row, column zero, and now one-dimensional": lambda t: t[:, 0],
            "t[:, 0:1] — the same numbers, kept upright": lambda t: t[:, 0:1],
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
def _(alt, mo, pd, show, slicing, torch):
    t2_sliced = torch.arange(48).reshape(6, 8)
    try:
        _result = slicing.value(t2_sliced)
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
        _shares = _result.untyped_storage().data_ptr() == t2_sliced.untyped_storage().data_ptr()
        _facts = mo.md(
            f"`{tuple(_result.shape)}` — {_result.dim()} "
            f"{'dimension' if _result.dim() == 1 else 'dimensions'}, {_result.numel()} elements, "
            f"and {'a **view**: it shares storage with `t`' if _shares else 'a **copy**: new storage'}."
        )
        _panel = mo.vstack([_facts, show(_result, "what you get back", cell=44) if _result.dim() <= 2 else mo.md("")])

    # Matched by value rather than by position, which is exact here only because
    # `torch.arange` gives every element a distinct one. It would mis-highlight duplicates.
    _cells = pd.DataFrame(
        [
            {"row": i, "col": j, "value": v, "picked": v in _picked}
            for i, _row in enumerate(t2_sliced.tolist())
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
    A slice is a view — the same storage read with different strides, which is why
    `tensor_2[:, 1] = 0` changed `tensor_2` itself. The boolean mask and the list of indices
    are the two that cost memory: neither can be expressed as a stride, so PyTorch gathers
    the elements into new storage.

    That makes the mask asymmetric, and this is the part worth remembering. *Read* it and
    you get a copy: `c = t[t % 7 == 0]` then `c[0] = -1` leaves `t` untouched. *Assign into*
    it and you write through: `t[t % 7 == 0] = -1` does change `t`, because that is not a
    read followed by a write — it is a single indexed assignment, and PyTorch scatters
    straight back into the original storage.

    `t[:]` sets the matching trap for anyone arriving from Python, where the same notation
    means the opposite. `lst[:]` is *the* shallow-copy idiom: `copy = lst[:]`, edit `copy`,
    and `lst` is untouched. On a tensor nothing is copied — you get a second view onto the
    same bytes, and editing it edits the original. `t.clone()` is the one that copies.

    Which leaves `t[:]` looking useless, since reading through it hands back what you
    already had. Its use is on the other side of the assignment: `w[:] = 0` fills the
    existing tensor in place, keeping its storage and its identity, where `w = 0` merely
    rebinds the name and abandons the tensor. Anything else pointing at that memory —
    another view, a NumPy array sharing it, a parameter held by a module — sees the first
    and misses the second.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Explore — what a tensor actually is

    Every slice in the previous section reported itself as a view or a copy, and the
    caption under every picture has been quietly printing a `stride` you were told to hold
    on to. Here is what both mean.

    A tensor is not its numbers. It is a *view* — a shape, a stride and an offset — onto one
    flat run of memory, and several tensors can describe the same run differently. That
    single sentence is what makes `x.T` free, what makes `reshape` sometimes copy, and, at
    the end of this notebook, what makes a NumPy array and a tensor able to be the same data
    under two names.

    Pick an operation. The strip is the storage: twelve numbers in the order they lie in
    memory, and those numbers never change. What moves is the shading, which marks the order
    this particular operation reads them in. The grid is what the result claims to be.

    - **stride** is how far to step, in elements, to move one position along each
      dimension. `x.T` does not move a single number: it swaps the two strides.
    - **contiguous** means the strides still walk the storage front to back. Transposing
      breaks that, and any operation needing a linear layout has to copy first.
    - **same storage** answers whether you got a view or a copy. Write into a view and
      the original changes with it.

    `x.T.view(2, 6)` is in the list on purpose: it is the error that the `x.T.reshape(2, 6)`
    beside it exists to avoid.
    """)
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(alt, mo, operation, pd, show, torch):
    import itertools

    def storage_strip(result, base, shares):
        """The twelve slots of `base`, shaded by the order `result` reads them in.

        This is the whole mechanism in one row: a view does not hold numbers, it holds a
        rule for walking someone else's memory, and the rule is the stride.
        """
        read_at = {}
        if shares:
            for rank, index in enumerate(itertools.product(*[range(s) for s in result.shape])):
                slot = result.storage_offset() + sum(i * s for i, s in zip(index, result.stride(), strict=True))
                read_at.setdefault(slot, rank)
        cells = pd.DataFrame(
            [
                {"slot": slot, "row": 0, "value": int(v), "order": read_at.get(slot)}
                for slot, v in enumerate(base.flatten())
            ]
        )
        # Both marks are placed on the same two band scales, so each number sits inside its
        # own square rather than at a pixel offset that happens to look right at one size.
        at = {
            "x": alt.X("slot:O", axis=None, scale=alt.Scale(paddingInner=0.06)),
            "y": alt.Y("row:O", axis=None),
        }
        squares = (
            alt.Chart(cells)
            .mark_rect()
            .encode(
                **at,
                # Pinned to every slot, not to the ranks this operation happens to use. Left to
                # the data extent, an operation reading four slots would paint rank 3 the
                # darkest -- shading that means something different per selection, and dark
                # text on a dark fill.
                color=alt.Color(
                    "order:Q",
                    scale=alt.Scale(range=["#dbe7f7", "#17457c"], domain=[0, base.numel() - 1]),
                    legend=None,
                ),
                tooltip=[alt.Tooltip("slot:O", title="storage slot"), alt.Tooltip("order:Q", title="read position")],
            )
        )
        labels = (
            alt.Chart(cells)
            .mark_text(fontSize=13, fontWeight=500)
            .encode(
                **at,
                text=alt.Text("value:Q", format=".0f"),
                color=alt.condition(
                    f"datum.order > {0.73 * (base.numel() - 1):.2f}", alt.value("#ffffff"), alt.value("#15181d")
                ),
            )
        )
        return (squares + labels).properties(width=44 * len(cells), height=48)

    x_storage = torch.arange(12).reshape(3, 4)
    _result = None
    _same_storage = False
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
                show(_result, "the tensor you get", cell=48),
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
            mo.md(
                "**storage** — the same twelve numbers, always, in the order they lie in memory"
                + (
                    ", shaded by the order the result reads them in"
                    if _same_storage
                    else ". This result copied its numbers out and indexes storage of its own."
                )
            ),
            storage_strip(_result, x_storage, _same_storage) if _result is not None else mo.md(""),
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Joining tensors** You can use `torch.cat` to concatenate a sequence of tensors along a given
    dimension. See also [torch.stack](https://pytorch.org/docs/stable/generated/torch.stack.html),
    another tensor joining operator that is subtly different from `torch.cat`.
    """)
    return


@app.cell
def _(mo, show, tensor_2, torch):
    t1 = torch.cat([tensor_2, tensor_2, tensor_2], dim=1)
    mo.vstack([show(tensor_2, "tensor_2"), show(t1, "torch.cat([tensor_2] * 3, dim=1)", cell=38)], gap=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Explore — `cat` against `stack`

    The difference is easy to see once the shapes are side by side, and it is worth ten
    seconds now, because it is the difference between a batch of 64 images and a single
    image with 64 channels.

    **`cat` joins along a dimension that already exists.** The inputs need to agree on
    every *other* dimension, the chosen one adds up, and the result has the same number of
    dimensions it started with.

    **`stack` creates a new dimension.** The inputs must have identical shapes, and the
    result has one dimension more than they did. `dim` says where the new axis goes.

    Two 2x3 tensors, every option:
    """)
    return


@app.cell(hide_code=True)
def _(mo, show, torch):
    _a = torch.arange(6).reshape(2, 3)
    _b = torch.arange(6, 12).reshape(2, 3)

    def _panel(result):
        # A 3-D result is drawn as the 2-D slices it is made of, which is what having a new
        # dimension actually looks like.
        if result.dim() == 2:
            return show(result, cell=44)
        return mo.hstack([show(plane, f"[{i}]", cell=44) for i, plane in enumerate(result)], justify="start", gap=1.5)

    _views = {
        f"cat, dim=0 → {tuple(torch.cat([_a, _b], 0).shape)}": _panel(torch.cat([_a, _b], 0)),
        f"cat, dim=1 → {tuple(torch.cat([_a, _b], 1).shape)}": _panel(torch.cat([_a, _b], 1)),
        f"stack, dim=0 → {tuple(torch.stack([_a, _b], 0).shape)}": _panel(torch.stack([_a, _b], 0)),
        f"stack, dim=1 → {tuple(torch.stack([_a, _b], 1).shape)}": _panel(torch.stack([_a, _b], 1)),
        f"stack, dim=2 → {tuple(torch.stack([_a, _b], 2).shape)}": _panel(torch.stack([_a, _b], 2)),
    }
    mo.vstack(
        [
            mo.hstack([show(_a, "a", cell=44), show(_b, "b", cell=44)], justify="start", gap=2),
            mo.ui.tabs(_views),
            mo.md(
                "The twelve numbers are numbered so you can follow them. Every tab holds all "
                "twelve; only the arrangement changes. `cat` keeps two dimensions and grows one "
                "of them, `stack` adds a third and leaves both originals intact inside it — which "
                "is why the stacked tabs are drawn as separate planes.\n\n"
                "The one to remember is `torch.stack(list_of_images)`: that is how a list of "
                "`(1, 28, 28)` samples becomes the `(64, 1, 28, 28)` batch a `DataLoader` hands "
                "you, and it is why every sample in a batch must have the same shape."
            ),
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Arithmetic operations**
    """)
    return


@app.cell
def _(mo, show, tensor_2, torch):
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

    mo.vstack(
        [
            mo.hstack(
                [
                    show(tensor_2, "tensor_2"),
                    show(y1, "y1 = tensor_2 @ tensor_2.T"),
                    show(z1, "z1 = tensor_2 * tensor_2"),
                ],
                justify="start",
                align="center",
                gap=2,
                wrap=True,
            ),
            mo.callout(
                mo.md(
                    "The three spellings of each really do agree: "
                    f"`y1 == y2 == y3` is **{bool(torch.equal(y1, y2) and torch.equal(y2, y3))}**, "
                    f"`z1 == z2 == z3` is **{bool(torch.equal(z1, z2) and torch.equal(z2, z3))}**. "
                    "The zeroed column of `tensor_2` survives into `z1` in the same place, because "
                    "element-wise multiplication never moves a number; in `y1` it is gone, because "
                    "every entry there is a dot product over the whole row."
                ),
                kind="neutral",
            ),
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Explore — matrix multiplication you can drag

    Drag horizontally inside a cell of `A` to change it; every panel recomputes.

    Two things are worth watching. `A @ A.T` stays symmetric whatever you do to `A`,
    because entry *(i, j)* is the dot product of rows *i* and *j* of `A`, and swapping *i*
    and *j* asks for the same dot product. And the two products differ in shape as well as
    in value: `A * A` is element-wise, same shape in and out, while `A @ A.T` contracts
    the three columns away and leaves 2x2. Nearly every shape error in a training script
    is that distinction, met without a picture.
    """)
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(editable, mo, show, torch):
    _a = torch.tensor(editable.value)
    mo.hstack(
        [
            show(_a, "A"),
            show(_a.T, "A.T"),
            show(_a @ _a.T, "A @ A.T  (2x2)"),
            show(_a * _a, "A * A  (2x3)"),
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

    Equal shapes multiply elementwise, which is the easy case. The rule underneath governs
    every other one. Shapes are lined up from the *right*; a missing dimension counts as 1,
    and a dimension of 1 is stretched to meet its partner. Anything else is an error.

    Type two shapes. The table shows the alignment PyTorch performs, and the error text
    when there isn't one — worth reading once deliberately, because it is the error you
    will meet with a batch dimension in the wrong place.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    left_shape = mo.ui.text("3, 1, 4", label="left")
    right_shape = mo.ui.text("2, 4", label="right")
    mo.hstack([left_shape, right_shape], justify="start", gap=1)
    return left_shape, right_shape


@app.cell(hide_code=True)
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
def _(mo, show, tensor_2):
    agg = tensor_2.sum()
    agg_item = agg.item()

    mo.hstack(
        [
            show(tensor_2, "tensor_2"),
            mo.md("### `.sum()`"),
            show(agg, "agg — a tensor with no dimensions"),
            mo.md(f"### `.item()`\n\n`{agg_item}`\n\n<small>a Python `{type(agg_item).__name__}`</small>"),
        ],
        justify="start",
        align="center",
        gap=1.5,
        wrap=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **In-place operations** Operations that store the result into the operand are called in-place.
    They are denoted by a `_` suffix. For example: `x.copy_(y)`, `x.t_()`, will change `x`.
    """)
    return


@app.cell
def _(mo, show, tensor_2):
    # Demonstrated on a copy. `tensor_2.add_(5)` would mutate a tensor five cells above
    # still read by four cells below, and marimo tracks reassignment rather than mutation,
    # so nothing downstream would be marked stale and every re-run would add another 5.
    _demo = tensor_2.clone()
    _before = _demo.clone()
    _demo.add_(5)

    mo.hstack(
        [show(_before, "demo"), mo.md("### `.add_(5)`"), show(_demo, "demo, after — same object")],
        justify="start",
        align="center",
        gap=2,
    )
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A change in the tensor reflects in the NumPy array.
    """)
    return


@app.cell
def _(mo, show, torch):
    t = torch.ones(5)
    n = t.numpy()
    # The mutation lives in the same cell that creates what it mutates. Were `t.add_(1)`
    # in a cell of its own it would not be idempotent -- marimo tracks reassignment, not
    # mutation, so re-running it alone would reach 3.0 while the cell that drew 1.0 stayed
    # on screen and was never marked stale.
    _before = (show(t, "t — a tensor"), show(n, "n = t.numpy() — the same five numbers"))
    t.add_(1)

    mo.vstack(
        [
            mo.hstack(list(_before), justify="start", gap=2),
            mo.md("### `t.add_(1)`"),
            mo.hstack([show(t, "t, after t.add_(1)"), show(n, "n, which nobody touched")], justify="start", gap=2),
            mo.md(
                "`n` was never assigned to and still changed. It is not a copy of `t`; it is a "
                "second label on the same storage, and `add_` wrote into that storage in place."
            ),
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # NumPy array to Tensor
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Changes in the NumPy array reflects in the tensor.
    """)
    return


@app.cell
def _(mo, np, show, torch):
    n_1 = np.ones(5)
    t_1 = torch.from_numpy(n_1)
    _before = show(t_1, "t_1, before")
    np.add(n_1, 1, out=n_1)

    mo.hstack(
        [_before, show(n_1, "n_1, written by NumPy"), show(t_1, "t_1, which followed")],
        justify="start",
        gap=2,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Where this leaves you

    The bridge is not a feature. It is the last consequence of the sentence at the top: a
    tensor is a shape, a stride and an offset over a run of memory, so a NumPy array and a
    tensor can be two descriptions of one run, and writing through either is writing to the
    same bytes. Nothing was copied and nothing was synchronized.

    The same sentence answers the rest of the notebook. `x.T` is free because it swaps two
    strides. `view` refuses where `reshape` copies because no stride pattern reads a
    transposed matrix as a flat one. A slice is a view and a boolean mask is a gather.
    `expand` costs nothing because a stride of zero reads the same memory repeatedly.

    Three things planted here are collected later. **dtype** — `Optimization` is where
    16-bit starts paying and the `eps` column starts mattering. **Contiguity** — the moment
    training goes to more than one GPU, the collectives that exchange gradients want a
    linear buffer, and the copy that makes one is not free. And **`torch.stack`**, which is
    the next notebook's whole job: a list of `(1, 28, 28)` samples going in, one
    `(64, 1, 28, 28)` batch coming out.
    """)
    return


if __name__ == "__main__":
    app.run()
