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
    *PyTorch basics, 2 of 8 — before this: [Quickstart](01-quickstart.py) · after:
    [Datasets & DataLoaders](03-datasets-and-dataloaders.py)*

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
    each picture is what the object *is*: its shape, its dtype, and its **stride**.
    """)
    return


@app.cell
def _():
    import numpy as np
    import torch

    return np, torch


@app.cell(hide_code=True)
def _():
    # The viewing vocabulary is shared with the sibling notebooks: one module, evolved in
    # place, never forked into a file. What show() draws, and the color policy every
    # notebook inherits, live in _viz.py.
    import altair as alt
    import pandas as pd
    from _viz import INK_DARK, INK_LIGHT, RAMP, show

    return INK_DARK, INK_LIGHT, RAMP, alt, pd, show


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initializing a tensor

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
def _(show, x_data):
    show(x_data, "x_data")
    return


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
    return np_array, x_np


@app.cell(hide_code=True)
def _(mo, np_array, show, x_np):
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
def _(torch, x_data):
    x_ones = torch.ones_like(x_data)  # retains the properties of x_data

    x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data
    return x_ones, x_rand


@app.cell(hide_code=True)
def _(mo, show, x_data, x_ones, x_rand):
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
def _(torch):
    shape = (2, 3)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)
    return ones_tensor, rand_tensor, zeros_tensor


@app.cell(hide_code=True)
def _(mo, ones_tensor, rand_tensor, show, zeros_tensor):
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
    Six numbers can be read as numbers. Four thousand cannot, and do not need to be: the
    left picture is the `torch.rand` from the cell above at image scale, 4,096 numbers
    drawn instead of listed.

    That is not an analogy. An image *is* a tensor — one number per pixel, brightness as
    magnitude — so drawing a tensor as an image shows it as what it already is, and every
    image the later notebooks classify arrives as exactly this.

    The right picture holds the same 4,096 slots, each storing its own position: `arange`
    counts them off, `reshape` folds the count into 64 rows. It comes out a top-to-bottom
    ramp because consecutive positions sit side by side within a row and each new row
    starts 64 later — the first look at a fact worth holding on to: the grid is one flat
    run of memory, laid into rows. Numbers that trace their own
    whereabouts also return, in the indexing explorer and again in the notation section.
    """)
    return


@app.cell(hide_code=True)
def _(mo, torch):
    _noise = torch.rand(64, 64)
    _counted = torch.arange(4096).reshape(64, 64)
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
                    mo.image(_counted, width=150, vmin=0, vmax=4095, rounded=True),
                    mo.md("<small>`torch.arange(4096).reshape(64, 64)` — each number is its own position</small>"),
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
    ## Attributes of a tensor

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
    read: `(4, 1)` means that stepping to the next row moves four positions through memory,
    and stepping to the next column moves one.

    And `(4, 1)` is not extra information — yet. A fresh tensor's stride is computed from
    its shape: the last axis steps 1, and each axis to its left steps the product of the
    sizes to its right, which for `(3, 4)` gives `(4, 1)`. Stride begins carrying news of
    its own the moment it disagrees with what the shape predicts, and those disagreements
    are where this notebook is headed. Hold on to it. It is printed under every picture
    from here on, and a later section is about nothing else.
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
    ## Operations on tensors

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
    ## The notation

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
    four-dimensional batch — a batch this section ends by building, because on a matrix
    that claim cannot even be seen. And `tensor_2[:, 1] = 0` puts an indexed expression on the
    *left* of an assignment, which writes into the tensor rather than reading from it —
    which is why `tensor_2` ends up with a column of zeros in it.
    """)
    return


@app.cell(hide_code=True)
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
    give you a column standing upright. It does not, however, and the below table of
    entries shows why: the integer `0`, means select one entry of that dimension, and
    thereby made dimension 1 disappear. What comes back has shape `(4,)` — four numbers
    in *one* dimension.

    A one-dimensional tensor has no orientation. Neither upright nor flat; these words only
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

    Two of the captions also say **not contiguous**, the flag's first appearance. A tensor
    is contiguous when reading it in order walks its storage front to back, one step per
    element, no gaps — equivalently, when its stride is still the one its shape predicts.
    `tensor_2[0]`, stride `(1,)`, is that definition made visible. The
    columns break it — both land on slots 0, 4, 8, 12, skipping three each time. Nothing
    about them is wrong; the flag describes how a tensor reads, not what it holds. It
    starts to cost only when an operation needs the elements as one unbroken run of
    memory, and which operations those are is the storage section's business below.

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
def _(INK_DARK, INK_LIGHT, RAMP, alt, mo, pd, show, slicing, torch):
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
    # The gaps between squares stay transparent (band padding, not strokes) and the fills
    # come from the shared ramp, so the picture obeys the reader's theme like show() does.
    _position = {
        "x": alt.X("col:O", axis=None, scale=alt.Scale(paddingInner=0.06)),
        "y": alt.Y("row:O", axis=None, scale=alt.Scale(paddingInner=0.06)),
    }
    _grid = (
        alt.Chart(_cells)
        .mark_rect()
        .encode(
            **_position,
            color=alt.Color(
                "picked:N",
                scale=alt.Scale(domain=[False, True], range=[RAMP[0], RAMP[3]]),
                legend=None,
            ),
            tooltip=[alt.Tooltip("value:Q", title="value"), "row:O", "col:O"],
        )
        + alt.Chart(_cells)
        .mark_text(fontSize=13, fontWeight=500)
        .encode(
            **_position,
            text=alt.Text("value:Q"),
            color=alt.condition("datum.picked", alt.value(INK_LIGHT), alt.value(INK_DARK)),
        )
    ).properties(width=8 * 46, height=6 * 46)

    mo.hstack(
        [
            mo.vstack(
                [_grid, mo.md("<small>`t`, with the selected elements filled in</small>")],
                align="center",
                gap=0.2,
            ),
            _panel,
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
    ### The last column, whatever the rank

    One promise made above cannot be tested on a matrix: that `...` binds the entries you
    wrote to the *last* dimensions, whatever the rank. `tensor_2[..., -1]` and
    `tensor_2[:, -1]` land on the same four numbers, because with two dimensions, counting
    from the left and counting from the right meet in the middle.

    They part at four dimensions — which is also where reading a tensor by eye gives out,
    so the demonstration needs a tensor built to be read. In this one, every value spells
    its own address, one digit per axis: `1234` lives at batch 1, channel 2, row 3,
    column 4. That is the axis order every batch of images in the later notebooks arrives
    in. Slice it however you like; the result identifies itself, because whichever digit
    has stopped changing is the axis you pinned.
    """)
    return


@app.cell
def _(torch):
    # Four vectors, each parked in its own axis by the `None` of the table above. How the
    # sum spreads them into four dimensions is broadcasting, a section of its own further
    # down; nothing here depends on it beyond trusting the digits.
    digits = (
        torch.arange(2)[:, None, None, None] * 1000  # batch   -> thousands digit
        + torch.arange(3)[:, None, None] * 100  # channel -> hundreds digit
        + torch.arange(4)[:, None] * 10  # row     -> tens digit
        + torch.arange(5)  # col     -> ones digit
    )
    return (digits,)


@app.cell(hide_code=True)
def _(digits, mo, show):
    def _planes(sliced):
        return mo.hstack(
            [show(plane, f"batch {i}", cell=44) for i, plane in enumerate(sliced)],
            justify="start",
            gap=1.5,
            wrap=True,
        )

    def _head(expr, sliced, note):
        return mo.md(f"#### `{expr}` → `{tuple(sliced.shape)}` — {note}")

    mo.vstack(
        [
            show(digits[1, 2], "digits[1, 2] — thousands pinned at 1, hundreds at 2", cell=44),
            _head("digits[..., -1]", digits[..., -1], "ones digit pinned: the last *column*"),
            _planes(digits[..., -1]),
            _head("digits[:, -1]", digits[:, -1], "hundreds digit pinned: the last *channel*"),
            _planes(digits[:, -1]),
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read the digits, not the grid positions. In `digits[..., -1]` the ones digit is pinned
    at 4 and the other three still run: `...` reached past batch, channel and row to bind
    the `-1` to the last axis, and the same spelling would have done the same at any rank.
    That is why library code leans on `...` — it cannot know how many batch dimensions a
    caller stacked in front. `digits[:, -1]` pinned the hundreds instead: the same `-1`,
    counted from the left, a different axis entirely. So use `:` to address an axis by its
    position from the front, and `...` when the axis you mean is defined by being last.

    And nothing was copied. Every panel above is a re-reading of the same 120 integers,
    and the column panels are flagged **not contiguous** for the reason the column of
    `tensor_2` was: their walk through the storage skips. What a view actually is, and
    what the skipping eventually costs, is the next section's subject.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What a tensor actually is

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
def _(INK_DARK, INK_LIGHT, RAMP, alt, mo, operation, pd, show, torch):
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
                    scale=alt.Scale(range=[RAMP[0], RAMP[-1]], domain=[0, base.numel() - 1]),
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
                    f"datum.order > {0.73 * (base.numel() - 1):.2f}", alt.value(INK_LIGHT), alt.value(INK_DARK)
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
def _(tensor_2, torch):
    t1 = torch.cat([tensor_2, tensor_2, tensor_2], dim=1)
    return (t1,)


@app.cell(hide_code=True)
def _(mo, show, t1, tensor_2):
    mo.vstack([show(tensor_2, "tensor_2"), show(t1, "torch.cat([tensor_2] * 3, dim=1)", cell=38)], gap=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `cat` against `stack`

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
def _(tensor_2, torch):
    # This computes the matrix multiplication between two tensors.
    # y1, y2, y3 will have the same value
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
    return y1, y2, y3, z1, z2, z3


@app.cell(hide_code=True)
def _(mo, show, tensor_2, torch, y1, y2, y3, z1, z2, z3):
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
    ## Matrix multiplication you can drag

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
    ## Broadcasting, one dimension at a time

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
    ### Where the matrix comes from

    The table says what shape comes out. It does not say how two vectors can add up to a
    matrix, and that is worth seeing once, because the word "stretching" suggests the wrong
    thing.

    `col` is five numbers with shape `(5, 1)` — the `None` from the indexing table, placed
    after a `:`, put the second dimension there. `row` is six numbers with shape `(6,)`.
    Aligned from the right, `1` meets `6` and a supplied `1` meets `5`, so both are read as
    `(5, 6)`. Here is what that reading looks like, with the stride under each picture.
    """)
    return


@app.cell(hide_code=True)
def _(mo, show, torch):
    _col = torch.arange(0, 50, 10)[:, None]
    _row = torch.arange(6)
    _col_read, _row_read = _col.expand(5, 6), _row.expand(5, 6)
    _sum = _col + _row

    def _shares(a, b):
        return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()

    mo.vstack(
        [
            mo.hstack([show(_col, "col"), show(_row, "row")], justify="start", align="center", gap=2),
            mo.md("### `+` reads them as"),
            mo.hstack(
                [
                    show(_col_read, "col, stretched along dimension 1"),
                    show(_row_read, "row, stretched along dimension 0"),
                ],
                justify="start",
                align="center",
                gap=2,
                wrap=True,
            ),
            mo.md("### and allocates"),
            show(_sum, "col + row"),
            mo.md(
                "| | elements in storage | same storage as the vector it came from |\n"
                "| --- | --- | --- |\n"
                f"| `col.expand(5, 6)` | {_col_read.untyped_storage().nbytes() // _col_read.element_size()} "
                f"| {'yes — a view' if _shares(_col_read, _col) else 'no'} |\n"
                f"| `row.expand(5, 6)` | {_row_read.untyped_storage().nbytes() // _row_read.element_size()} "
                f"| {'yes — a view' if _shares(_row_read, _row) else 'no'} |\n"
                f"| `col + row` | {_sum.untyped_storage().nbytes() // _sum.element_size()} "
                f"| {'yes' if _shares(_sum, _col) or _shares(_sum, _row) else '**no — new storage**'} |"
            ),
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read, not rewritten. The middle row shows both operands as `+` sees them, and the
    strides say everything: `(1, 0)` and `(0, 1)`. A `0` on the stretched axis means that
    stepping along it goes nowhere in memory. Every row of `row` is the same six numbers
    being read again; every column of `col` is the same five. Neither has been copied — the
    table shows each still holding its five or six elements and still sharing storage with
    the vector it came from — and neither is contiguous, which the captions say. This is
    `x[:1].expand(3, 4)` from the storage strip, met where it earns its keep.

    The result is the first thing allocated. Thirty new numbers, stride `(6, 1)`, storage of
    its own. `+` did what it always does — one addition per position — to operands that had
    been reinterpreted before it ran. Broadcasting changed what was added, not how. And `+`
    is no more matrix addition here than it is anywhere else: when two shapes already match,
    "matrix addition" is only the name for the same elementwise loop.

    It also settles a debt: the digit tensor of the notation section was built by exactly
    this machinery — four vectors, each parked in its own axis by `None`, spread across one
    another into four dimensions.
    """)
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
    return agg, agg_item


@app.cell(hide_code=True)
def _(agg, agg_item, mo, show, tensor_2):
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

    The bridge is not a feature. A tensor is a shape, a stride and an offset over a run of
    memory, so a NumPy array and a tensor can be two descriptions of one run, and writing
    through either is writing to the same bytes. Nothing was copied and nothing was
    synchronized.

    `x.T` is free because it swaps two strides. `view` refuses where `reshape` copies
    because no stride pattern reads a transposed matrix as a flat one. A slice is a view
    and a boolean mask is a gather. `expand` costs nothing because a stride of zero reads
    the same memory repeatedly.

    Three things return later. **dtype** — `Optimization` is where
    16-bit starts paying and the `eps` column starts mattering. **Contiguity** — the moment
    training goes to more than one GPU, the collectives that exchange gradients want a
    linear buffer, and the copy that makes one is not free. And **`torch.stack`**, which is
    the next notebook's whole job: a list of `(1, 28, 28)` samples going in, one
    `(64, 1, 28, 28)` batch coming out.
    """)
    return


if __name__ == "__main__":
    app.run()
