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
    *PyTorch basics, 5 of 8 — before this: [Transforms](04-transforms.py) · after:
    [Autograd](06-autograd.py)*

    # Build the Neural Network
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Today's target** — run the notebook; consuming it means having built an `nn.Module` and
    > followed one forward pass through it, layer by layer, shape by shape.
    >
    > **Marc's depth line** — part of the base Marc named: the forward, the backward, a loss
    > computed with a simple model — the model itself stays a black box for training. Half a day
    > here is fine.
    >
    > **Stop-line** — done means: ran it, could explain to Marc what `forward` runs and where the
    > parameters live, questions captured — close it.
    >
    > **Capture** — `scripts/q "your question"` appends it to Friday's file for Marc.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Every layer in PyTorch is a **module** — a subclass of
    [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) — and a neural
    network is itself a module built from other modules. That one recursive idea is the whole
    [torch.nn](https://pytorch.org/docs/stable/nn.html) namespace: it provides the building
    blocks, and nesting them is how arbitrarily complex architectures stay one object that can
    be printed, moved and saved whole.

    This notebook builds such a module to classify FashionMNIST images, then follows one forward
    pass through it, layer by layer, shape by shape.
    """)
    return


@app.cell
def _():
    import torch
    from torch import nn

    return nn, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Get a device for training

    We want to train on an
    [accelerator](https://pytorch.org/docs/stable/torch.html#accelerators) when one is present —
    CUDA here; the same API covers Apple's Metal Performance Shaders (MPS), Meta's Training and
    Inference Accelerator (MTIA) and Intel's GPUs (XPU). Otherwise we fall back to the CPU.
    """)
    return


@app.cell
def _(torch):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define the class

    We define our neural network by subclassing `nn.Module`: the layers are initialized in
    `__init__`, and what the module does to its input lives in the `forward` method.
    """)
    return


@app.cell
def _(nn):
    class NeuralNetwork(nn.Module):
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

    return (NeuralNetwork,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We create an instance of `NeuralNetwork`, move it to the `device`, and print its structure.
    """)
    return


@app.cell
def _(NeuralNetwork, device):
    model = NeuralNetwork().to(device)
    print(model)
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Two better pictures of the same model

    `print(model)` gives the constructor arguments back. marimo formats an `nn.Module`
    itself, which adds what the constructor does not say: how many parameters each layer
    holds, what dtype and device they are on, and whether they are trainable. Fold the
    stack open.

    Underneath, torchview traces an actual forward pass and draws what happened. The two
    pictures answer different questions. The tree is what the model *contains* — nesting,
    parameter counts, devices — and it is built by walking attributes, so it would look
    the same if `forward` never called half of them. The graph is what the data *did*, and
    its edges carry the shapes that actually flowed.

    For this model the two agree, because `forward` runs the container in order. They stop
    agreeing the moment a `forward` does anything else: a skip connection, a branch, a
    layer applied twice, a layer defined and never called. Only the graph can show that,
    which is why it is the picture worth drawing for someone else's model.
    """)
    return


@app.cell
def _(model):
    model
    return


@app.cell(hide_code=True)
def _(device, mo, model):
    from torchview import draw_graph

    _graph = draw_graph(model, input_size=(1, 1, 28, 28), device=device, expand_nested=True, depth=3)
    # A transparent graph is black-on-black in a dark editor theme; graphviz has no
    # notion of the surrounding page, so the card carries its own background.
    _graph.visual_graph.graph_attr.update(bgcolor="white", rankdir="TB", margin="8")
    _svg = _graph.visual_graph.pipe(format="svg").decode()
    _card = "background:white;border-radius:8px;padding:8px;display:inline-block"
    mo.Html(f'<div style="{_card}">{_svg[_svg.index("<svg") :]}</div>')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To use the model, we pass it the input data. Calling the model executes its `forward` along
    with some [background
    operations](https://github.com/pytorch/pytorch/blob/270111b7b611d174967ed204776985cefca9c144/torch/nn/modules/module.py#L866)
    — hooks, chiefly — which is why `model.forward()` is never called directly, even though it
    would return the same tensor here.

    The call returns a 2-dimensional tensor: dim=0 indexes the batch, one row per input image,
    and dim=1 holds 10 raw scores, one per class — the **logits**. Passing them through an
    instance of the `nn.Softmax` module turns them into prediction probabilities.
    """)
    return


@app.cell
def _(device, model, nn, torch):
    X = torch.rand(1, 28, 28, device=device)
    logits = model(X)
    _pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = _pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")
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
    ## Model layers

    Let's break down the layers in the FashionMNIST model. To illustrate, we take a sample
    minibatch of 3 images of size 28x28 and watch what happens to it as it passes through the
    network.
    """)
    return


@app.cell
def _(torch):
    input_image = torch.rand(3, 28, 28)
    print(input_image.size())
    return (input_image,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## nn.Flatten

    We initialize the [nn.Flatten](https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html)
    layer to convert each 2D 28x28 image into a contiguous array of 784 pixel values; the
    minibatch dimension at dim=0 is maintained.
    """)
    return


@app.cell
def _(input_image, nn):
    flatten = nn.Flatten()
    flat_image = flatten(input_image)
    print(flat_image.size())
    return flat_image, flatten


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## nn.Linear

    The [linear layer](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) is a module
    that applies a linear transformation on the input using its stored weights and biases.
    """)
    return


@app.cell
def _(flat_image, nn):
    layer1 = nn.Linear(in_features=28 * 28, out_features=20)
    hidden1 = layer1(flat_image)
    print(hidden1.size())
    return hidden1, layer1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## nn.ReLU

    Non-linear activations are what create the complex mappings between the model's inputs and
    outputs. They are applied after linear transformations to introduce *nonlinearity*, helping
    neural networks learn a wide variety of phenomena.

    This model uses [nn.ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html)
    between its linear layers; other activations exist to introduce non-linearity the same way.
    """)
    return


@app.cell
def _(hidden1, nn):
    hidden1_relu = nn.ReLU()(hidden1)
    print(f"Before ReLU: {hidden1}\n\n")
    print(f"After ReLU: {hidden1_relu}")
    return (hidden1_relu,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What ReLU did, as a picture

    Sixty numbers printed twice is accurate and unreadable. The same two tensors drawn on
    one shared color scale: three rows, one per image in the minibatch, twenty columns,
    one per output unit. Blue is positive, orange negative, and zero is the neutral
    near-white between them.

    Every orange square in the first goes neutral in the second and nothing else moves. That
    is the entire operation — `max(x, 0)`, elementwise — and it is the whole reason the
    network is not equivalent to a single matrix. Stack two linear layers with nothing
    between them and their product is another linear layer; the model would have the depth
    but not the expressiveness.

    Watch the fraction below rather than the picture, though. A unit that lands on the
    zero side for *every* input in the batch contributes no gradient to its incoming
    weights on that step, and one that does so for every input in the dataset is dead for
    good. Half the activations being clipped is normal at initialization; most of them
    being clipped is a symptom.
    """)
    return


@app.cell(hide_code=True)
def _():
    import altair as alt
    import pandas as pd
    from _viz import ACCENT, BASE, POLARITY

    def activation_map(values, title, limit):
        """Draw a small 2D activation tensor on a fixed diverging scale."""
        cells = pd.DataFrame(
            [
                {"unit": column, "image": row, "value": float(value)}
                for row, line in enumerate(values.detach().tolist())
                for column, value in enumerate(line)
            ]
        )
        # The gaps between squares stay transparent (band padding, not strokes) and the
        # fills come from the shared diverging ramp, so the picture obeys the reader's
        # theme the way _viz.show() does.
        return (
            alt.Chart(cells)
            .mark_rect()
            .encode(
                x=alt.X("unit:O", title=None, axis=None, scale=alt.Scale(paddingInner=0.06)),
                y=alt.Y("image:O", title=None, axis=None, scale=alt.Scale(paddingInner=0.06)),
                color=alt.Color(
                    "value:Q",
                    scale=alt.Scale(range=POLARITY, domain=[-limit, limit]),
                    legend=None,
                ),
                tooltip=[alt.Tooltip("value:Q", format=".3f"), "image:O", "unit:O"],
            )
            .properties(width=20 * 26, height=3 * 26, title=title)
        )

    return ACCENT, BASE, activation_map, alt, pd


@app.cell(hide_code=True)
def _(activation_map, hidden1, hidden1_relu, mo):
    _limit = hidden1.abs().max().item()
    _zeroed = (hidden1_relu == 0).float().mean().item()
    mo.vstack(
        [
            activation_map(hidden1, "before ReLU", _limit),
            activation_map(hidden1_relu, "after ReLU", _limit),
            mo.md(f"**{_zeroed:.0%}** of the {hidden1_relu.numel()} activations are now exactly zero."),
        ],
        gap=0.6,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## nn.Sequential

    [nn.Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) is an
    ordered container of modules: data passes through all of them in the order they were given.
    Below, the `flatten` and `layer1` modules just walked through are chained with a fresh ReLU
    and a final Linear into a quick network — the same shape of pipeline `NeuralNetwork` wraps in
    a class.
    """)
    return


@app.cell
def _(flatten, input_image, layer1, nn):
    seq_modules = nn.Sequential(flatten, layer1, nn.ReLU(), nn.Linear(20, 10))
    seq_logits = seq_modules(input_image)
    print(seq_logits.size())
    return (seq_logits,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## nn.Softmax

    The last linear layer returns logits — raw values in (-∞, ∞), as met after the first forward
    pass above. The [nn.Softmax](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html)
    module rescales them to \[0, 1\], the model's predicted probability for each class. The `dim`
    parameter names the dimension along which the values must sum to 1: `dim=1`, across the
    classes of one image — not `dim=0`, across the batch.
    """)
    return


@app.cell
def _(nn, seq_logits):
    softmax = nn.Softmax(dim=1)
    _pred_probab = softmax(seq_logits)
    print(_pred_probab)
    print(f"row sums: {_pred_probab.sum(dim=1)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model parameters

    Many layers inside a neural network are *parameterized* — they hold weights and biases that
    are optimized during training. Subclassing `nn.Module` automatically tracks all fields
    defined inside your model object, and makes every parameter reachable through the model's
    `parameters()` or `named_parameters()` methods.

    Here we iterate over each parameter of `model` and print its name, size and a preview of its
    values.
    """)
    return


@app.cell
def _(model):
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The parameters as a table, and where their values came from

    The loop above prints six blocks of numbers. The same six as a table are easier to
    total, and the totals are the ones that decide whether a model fits on a card.

    Then the question the tutorial does not ask: who chose those starting values? Nobody
    typed them, and they are not random in the loose sense. `nn.Linear` draws both weights
    and bias from a uniform distribution over ±1/√fan_in — for the first layer, fan_in is
    784, so the bound is 0.0357 and nothing lands outside it. The histogram is flat
    between the two rules and empty beyond them.

    That bound is not arbitrary. It keeps the variance of a layer's output close to the
    variance of its input, so a signal passing through several layers neither collapses to
    zero nor explodes — which is why a fresh classifier's loss starts near ln(10) = 2.303
    rather than somewhere unhelpful, and why initialization has its own literature.
    """)
    return


@app.cell(hide_code=True)
def _(mo, model):
    _rows = [
        {
            "parameter": name,
            "shape": str(tuple(parameter.shape)),
            "count": parameter.numel(),
            "MB": round(parameter.numel() * parameter.element_size() / 1024**2, 3),
            "trainable": parameter.requires_grad,
        }
        for name, parameter in model.named_parameters()
    ]
    _total = sum(row["count"] for row in _rows)
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(f"{_total:,}", label="parameters", bordered=True),
                    mo.stat(f"{sum(row['MB'] for row in _rows):.1f} MB", label="as float32", bordered=True),
                    mo.stat(
                        f"{max(_rows, key=lambda row: row['count'])['parameter']}", label="largest", bordered=True
                    ),
                ],
                justify="start",
                gap=1,
            ),
            mo.ui.table(_rows, selection=None),
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(ACCENT, BASE, alt, mo, model, pd, torch):
    import math

    _first = (
        next(parameter for name, parameter in model.named_parameters() if name.endswith("0.weight")).detach().cpu()
    )
    _bound = 1 / math.sqrt(_first.shape[1])
    # Binned here rather than by altair: handing the chart all 401,408 raw weights trips
    # altair's 5,000-row limit outside marimo and ships the lot to the browser inside it.
    # Sixty pre-counted bars are the same picture.
    _counts, _edges = torch.histogram(_first.flatten(), bins=60)
    _bins = pd.DataFrame({"start": _edges[:-1].tolist(), "end": _edges[1:].tolist(), "count": _counts.tolist()})
    _histogram = (
        alt.Chart(_bins)
        .mark_bar(color=BASE)
        .encode(
            x=alt.X("start:Q", bin="binned", title="weight value"),
            x2="end:Q",
            y=alt.Y("count:Q", title="count"),
        )
        .properties(width=460, height=200, title=f"{tuple(_first.shape)} weights, fresh from nn.Linear")
    )
    _limits = (
        alt.Chart(pd.DataFrame({"edge": [-_bound, _bound]}))
        .mark_rule(color=ACCENT, strokeDash=[6, 4], strokeWidth=2)
        .encode(x="edge:Q")
    )
    mo.vstack(
        [
            _histogram + _limits,
            mo.md(
                f"fan_in = {_first.shape[1]}, so the bound is ±{_bound:.4f}. "
                f"Measured: min {_first.min():.4f}, max {_first.max():.4f}, "
                f"standard deviation {_first.std():.4f} — against the bound/√3 = "
                f"{_bound / math.sqrt(3):.4f} a uniform distribution predicts."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### One number decides the size of this model

    The count in the table is a function of one choice, the hidden width — the 512 in the
    constructor. The arithmetic is short enough to write out. Then drag the underlined number
    in the sentence below.
    """)
    return


@app.cell
def _():
    def parameter_count(width):
        # 784 -> width -> width -> 10, each Linear carrying one bias per output.
        return 784 * width + width + width * width + width + width * 10 + 10

    return (parameter_count,)


@app.cell(hide_code=True)
def _(mo):
    from wigglystuff import TangleSlider

    hidden_width = mo.ui.anywidget(TangleSlider(amount=512, min_value=16, max_value=2048, step=16, digits=0))
    return (hidden_width,)


@app.cell(hide_code=True)
def _(ACCENT, BASE, alt, hidden_width, mo, parameter_count, pd):
    _width = int(hidden_width.amount)
    _sentence = mo.md(
        f"""
        With a hidden width of {hidden_width} units, this network holds
        **{parameter_count(_width):,}** parameters and occupies
        **{parameter_count(_width) * 4 / 1024**2:.1f} MB** in float32 — before the
        optimizer, which will want a copy or two of its own.
        """
    )
    _curve = pd.DataFrame({"width": range(16, 2049, 16)})
    _curve["parameters"] = [parameter_count(w) for w in _curve["width"]]
    _chart = (
        alt.Chart(_curve)
        .mark_line(color=BASE)
        .encode(x=alt.X("width:Q", title="hidden width"), y=alt.Y("parameters:Q", title="parameters"))
        .properties(width=420, height=200)
    )
    _here = (
        alt.Chart(pd.DataFrame({"width": [_width], "parameters": [parameter_count(_width)]}))
        .mark_point(size=120, filled=True, color=ACCENT)
        .encode(x="width:Q", y="parameters:Q")
    )
    mo.vstack([_sentence, _chart + _here])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The curve bends because the middle layer is `width x width` while the other two are
    linear in `width`. Below roughly 800 units the 784-wide input layer dominates and the
    model grows in a straight line; above it the square term takes over. Widening a
    network is cheap until it suddenly is not, and the crossover is set by the input size
    — which is the same arithmetic that makes hidden size the expensive dimension in a
    transformer.
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
    ## Where to go next

    - The [torch.nn API](https://pytorch.org/docs/stable/nn.html) is the shelf this notebook took
      five modules from. Skipped: convolution and pooling layers (the reason image models
      outperform this flattened one), `nn.Dropout` and the normalization layers, and the
      containers beyond `Sequential` — `nn.ModuleList`, `nn.ModuleDict` — for a `forward` that is
      not a straight line.
    - Every module also carries a mode: `model.train()` and `model.eval()` flip the behavior of
      layers like dropout and batch norm. Nothing in this model reacts to the switch yet; the
      optimization notebook calls both, in their places in the loop.
    - Next: [Autograd](06-autograd.py), where the `trainable` column in the table above stops
      being a flag — every parameter with `requires_grad=True` collects a gradient there, and the
      backward pass that computes it is the subject.
    """)
    return


if __name__ == "__main__":
    app.run()
