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


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [Learn the Basics](intro.html) \|\| [Quickstart](quickstart_tutorial.html) \|\|
    [Tensors](tensorqs_tutorial.html) \|\| [Datasets & DataLoaders](data_tutorial.html) \|\|
    [Transforms](transforms_tutorial.html) \|\| [Build Model](buildmodel_tutorial.html) \|\|
    [Autograd](autogradqs_tutorial.html) \|\| [Optimization](optimization_tutorial.html) \|\|
    **Save & Load Model**

    # Save and Load the Model

    In this section we will look at how to persist model state with saving, loading and running
    model predictions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Today's target** — run the notebook; consuming it means having saved a `state_dict`,
    > loaded it into a fresh model, and run a prediction from the reload.
    >
    > **Marc's depth line** — when Marc mapped what to really focus on — forward and backward,
    > loss, gradients, optimizer, dataloader — this one never came up. Shallow is enough.
    >
    > **Stop-line** — done means: ran it, could explain to Marc what a `state_dict` carries and
    > what it does not, questions captured — close it.
    >
    > **Capture** — `scripts/q "your question"` appends it to Friday's file for Marc.
    """)
    return


@app.cell
def _():
    import torch
    import torchvision.models as models

    return models, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Saving and Loading Model Weights

    PyTorch models store the learned parameters in an internal state dictionary, called
    `state_dict`. These can be persisted via the `torch.save` method:
    """)
    return


@app.cell
def _(models, torch):
    model = models.vgg16(weights="IMAGENET1K_V1")
    torch.save(model.state_dict(), "model_weights.pth")
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What 528 megabytes is made of

    That cell downloaded a real trained model and wrote it to disk, and both facts pass
    without comment. VGG16 is a useful thing to look at precisely because it is old enough
    to be badly proportioned: the tree below, and the table under it, show where its
    parameters actually live.

    Fold `features` open — thirteen convolutions, the part of the network everyone pictures
    when they think of VGG — and then look at what `classifier.0` alone holds. Roughly
    three quarters of the model is in one fully-connected layer, and it is there because
    flattening a 512x7x7 feature map into 25,088 inputs and mapping it to 4,096 outputs
    costs 25,088 x 4,096 weights. Every architecture since replaces that layer with
    pooling, and that single change is most of why a modern network with better accuracy
    is smaller than this one.
    """)
    return


@app.cell
def _(model):
    model
    return


@app.cell
def _(mo, model):
    _entries = model.state_dict()
    _total = sum(tensor.numel() for tensor in _entries.values())
    _rows = [
        {
            "entry": name,
            "shape": str(tuple(tensor.shape)),
            "parameters": tensor.numel(),
            "share": f"{100 * tensor.numel() / _total:.1f}%",
            "MB": round(tensor.numel() * tensor.element_size() / 1024**2, 2),
        }
        for name, tensor in sorted(_entries.items(), key=lambda item: -item[1].numel())
    ]
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(f"{_total:,}", label="parameters", bordered=True),
                    mo.stat(f"{sum(row['MB'] for row in _rows):.0f} MB", label="as float32", bordered=True),
                    mo.stat(f"{len(_rows)}", label="state_dict entries", bordered=True),
                    mo.stat(_rows[0]["share"], label=f"in {_rows[0]['entry']}", bordered=True),
                ],
                justify="start",
                gap=1,
                wrap=True,
            ),
            mo.ui.table(_rows, selection=None, page_size=8),
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To load model weights, you need to create an instance of the same model first, and then load
    the parameters using `load_state_dict()` method.

    In the code below, we set `weights_only=True` to limit the functions executed during unpickling
    to only those necessary for loading weights. Using `weights_only=True` is considered a best
    practice when loading weights.
    """)
    return


@app.cell
def _(models, torch):
    model_1 = models.vgg16()  # we do not specify ``weights``, i.e. create untrained model
    model_1.load_state_dict(torch.load("model_weights.pth", weights_only=True))
    model_1.eval()
    return (model_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > [!NOTE]
    > be sure to call `model.eval()` method before inferencing to set the dropout and batch
    > normalization layers to evaluation mode. Failing to do this will yield inconsistent inference
    > results.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Saving and Loading Models with Shapes

    When loading model weights, we needed to instantiate the model class first, because the class
    defines the structure of a network. We might want to save the structure of this class together
    with the model, in which case we can pass `model` (and not `model.state_dict()`) to the saving
    function:
    """)
    return


@app.cell
def _(model_1, torch):
    torch.save(model_1, "model.pth")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can then load the model as demonstrated below.

    As described in [Saving and loading
    torch.nn.Modules](https://pytorch.org/docs/main/notes/serialization.html#saving-and-loading-torch-nn-modules),
    saving `state_dict` is considered the best practice. However, below we use `weights_only=False`
    because this involves loading the model, which is a legacy use case for `torch.save`.
    """)
    return


@app.cell
def _(torch):
    model_2 = torch.load("model.pth", weights_only=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The two files, on disk

    The tutorial says saving a `state_dict` is best practice and that the other way
    "relies on the actual class definition to be available". Both files now exist, so the
    claim is checkable.

    They are the same size to within a rounding error. Saving the model rather than its
    weights does not save the code — it pickles a *reference* to `torchvision.models.VGG`
    and the tensors separately, so you get a file that is no smaller, no more portable,
    and now breaks if the class it names moves or its signature changes. There is no
    trade-off being made here; the second form is simply worse, and it survives because
    it is one character shorter to write.

    `weights_only` is the other half. `torch.load` unpickles, and unpickling arbitrary
    data runs arbitrary code — a checkpoint downloaded from a model hub is an executable
    if you let it be. In torch 2.13 the parameter defaults to `None`, which resolves to
    *true*: loading the pickled model above without passing `weights_only=False`
    explicitly raises `UnpicklingError` rather than quietly running it. The safe thing is
    the default now, and the tutorial's `weights_only=False` is the opt-out.
    """)
    return


@app.cell
def _(mo, torch):
    from pathlib import Path

    _files = [
        ("model_weights.pth", "`model.state_dict()` — an ordered mapping of name to tensor"),
        ("model.pth", "the module object, pickled: the same tensors plus a reference to the class"),
    ]
    _rows = [
        {"file": name, "MB on disk": round(Path(name).stat().st_size / 1024**2, 1), "what is in it": description}
        for name, description in _files
        if Path(name).exists()
    ]

    try:
        torch.load("model.pth")
    except Exception as _refusal:
        _default_behavior = f"`{type(_refusal).__name__}` — {str(_refusal).splitlines()[0]}"
    else:
        _default_behavior = "loaded without complaint"

    mo.vstack(
        [
            mo.ui.table(_rows, selection=None),
            mo.md(f"`torch.load('model.pth')` with no `weights_only` argument: {_default_behavior}"),
            mo.callout(
                mo.md(
                    "**A collision worth knowing about.** This notebook writes `model.pth` into "
                    "the directory it sits in, and so does the quickstart notebook — a 528 MB "
                    "VGG16 over a 2.6 MB classifier, or the other way round, depending on which "
                    "you ran last. Whichever loses, the *other* notebook's load cell then fails "
                    "with an error about the wrong keys. Both filenames come from upstream and "
                    "are left as they are; they are ignored by git."
                ),
                kind="warn",
            ),
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > [!NOTE]
    > This approach uses Python [pickle](https://docs.python.org/3/library/pickle.html) module when
    > serializing the model, thus it relies on the actual class definition to be available when
    > loading the model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Related Tutorials

    - [Saving and Loading a General Checkpoint in
      PyTorch](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
    - [Tips for loading an nn.Module from a
      checkpoint](https://pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html?highlight=loading%20nn%20module%20from%20checkpoint)
    """)
    return


if __name__ == "__main__":
    app.run()
