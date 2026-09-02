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
    *PyTorch basics, 8 of 8 — before this: [Optimization](07-optimization-loop.py)*

    # Save and Load the Model

    Training produces a model worth keeping. This last notebook is about persisting that
    state: saving it to disk, loading it into a fresh model, and running a prediction from
    the reload.
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
    ## Saving and Loading Model Weights

    A PyTorch model keeps its learned state in an internal state dictionary, its
    `state_dict`: an ordered mapping from name to tensor, holding the parameters plus any
    registered buffers (batch normalization's running statistics, for instance) — and
    nothing else, no code and no structure. `torch.save` persists it:
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
    ### What 528 megabytes is made of

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


@app.cell(hide_code=True)
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
    To load the weights back, first build an instance of the same architecture — the file
    holds tensors, not the code that gives them meaning — then hand the loaded dictionary
    to `load_state_dict()`.

    `weights_only=True` restricts unpickling to plain tensor data, refusing anything that
    would run code. It is already the default in the torch this repo pins — *The two
    files, on disk* below shows the refusal — so writing it out is documentation: this
    file is expected to hold nothing but weights.
    """)
    return


@app.cell
def _(models, torch):
    model_1 = models.vgg16()  # no weights= argument: the same architecture, untrained
    model_1.load_state_dict(torch.load("model_weights.pth", weights_only=True))
    model_1.eval()  # switches to inference behavior -- demonstrated just below
    return (model_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A prediction from the reload

    "Run" is the last third of save-load-run. The probe below is random noise, so which of
    the 1,000 ImageNet classes wins means nothing — the question worth asking is whether
    the reloaded model gives the same answer twice.

    It does not have to. The module tree above shows `Dropout(p=0.5)` twice inside
    `classifier`: in training mode, dropout zeroes a random half of that layer's inputs on
    every call — regularization while learning, pure nondeterminism at inference.
    `model_1.eval()` switches dropout (and batch normalization, which VGG16 predates) to
    deterministic inference behavior; `.train()` is the way back, and the optimization
    loop of the previous notebook toggled between the two on purpose. Predict what
    distinguishes the two rows before running the cell.
    """)
    return


@app.cell
def _(model_1, models, torch):
    _probe = torch.randn(1, 3, 224, 224, generator=torch.Generator().manual_seed(0))
    _labels = models.VGG16_Weights.IMAGENET1K_V1.meta["categories"]

    def _top1(net):
        with torch.no_grad():
            scores = net(_probe)
        return f"{_labels[scores.argmax()]} ({scores.max().item():.2f})"

    model_1.train()  # dropout active again; restored to eval() below, so re-running is safe
    _while_training = [_top1(model_1) for _ in range(3)]
    model_1.eval()
    _after_eval = [_top1(model_1) for _ in range(3)]
    {"model_1.train()": _while_training, "model_1.eval()": _after_eval}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Saving and Loading the Whole Model

    Loading weights required instantiating the model class first, because the file held no
    structure. The second form `torch.save` accepts promises to skip that step: pass
    `model` itself, rather than `model.state_dict()`, to the saving function:
    """)
    return


@app.cell
def _(model_1, torch):
    torch.save(model_1, "model.pth")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Loading it back needs two things. `weights_only=False`, because the file now holds a
    pickled Python object rather than plain tensors. And `torchvision` importable at load
    time, because pickle stores a *reference* to the class, not its code — the reliance
    [Saving and loading
    torch.nn.Modules](https://docs.pytorch.org/docs/stable/notes/serialization.html#saving-and-loading-torch-nn-modules)
    names when it calls the `state_dict` route the best practice and this one a legacy use
    of `torch.save`. The next section measures why.
    """)
    return


@app.cell
def _(torch):
    model_2 = torch.load("model.pth", weights_only=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The two files, on disk

    Best practice against legacy is the documentation's framing; both files now exist, so
    it is checkable here.

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
    the default now, and the `weights_only=False` written above is the opt-out.
    """)
    return


@app.cell
def _(torch):
    try:
        torch.load("model.pth")  # no weights_only argument: the default decides
        bare_load = "loaded without complaint"
    except Exception as _refusal:
        bare_load = f"`{type(_refusal).__name__}` — {str(_refusal).split('.')[0]}."
    return (bare_load,)


@app.cell(hide_code=True)
def _(bare_load, mo):
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

    mo.vstack(
        [
            mo.ui.table(_rows, selection=None),
            mo.md(f"`torch.load('model.pth')` with no `weights_only` argument: {bare_load}"),
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
    ---

    ## Where to go next

    This closes the series: tensors, data, a model, the loop that trains it, and now the
    state that outlives the process. The `state_dict` idea keeps scaling up from here —
    each of these is the same mapping-of-name-to-tensor wearing more machinery:

    - **A resumable checkpoint is more than model weights.** Mid-run state also includes
      the optimizer's `state_dict` (Adam keeps two running moments per parameter, so this
      can outweigh the model itself), the epoch, and the last loss — [Saving and Loading a
      General
      Checkpoint](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
      is the recipe.
    - **Loading has sharp edges at scale.** [Tips for loading an nn.Module from a
      checkpoint](https://pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html)
      covers `mmap=True` and `assign=True` — how to load a model without materializing it
      in memory twice.
    - **The ecosystem has moved past pickle.** The Hugging Face Hub serves
      [safetensors](https://huggingface.co/docs/safetensors): tensors plus a JSON header,
      nothing executable, so the `weights_only` question this notebook demonstrated is
      designed away rather than defaulted away.
    - **Distributed training splits the dictionary itself.** Once a model shards across
      GPUs, no single rank holds a full `state_dict`, and
      [torch.distributed.checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)
      saves and loads the pieces in parallel — where this repository is headed next.
    """)
    return


if __name__ == "__main__":
    app.run()
