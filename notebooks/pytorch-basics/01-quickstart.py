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
    *PyTorch basics, 1 of 8 — after: [Tensors](02-tensors.py)*

    # Quickstart

    One complete pass, at speed: load a dataset, build a network, train it, save it, and
    look at exactly what it gets wrong. Each section compresses a later notebook in this
    series; the *read more* links point there.

    ## Working with data

    PyTorch has two [primitives to work with data](https://pytorch.org/docs/stable/data.html):
    `torch.utils.data.DataLoader` and `torch.utils.data.Dataset`. `Dataset` stores the samples and
    their corresponding labels, and `DataLoader` wraps an iterable around the `Dataset`.
    """)
    return


@app.cell
def _():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from torchvision.transforms import v2

    return DataLoader, datasets, nn, torch, v2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    PyTorch offers domain-specific libraries such as
    [TorchText](https://pytorch.org/text/stable/index.html),
    [TorchVision](https://pytorch.org/vision/stable/index.html), and
    [TorchAudio](https://pytorch.org/audio/stable/index.html), all of which include datasets. For
    this tutorial, we will be using a TorchVision dataset.

    The `torchvision.datasets` module contains `Dataset` objects for many real-world vision data
    like CIFAR, COCO ([full list here](https://pytorch.org/vision/stable/datasets.html)). In this
    tutorial, we use the FashionMNIST dataset. Every TorchVision `Dataset` includes two arguments:
    `transform` and `target_transform` to modify the samples and labels respectively.
    """)
    return


@app.cell
def _(datasets, torch, v2):
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    )
    return test_data, training_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We pass the `Dataset` as an argument to `DataLoader`. This wraps an iterable over our dataset,
    and supports automatic batching, sampling, shuffling and multiprocess data loading. Here we
    define a batch size of 64, i.e. each element in the dataloader iterable will return a batch of
    64 features and labels.
    """)
    return


@app.cell
def _(DataLoader, test_data, training_data):
    batch_size = 64

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    for X, _y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {_y.shape} {_y.dtype}")
        break
    return test_dataloader, train_dataloader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read more in [Datasets & DataLoaders](03-datasets-and-dataloaders.py).
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
    ## Creating Models

    To define a neural network in PyTorch, we create a class that inherits from
    [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html). We define the
    layers of the network in the `__init__` function and specify how data will pass through the
    network in the `forward` function. To accelerate operations in the neural network, we move it
    to the [accelerator](https://pytorch.org/docs/stable/torch.html#accelerators) such as CUDA,
    MPS, MTIA, or XPU. If the current accelerator is available, we will use it. Otherwise, we use
    the CPU.
    """)
    return


@app.cell
def _(nn, torch):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    # Define model
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28 * 28, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 10)
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    model = NeuralNetwork().to(device)
    print(model)
    return NeuralNetwork, device, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The same object, rendered rather than printed

    `print(model)` produced the text above. Returning the module instead hands it to
    marimo's formatter: parameter counts per layer, dtype and device, and a color per
    layer kind. Open `linear_relu_stack` and read the counts — 401,920 of the model's
    669,706 parameters are in the *first* `Linear`, because it is the only one that
    meets all 784 pixels. That ratio is why input resolution costs more than depth.
    """)
    return


@app.cell
def _(model):
    model
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read more in [Build Model](05-build-model.py).
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
    ## Optimizing the Model Parameters

    To train a model, we need a [loss
    function](https://pytorch.org/docs/stable/nn.html#loss-functions) and an
    [optimizer](https://pytorch.org/docs/stable/optim.html).
    """)
    return


@app.cell
def _(model, nn, torch):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return loss_fn, optimizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In a single training loop, the model makes predictions on the training dataset (fed to it in
    batches), and backpropagates the prediction error to adjust the model's parameters.
    """)
    return


@app.cell
def _(device):
    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, _y) in enumerate(dataloader):
            X, _y = (X.to(device), _y.to(device))

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, _y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = (loss.item(), (batch + 1) * len(X))
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return (train,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We also check the model's performance against the test dataset to ensure it is learning.
    """)
    return


@app.cell
def _(device, torch):
    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = (0, 0)
        with torch.no_grad():
            for X, _y in dataloader:
                X, _y = (X.to(device), _y.to(device))
                pred = model(X)
                test_loss = test_loss + loss_fn(pred, _y).item()
                correct = correct + (pred.argmax(1) == _y).type(torch.float).sum().item()
        test_loss = test_loss / num_batches
        correct = correct / size
        print(f"Test Error: \n Accuracy: {100 * correct:>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return (test,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The training process is conducted over several iterations (*epochs*). During each epoch, the
    model learns parameters to make better predictions. We print the model's accuracy and loss at
    each epoch; we'd like to see the accuracy increase and the loss decrease with every epoch.
    """)
    return


@app.cell
def _(
    loss_fn,
    model,
    optimizer,
    test,
    test_dataloader,
    train,
    train_dataloader,
):
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read more in [Optimization](07-optimization-loop.py).
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
    ## Saving Models

    A common way to save a model is to serialize the internal state dictionary (containing the
    model parameters).
    """)
    return


@app.cell
def _(model, torch):
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading Models

    The process for loading a model includes re-creating the model structure and loading the state
    dictionary into it.
    """)
    return


@app.cell
def _(NeuralNetwork, device, torch):
    model_1 = NeuralNetwork().to(device)
    model_1.load_state_dict(torch.load("model.pth", weights_only=True))
    return (model_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This model can now be used to make predictions.
    """)
    return


@app.cell
def _(device, model_1, test_data, torch):
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
    model_1.eval()
    x, _y = (test_data[0][0], test_data[0][1])
    with torch.no_grad():
        x = x.to(device)
        pred = model_1(x)
        predicted, actual = (classes[pred[0].argmax(0)], classes[_y])
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
    return (classes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read more in [Save & Load Model](08-save-load-run.py).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## All ten thousand test images, not just the first

    The cell above predicts `test_data[0]` and prints one line. The same three lines run
    over the whole test set below, once, and everything after it is a *view* onto those
    stored predictions rather than another forward pass. That separation is worth
    noticing: it is the same reason a training loop keeps `model.eval()` and
    `torch.no_grad()` around inference, and the reason an evaluation is cheap to slice
    afterwards but expensive to redo.
    """)
    return


@app.cell
def _():
    import altair as alt
    import pandas as pd

    return alt, pd


@app.cell
def _(DataLoader, device, model_1, test_data, torch):
    @torch.no_grad()
    def _predict_everything():
        model_1.eval()
        batched = [model_1(images.to(device)) for images, _ in DataLoader(test_data, batch_size=512)]
        return torch.cat(batched).cpu()

    test_probabilities = _predict_everything().softmax(1)
    test_predicted = test_probabilities.argmax(1)
    test_actual = test_data.targets
    test_hits = test_predicted == test_actual
    return test_actual, test_hits, test_predicted, test_probabilities


@app.cell
def _(mo, test_hits, test_probabilities):
    mo.hstack(
        [
            mo.stat(
                f"{test_hits.float().mean():.1%}",
                label="Accuracy",
                caption=f"{len(test_hits):,} images",
                bordered=True,
            ),
            mo.stat(f"{(~test_hits).sum():,}", label="Mistakes", caption="browsable below", bordered=True),
            mo.stat(
                f"{test_probabilities.max(1).values.mean():.0%}",
                label="Mean confidence",
                caption="probability of the chosen class",
                bordered=True,
            ),
        ],
        justify="start",
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Where the mistakes are

    A single accuracy number hides *which* classes the model confuses, and the answer is
    never uniform. Click a square: the images behind it appear below. The color scale is
    symlog: one picture holds counts from single digits to nearly a thousand (each row is
    1,000 test images), and on a linear scale most of the mistake cells — the interesting
    ones — would be indistinguishable from an empty cell.
    """)
    return


@app.cell
def _(alt, classes, mo, pd, test_actual, test_predicted):
    _pairs = pd.DataFrame(
        {
            "actual": [classes[i] for i in test_actual.tolist()],
            "predicted": [classes[i] for i in test_predicted.tolist()],
        }
    )
    _heatmap = (
        alt.Chart(_pairs.groupby(["actual", "predicted"], as_index=False).size())
        .mark_rect()
        .encode(
            x=alt.X("predicted:N", sort=classes, title="predicted"),
            y=alt.Y("actual:N", sort=classes, title="actual"),
            color=alt.Color("size:Q", scale=alt.Scale(scheme="blues", type="symlog"), title="images"),
            tooltip=["actual", "predicted", "size"],
        )
        .properties(width=420, height=420)
    )
    confusion = mo.ui.altair_chart(_heatmap)
    confusion
    return (confusion,)


@app.cell
def _(classes, confusion, mo, test_actual, test_hits, test_predicted, torch):
    if len(confusion.value):
        _selected = {(row.actual, row.predicted) for row in confusion.value.itertuples()}
        _mask = torch.tensor(
            [
                (classes[a], classes[p]) in _selected
                for a, p in zip(test_actual.tolist(), test_predicted.tolist(), strict=True)
            ]
        )
        _what = "in the squares you selected"
    else:
        _mask = ~test_hits
        _what = "the model got wrong — select squares above to narrow this down"
    picked = _mask.nonzero().flatten()
    mo.md(f"**{len(picked):,}** images {_what}.")
    return (picked,)


@app.cell
def _(mo, picked):
    offset = mo.ui.slider(
        0, max(len(picked) - 10, 0), step=10, value=0, label="browse", full_width=True, show_value=True
    )
    offset
    return (offset,)


@app.cell
def _(classes, mo, offset, picked, test_actual, test_data, test_predicted, test_probabilities):
    _cards = [
        mo.vstack(
            [
                mo.image(test_data[i][0].squeeze(0), width=88, vmin=0, vmax=1, rounded=True),
                mo.md(
                    f"said **{classes[test_predicted[i]]}** "
                    f"({test_probabilities[i].max():.0%})<br>was {classes[test_actual[i]]}"
                ),
            ],
            gap=0.25,
        )
        for i in picked[offset.value : offset.value + 10].tolist()
    ]
    mo.hstack(_cards, justify="start", wrap=True, gap=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The squares off the diagonal are not scattered: they cluster into upper-body garments
    and into footwear. Nothing in the loss function said those groups existed — the model
    was told only that the ten labels are distinct — so the structure comes from the
    pixels. Rather than name the pairs here, where they would go stale the first time you
    change an epoch count, the table below reads them off the run you just did.
    """)
    return


@app.cell
def _(classes, mo, pd, test_actual, test_hits, test_predicted):
    _misses = pd.DataFrame(
        {
            "true label": [classes[i] for i in test_actual[~test_hits].tolist()],
            "the model said": [classes[i] for i in test_predicted[~test_hits].tolist()],
        }
    )
    _ranked = _misses.groupby(["true label", "the model said"], as_index=False).size()
    _ranked = _ranked.sort_values("size", ascending=False).head(8).rename(columns={"size": "images"})
    mo.ui.table(_ranked, selection=None, label="Where the mistakes concentrate")
    return


@app.cell
def _(classes, mo, test_actual, test_hits, torch):
    _recall = torch.stack([test_hits[test_actual == i].float().mean() for i in range(len(classes))])
    _worst = _recall.argmin().item()
    mo.md(
        f"""
        Read the extreme case off that table: **{classes[_worst]}** is recovered
        {_recall[_worst]:.0%} of the time, against {_recall.max():.0%} for
        {classes[_recall.argmax().item()]}. A single accuracy figure averages those two
        together, which is the argument for looking at a confusion matrix before
        believing one number — and, later, the argument for a per-class metric in any
        evaluation that decides whether a training change helped.
        """
    )
    return


if __name__ == "__main__":
    app.run()
