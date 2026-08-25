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
    [Learn the Basics](intro.html) \|\| **Quickstart** \|\| [Tensors](tensorqs_tutorial.html) \|\|
    [Datasets & DataLoaders](data_tutorial.html) \|\| [Transforms](transforms_tutorial.html) \|\|
    [Build Model](buildmodel_tutorial.html) \|\| [Autograd](autogradqs_tutorial.html) \|\|
    [Optimization](optimization_tutorial.html) \|\| [Save & Load Model](saveloadrun_tutorial.html)

    # Quickstart

    This section runs through the API for common tasks in machine learning. Refer to the links in
    each section to dive deeper.

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
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    # Create data loaders.
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    for X, _y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {_y.shape} {_y.dtype}")
        break
    return test_dataloader, train_dataloader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read more about [loading data in PyTorch](data_tutorial.html).
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
    # Creating Models

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
    Read more about [building neural networks in PyTorch](buildmodel_tutorial.html).
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
    # Optimizing the Model Parameters

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
            pred = model(X)
            loss = loss_fn(pred, _y)  # Compute prediction error
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch % 100 == 0:  # Backpropagation
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
    Read more about [Training your model](optimization_tutorial.html).
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
    # Saving Models

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
    # Loading Models

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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read more about [Saving & Loading your model](saveloadrun_tutorial.html).
    """)
    return


if __name__ == "__main__":
    app.run()
