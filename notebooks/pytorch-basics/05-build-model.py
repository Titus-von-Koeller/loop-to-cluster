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
    [Learn the Basics](intro.html) \|\| [Quickstart](quickstart_tutorial.html) \|\|
    [Tensors](tensorqs_tutorial.html) \|\| [Datasets & DataLoaders](data_tutorial.html) \|\|
    [Transforms](transforms_tutorial.html) \|\| **Build Model** \|\|
    [Autograd](autogradqs_tutorial.html) \|\| [Optimization](optimization_tutorial.html) \|\| [Save
    & Load Model](saveloadrun_tutorial.html)

    # Build the Neural Network

    Neural networks comprise of layers/modules that perform operations on data. The
    [torch.nn](https://pytorch.org/docs/stable/nn.html) namespace provides all the building blocks
    you need to build your own neural network. Every module in PyTorch subclasses the
    [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html). A neural network
    is a module itself that consists of other modules (layers). This nested structure allows for
    building and managing complex architectures easily.

    In the following sections, we'll build a neural network to classify images in the FashionMNIST
    dataset.
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
    # Get Device for Training

    We want to be able to train our model on an
    [accelerator](https://pytorch.org/docs/stable/torch.html#accelerators) such as CUDA, MPS, MTIA,
    or XPU. If the current accelerator is available, we will use it. Otherwise, we use the CPU.
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
    # Define the Class

    We define our neural network by subclassing `nn.Module`, and initialize the neural network
    layers in `__init__`. Every `nn.Module` subclass implements the operations on input data in the
    `forward` method.
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
    We create an instance of `NeuralNetwork`, and move it to the `device`, and print its structure.
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
    To use the model, we pass it the input data. This executes the model's `forward`, along with
    some [background
    operations](https://github.com/pytorch/pytorch/blob/270111b7b611d174967ed204776985cefca9c144/torch/nn/modules/module.py#L866).
    Do not call `model.forward()` directly!

    Calling the model on the input returns a 2-dimensional tensor with dim=0 corresponding to each
    output of 10 raw predicted values for each class, and dim=1 corresponding to the individual
    values of each output. We get the prediction probabilities by passing it through an instance of
    the `nn.Softmax` module.
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
    # Model Layers

    Let's break down the layers in the FashionMNIST model. To illustrate it, we will take a sample
    minibatch of 3 images of size 28x28 and see what happens to it as we pass it through the
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
    # nn.Flatten

    We initialize the [nn.Flatten](https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html)
    layer to convert each 2D 28x28 image into a contiguous array of 784 pixel values ( the
    minibatch dimension (at dim=0) is maintained).
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
    # nn.Linear

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
    # nn.ReLU

    Non-linear activations are what create the complex mappings between the model's inputs and
    outputs. They are applied after linear transformations to introduce *nonlinearity*, helping
    neural networks learn a wide variety of phenomena.

    In this model, we use [nn.ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html)
    between our linear layers, but there's other activations to introduce non-linearity in your
    model.
    """)
    return


@app.cell
def _(hidden1, nn):
    print(f"Before ReLU: {hidden1}\n\n")
    hidden1_1 = nn.ReLU()(hidden1)
    print(f"After ReLU: {hidden1_1}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # nn.Sequential

    [nn.Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) is an
    ordered container of modules. The data is passed through all the modules in the same order as
    defined. You can use sequential containers to put together a quick network like `seq_modules`.
    """)
    return


@app.cell
def _(flatten, layer1, nn, torch):
    seq_modules = nn.Sequential(flatten, layer1, nn.ReLU(), nn.Linear(20, 10))
    input_image_1 = torch.rand(3, 28, 28)
    logits_1 = seq_modules(input_image_1)
    return (logits_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # nn.Softmax

    The last linear layer of the neural network returns <span class="title-ref">logits</span> - raw
    values in \[-infty, infty\] - which are passed to the
    [nn.Softmax](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html) module. The
    logits are scaled to values \[0, 1\] representing the model's predicted probabilities for each
    class. `dim` parameter indicates the dimension along which the values must sum to 1.
    """)
    return


@app.cell
def _(logits_1, nn):
    softmax = nn.Softmax(dim=1)
    _pred_probab = softmax(logits_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model Parameters

    Many layers inside a neural network are *parameterized*, i.e. have associated weights and
    biases that are optimized during training. Subclassing `nn.Module` automatically tracks all
    fields defined inside your model object, and makes all parameters accessible using your model's
    `parameters()` or `named_parameters()` methods.

    In this example, we iterate over each parameter, and print its size and a preview of its
    values.
    """)
    return


@app.cell
def _(model):
    print(f"Model structure: {model}\n\n")

    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
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
    # Further Reading

    - [torch.nn API](https://pytorch.org/docs/stable/nn.html)
    """)
    return


if __name__ == "__main__":
    app.run()
