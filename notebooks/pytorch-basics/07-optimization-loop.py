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
    [Transforms](transforms_tutorial.html) \|\| [Build Model](buildmodel_tutorial.html) \|\|
    [Autograd](autogradqs_tutorial.html) \|\| **Optimization** \|\| [Save & Load
    Model](saveloadrun_tutorial.html)

    # Optimizing Model Parameters

    Now that we have a model and data it's time to train, validate and test our model by optimizing
    its parameters on our data. Training a model is an iterative process; in each iteration the
    model makes a guess about the output, calculates the error in its guess (*loss*), collects the
    derivatives of the error with respect to its parameters (as we saw in the [previous
    section](autogradqs_tutorial.html)), and **optimizes** these parameters using gradient descent.
    For a more detailed walkthrough of this process, check out this video on [backpropagation from
    3Blue1Brown](https://www.youtube.com/watch?v=tIeHLnjs5U8).

    ## Prerequisite Code

    We load the code from the previous sections on [Datasets & DataLoaders](data_tutorial.html) and
    [Build Model](buildmodel_tutorial.html).
    """)
    return


@app.cell
def _():
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

    train_dataloader = DataLoader(training_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)

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

    model = NeuralNetwork()
    return model, nn, test_dataloader, torch, train_dataloader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Hyperparameters

    Hyperparameters are adjustable parameters that let you control the model optimization process.
    Different hyperparameter values can impact model training and convergence rates ([read
    more](https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html) about
    hyperparameter tuning)

    We define the following hyperparameters for training:

    - **Number of Epochs** - the number of times to iterate over the dataset
    - **Batch Size** - the number of data samples propagated through the network before the
      parameters are updated
    - **Learning Rate** - how much to update models parameters at each batch/epoch. Smaller values
      yield slow learning speed, while large values may result in unpredictable behavior during
      training.
    """)
    return


@app.cell
def _():
    learning_rate = 0.001
    batch_size = 64
    _epochs = 5
    return batch_size, learning_rate


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Optimization Loop

    Once we set our hyperparameters, we can then train and optimize our model with an optimization
    loop. Each iteration of the optimization loop is called an **epoch**.

    Each epoch consists of two main parts:

    - **The Train Loop** - iterate over the training dataset and try to converge to optimal
      parameters.
    - **The Validation/Test Loop** - iterate over the test dataset to check if model performance is
      improving.

    Let's briefly familiarize ourselves with some of the concepts used in the training loop. Jump
    ahead to see the `full-impl-label` of the optimization loop.

    ## Loss Function

    When presented with some training data, our untrained network is likely not to give the correct
    answer. **Loss function** measures the degree of dissimilarity of obtained result to the target
    value, and it is the loss function that we want to minimize during training. To calculate the
    loss we make a prediction using the inputs of our given data sample and compare it against the
    true data label value.

    Common loss functions include
    [nn.MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss)
    (Mean Square Error) for regression tasks, and
    [nn.NLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss)
    (Negative Log Likelihood) for classification.
    [nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss)
    combines `nn.LogSoftmax` and `nn.NLLLoss`.

    We pass our model's output logits to `nn.CrossEntropyLoss`, which will normalize the logits and
    compute the prediction error.
    """)
    return


@app.cell
def _(nn):
    # Initialize the loss function
    _loss_fn = nn.CrossEntropyLoss()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Optimizer

    Optimization is the process of adjusting model parameters to reduce model error in each
    training step. **Optimization algorithms** define how this process is performed (in this
    example we use Stochastic Gradient Descent). All optimization logic is encapsulated in the
    `optimizer` object. Here, we use the SGD optimizer; additionally, there are many [different
    optimizers](https://pytorch.org/docs/stable/optim.html) available in PyTorch such as ADAM and
    RMSProp, that work better for different kinds of models and data.

    We initialize the optimizer by registering the model's parameters that need to be trained, and
    passing in the learning rate hyperparameter.
    """)
    return


@app.cell
def _(learning_rate, model, torch):
    _optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Inside the training loop, optimization happens in three steps:

    - Call `optimizer.zero_grad()` to reset the gradients of model parameters. Gradients by default
      add up; to prevent double-counting, we explicitly zero them at each iteration.
    - Backpropagate the prediction loss with a call to `loss.backward()`. PyTorch deposits the
      gradients of the loss w.r.t. each parameter.
    - Once we have our gradients, we call `optimizer.step()` to adjust the parameters by the
      gradients collected in the backward pass.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Full Implementation

    We define `train_loop` that loops over our optimization code, and `test_loop` that evaluates
    the model's performance against our test data.
    """)
    return


@app.cell
def _(batch_size, torch):
    def train_loop(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()  # Set the model to training mode - important for batch normalization and dropout layers
        for batch, (X, y) in enumerate(dataloader):  # Unnecessary in this situation but added for best practices
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()  # Compute prediction and loss
            optimizer.step()
            optimizer.zero_grad()
            if batch % 100 == 0:
                loss, current = (loss.item(), batch * batch_size + len(X))  # Backpropagation
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test_loop(dataloader, model, loss_fn):
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = (0, 0)
        with torch.no_grad():
            for X, y in dataloader:
                pred = model(X)
                test_loss += loss_fn(
                    pred, y
                ).item()  # Set the model to evaluation mode - important for batch normalization and dropout layers
                correct += (
                    (pred.argmax(1) == y).type(torch.float).sum().item()
                )  # Unnecessary in this situation but added for best practices
        test_loss /= num_batches
        correct /= size
        print(
            f"Test Error: \n Accuracy: {100 * correct:>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )  # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode  # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True

    return test_loop, train_loop


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We initialize the loss function and optimizer, and pass it to `train_loop` and `test_loop`.
    Feel free to increase the number of epochs to track the model's improving performance.
    """)
    return


@app.cell
def _(
    learning_rate,
    model,
    nn,
    test_dataloader,
    test_loop,
    torch,
    train_dataloader,
    train_loop,
):
    _loss_fn = nn.CrossEntropyLoss()
    _optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    _epochs = 10
    for t in range(_epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, _loss_fn, _optimizer)
        test_loop(test_dataloader, model, _loss_fn)
    print("Done!")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Further Reading

    - [Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions)
    - [torch.optim](https://pytorch.org/docs/stable/optim.html)
    - [Warmstart Training a
      Model](https://pytorch.org/tutorials/recipes/recipes/warmstarting_model_using_parameters_from_a_different_model.html)
    """)
    return


if __name__ == "__main__":
    app.run()
