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
# The training cell added below is the one thing here expensive enough to want a guard,
# and it has one: mo.stop on a run button, so autorun re-runs it only after a click.
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
    return (
        DataLoader,
        NeuralNetwork,
        model,
        nn,
        test_dataloader,
        torch,
        train_dataloader,
        training_data,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Explore — the model this notebook is about to train

    The prerequisite cell builds it silently. Returning it draws marimo's tree, which is
    worth one look here for the line the other notebooks do not have cause to point at:
    **device `cpu`**. Nothing in this tutorial moves the model to an accelerator, so the
    ten epochs above run on the processor. That is also the one substantive difference in
    the training cell further down, which does move it, and finishes in seconds.
    """)
    return


@app.cell
def _(model):
    model
    return


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
    ## Explore — the same loop, with the curve drawn as it runs

    A hundred and twenty printed loss values are a time series rendered as text. The cell
    below runs the identical loop and draws it instead: training loss every twenty
    batches, test loss and accuracy once per epoch, redrawn in place while the run
    proceeds.

    Every run starts from the same seed and a fresh model, so two runs differ only by
    what you changed. Worth trying, in this order:

    - **learning rate 0.001**, the tutorial's value, for two epochs. The curve falls and
      is still falling when it stops. That is what an undertrained model looks like, and
      the 10 epochs above are still on the same slope.
    - **learning rate 0.1.** Roughly a hundred times fewer steps to reach the same loss.
      The tutorial's value is not a default worth keeping; it is a value chosen to be
      safe without a schedule.
    - **learning rate 1.0.** The loss stops falling and starts wandering, or leaves. The
      failure is not gradual — there is a threshold, and past it the step overshoots the
      curvature it is descending.
    - **Adam at 0.001** against **SGD at 0.001.** Same number, different distance
      travelled, because Adam divides each parameter's step by a running estimate of that
      parameter's own gradient scale. A learning rate means something different for each.

    The model here is moved to the accelerator, unlike the tutorial's, which is why this
    finishes in seconds.
    """)
    return


@app.cell
def _(mo):
    optimizer_pick = mo.ui.dropdown(
        {"SGD": "sgd", "SGD, momentum 0.9": "momentum", "Adam": "adam"},
        value="SGD",
        label="optimizer",
    )
    rate_pick = mo.ui.slider(
        steps=[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
        value=0.001,
        label="learning rate",
        show_value=True,
    )
    epochs_pick = mo.ui.slider(1, 5, value=2, label="epochs", show_value=True)
    batch_pick = mo.ui.slider(steps=[32, 64, 128, 256], value=64, label="batch size", show_value=True)
    start_training = mo.ui.run_button(label="Train", kind="success")
    mo.hstack(
        [
            mo.vstack([optimizer_pick, rate_pick]),
            mo.vstack([epochs_pick, batch_pick]),
            start_training,
        ],
        justify="start",
        align="center",
        gap=2,
        wrap=True,
    )
    return batch_pick, epochs_pick, optimizer_pick, rate_pick, start_training


@app.cell
def _():
    import altair as alt
    import pandas as pd

    def loss_curves(history, checkpoints):
        """Train loss against test loss, on one pair of axes."""
        layers = [
            alt.Chart(pd.DataFrame(history))
            .mark_line(color="#4c78a8", opacity=0.8)
            .encode(
                x=alt.X("batches:Q", title="batches seen"),
                y=alt.Y("loss:Q", title="loss", scale=alt.Scale(zero=False)),
            )
        ]
        if checkpoints:
            frame = pd.DataFrame(checkpoints)
            layers.append(
                alt.Chart(frame)
                .mark_line(color="#e45756", point=True)
                .encode(x="batches:Q", y="loss:Q", tooltip=["epoch:Q", "loss:Q", "accuracy:Q"])
            )
        return alt.layer(*layers).properties(width=560, height=260)

    return alt, loss_curves, pd


@app.cell
def _(
    DataLoader,
    NeuralNetwork,
    batch_pick,
    epochs_pick,
    loss_curves,
    mo,
    nn,
    optimizer_pick,
    rate_pick,
    start_training,
    test_dataloader,
    torch,
    training_data,
):
    mo.stop(
        not start_training.value,
        mo.md("Set the controls above, then press **Train**. Nothing runs until you do."),
    )

    _device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    torch.manual_seed(0)
    _model = NeuralNetwork().to(_device)
    _loader = DataLoader(training_data, batch_size=int(batch_pick.value), shuffle=True)
    _loss_fn = nn.CrossEntropyLoss()
    _optimizer = {
        "sgd": lambda p: torch.optim.SGD(p, lr=rate_pick.value),
        "momentum": lambda p: torch.optim.SGD(p, lr=rate_pick.value, momentum=0.9),
        "adam": lambda p: torch.optim.Adam(p, lr=rate_pick.value),
    }[optimizer_pick.value](_model.parameters())

    history, checkpoints, seen = [], [], 0
    with mo.status.progress_bar(
        total=epochs_pick.value * len(_loader), title="training", completion_title="done"
    ) as bar:
        # The progress bar is output 0; the chart is appended as output 1 and replaced in
        # place, since a plain output.replace would take the bar with it.
        mo.output.append(loss_curves(history, checkpoints))
        for _epoch in range(epochs_pick.value):
            _model.train()
            for _images, _labels in _loader:
                _images, _labels = _images.to(_device), _labels.to(_device)
                _batch_loss = _loss_fn(_model(_images), _labels)
                _batch_loss.backward()
                _optimizer.step()
                _optimizer.zero_grad()
                seen += 1
                if seen % 20 == 0:
                    history.append({"batches": seen, "loss": _batch_loss.item()})
                    mo.output.replace_at_index(loss_curves(history, checkpoints), 1)
                bar.update(subtitle=f"epoch {_epoch + 1}, loss {_batch_loss.item():.3f}")

            _model.eval()
            _total_loss, _hits, _count = 0.0, 0, 0
            with torch.no_grad():
                for _images, _labels in test_dataloader:
                    _images, _labels = _images.to(_device), _labels.to(_device)
                    _logits = _model(_images)
                    _total_loss += _loss_fn(_logits, _labels).item() * len(_labels)
                    _hits += (_logits.argmax(1) == _labels).sum().item()
                    _count += len(_labels)
            checkpoints.append(
                {
                    "epoch": _epoch + 1,
                    "batches": seen,
                    "loss": _total_loss / _count,
                    "accuracy": round(100 * _hits / _count, 2),
                }
            )
            mo.output.replace_at_index(loss_curves(history, checkpoints), 1)

    mo.output.append(
        mo.hstack(
            [
                mo.stat(f"{checkpoints[-1]['accuracy']:.1f}%", label="test accuracy", bordered=True),
                mo.stat(f"{checkpoints[-1]['loss']:.3f}", label="test loss", bordered=True),
                mo.stat(f"{seen:,}", label="optimizer steps", bordered=True),
                mo.stat(_device, label="device", bordered=True),
            ],
            justify="start",
            gap=1,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Explore — what the three optimizers actually do, on a surface you can turn

    In 669,706 dimensions the difference between SGD and Adam is a number on a chart. In
    two, it is a shape. The surface below is a narrow valley — `0.05x² + y²`, twenty times
    steeper across than along — and all three optimizers are the real `torch.optim`
    classes stepping on it, not imitations.

    Drag to rotate. The numbers below are the distance still left to the minimum after the
    run, which is the only honest way to compare the three.

    Start at **0.06** and walk the rate up. Plain SGD tells the whole story on its own:

    - At 0.06 it does not bounce at all. It kills the steep direction in a few steps and
      then crawls along the floor, ending 1.98 from the minimum having started at 4.9.
    - At 0.3 it arrives. At 0.6 and 0.9 it oscillates violently across the valley — the
      curve visibly zig-zags — and *still* converges, in fact fastest of all at 0.9.
    - At exactly 1.0 the oscillation stops decaying: `y` flips sign every step at constant
      size, forever. At 1.02 it explodes. That edge is not luck, it is `2/λ`, where λ = 2
      is the curvature of the steep direction, and it does not depend on the shallow one.

    So one number is asked to do two jobs. The ceiling comes from the steepest direction,
    the progress from the shallowest, and the ratio between them — 20 here, the condition
    number — is how much worse than ideal SGD has to be. Real models are far worse
    conditioned than 20.

    The other two buy their way out differently. **Momentum** integrates the gradient, so
    the shallow direction accumulates speed; it rings in the steep direction more than SGD
    does, not less, and still lands about a thousand times closer at 0.06. **Adam**
    divides each coordinate's step by a running estimate of that coordinate's own gradient
    size, which removes the anisotropy rather than tolerating it — that per-parameter
    rescaling is also why an Adam learning rate does not transfer to SGD.
    """)
    return


@app.cell
def _(mo):
    valley_rate = mo.ui.slider(
        steps=[0.001, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 0.9, 1.0, 1.02],
        value=0.06,
        label="learning rate",
        show_value=True,
    )
    valley_steps = mo.ui.slider(20, 400, value=140, step=20, label="steps", show_value=True)
    # 1.0 is 2/λ for the steep direction: the exact edge of stability, where the
    # oscillation neither grows nor decays. 1.02 is past it.
    mo.hstack([valley_rate, valley_steps], justify="start", gap=2)
    return valley_rate, valley_steps


@app.cell
def _(mo, torch, valley_rate, valley_steps):
    import numpy as np
    import plotly.graph_objects as plot

    def valley(x, y):
        """Twenty times steeper across than along. Any anisotropy would do."""
        return 0.05 * x**2 + y**2

    def descend(build):
        point = torch.tensor([-4.6, 1.7], requires_grad=True)
        optimizer = build([point])
        path = [point.detach().clone()]
        for _ in range(valley_steps.value):
            valley(point[0], point[1]).backward()
            optimizer.step()
            optimizer.zero_grad()
            path.append(point.detach().clone())
        return torch.stack(path)

    runs = {
        "SGD": ("#e45756", descend(lambda p: torch.optim.SGD(p, lr=valley_rate.value))),
        "momentum 0.9": ("#f2a154", descend(lambda p: torch.optim.SGD(p, lr=valley_rate.value, momentum=0.9))),
        "Adam": ("#54a24b", descend(lambda p: torch.optim.Adam(p, lr=valley_rate.value))),
    }

    _grid_x, _grid_y = np.meshgrid(np.linspace(-5, 5, 90), np.linspace(-2, 2, 60))
    _figure = plot.Figure(
        data=[
            plot.Surface(
                x=_grid_x,
                y=_grid_y,
                z=valley(_grid_x, _grid_y),
                colorscale="Blues",
                opacity=0.75,
                showscale=False,
                contours={"z": {"show": True, "usecolormap": True, "width": 1}},
            )
        ]
    )
    for _name, (_color, _path) in runs.items():
        _xs, _ys = _path[:, 0].numpy(), _path[:, 1].numpy()
        _figure.add_trace(
            plot.Scatter3d(
                x=_xs,
                y=_ys,
                # Lifted a little so the line reads against the surface rather than through it.
                z=valley(_xs, _ys) + 0.15,
                mode="lines+markers",
                line={"color": _color, "width": 5},
                marker={"size": 2, "color": _color},
                name=_name,
            )
        )
    _figure.update_layout(
        height=560,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        # Fixed ranges, so a diverging run leaves the frame instead of flattening the
        # surface into a plane. The distance readout below still reports where it went.
        scene={
            "xaxis": {"title": "x  (shallow)", "range": [-5, 5]},
            "yaxis": {"title": "y  (steep)", "range": [-2, 2]},
            "zaxis": {"title": "loss", "range": [0, 5]},
        },
        legend={"orientation": "h", "y": 0.02},
    )
    mo.vstack(
        [
            _figure,
            mo.hstack(
                [
                    mo.stat(f"{_path[-1].norm():.3f}", label=_name, caption="distance left to (0, 0)", bordered=True)
                    for _name, (_, _path) in runs.items()
                ],
                justify="start",
                gap=1,
            ),
        ]
    )
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
