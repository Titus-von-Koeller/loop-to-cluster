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
# Cells under an "Explore" heading are additions; everything else is the upstream
# tutorial as converted.

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
    **Autograd** \|\| [Optimization](optimization_tutorial.html) \|\| [Save & Load
    Model](saveloadrun_tutorial.html)

    # Automatic Differentiation with `torch.autograd`

    When training neural networks, the most frequently used algorithm is **back propagation**. In
    this algorithm, parameters (model weights) are adjusted according to the **gradient** of the
    loss function with respect to the given parameter.

    To compute those gradients, PyTorch has a built-in differentiation engine called
    `torch.autograd`. It supports automatic computation of gradient for any computational graph.

    Consider the simplest one-layer neural network, with input `x`, parameters `w` and `b`, and
    some loss function. It can be defined in PyTorch in the following manner:
    """)
    return


@app.cell
def _():
    import torch

    x = torch.ones(5)  # input tensor
    y = torch.zeros(3)  # expected output
    w = torch.randn(5, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)
    z = torch.matmul(x, w) + b
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
    return b, loss, torch, w, x, y, z


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tensors, Functions and Computational graph

    This code defines the following **computational graph**:

    <figure>
    <img src="/_static/img/basics/comp-graph.png" alt="/_static/img/basics/comp-graph.png" />
    </figure>

    In this network, `w` and `b` are **parameters**, which we need to optimize. Thus, we need to be
    able to compute the gradients of loss function with respect to those variables. In order to do
    that, we set the `requires_grad` property of those tensors.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > [!NOTE]
    > You can set the value of `requires_grad` when creating a tensor, or later by using
    > `x.requires_grad_(True)` method.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A function that we apply to tensors to construct computational graph is in fact an object of
    class `Function`. This object knows how to compute the function in the *forward* direction, and
    also how to compute its derivative during the *backward propagation* step. A reference to the
    backward propagation function is stored in `grad_fn` property of a tensor. You can find more
    information of `Function` [in the
    documentation](https://pytorch.org/docs/stable/autograd.html#function).
    """)
    return


@app.cell
def _(loss, z):
    print(f"Gradient function for z = {z.grad_fn}")
    print(f"Gradient function for loss = {loss.grad_fn}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Explore — the graph itself, walked and drawn

    The section above says a graph exists and prints two of its nodes. It is reachable:
    every tensor that required grad carries `.grad_fn`, and every `grad_fn` carries
    `.next_functions`, the edges to whatever produced its inputs. Walking that from `loss`
    to the leaves is a dozen lines, and it draws the picture the text describes.

    Read it bottom-up, which is the direction `backward()` travels. The ovals are
    `AccumulateGrad`: the ends of the road, one per leaf tensor with `requires_grad=True`,
    and the only places a `.grad` is ever written. The double-outlined box at the top is
    the root you called `backward()` on; the plain boxes between them are operations.

    One node is not in the code you wrote. `torch.matmul(x, w)` with a 1-D `x` becomes
    unsqueeze, matrix multiply, squeeze — so a `SqueezeBackward` sits in the graph
    recording a reshape you never asked for. The graph is a record of what *ran*, and it
    is built during the forward pass, which is why it can follow an `if` statement and why
    it has to be rebuilt every iteration.
    """)
    return


@app.cell
def _(b, mo, w, x):
    import graphviz

    def backward_graph(root, named):
        """Walk .grad_fn back to the leaves and draw it.

        The three node kinds are told apart by shape first and color second. Hue is a
        selective variable, not an ordered one, and roughly one man in twelve cannot
        separate a red node from a green one; a rectangle, an oval and a doubled outline
        survive that, and survive printing in gray as well.
        """
        dot = graphviz.Digraph()
        dot.attr(bgcolor="transparent", rankdir="BT", margin="8")
        dot.attr("node", style="rounded,filled", fontname="Helvetica", fontsize="11", fontcolor="#15181d")
        dot.attr("edge", color="#8a8f98")
        seen = set()

        def walk(node):
            if node is None or id(node) in seen:
                return
            seen.add(id(node))
            kind = type(node).__name__
            if kind == "AccumulateGrad":
                tensor = node.variable
                label = f"{named.get(id(tensor), 'leaf')}\\n{tuple(tensor.shape)}"
                dot.node(str(id(node)), label, shape="oval", fillcolor="#d5eee4", color="#199e70")
            elif node is root.grad_fn:
                dot.node(
                    str(id(node)),
                    kind.removesuffix("0"),
                    shape="box",
                    peripheries="2",
                    fillcolor="#f7e0d5",
                    color="#d95926",
                )
            else:
                dot.node(str(id(node)), kind.removesuffix("0"), shape="box", fillcolor="#dbe7f7", color="#2a78d6")
            for child, _ in node.next_functions:
                if child is not None:
                    walk(child)
                    dot.edge(str(id(child)), str(id(node)))

        walk(root.grad_fn)
        return dot

    def as_svg(dot):
        svg = dot.pipe(format="svg").decode()
        return mo.Html(
            '<div style="background:white;border-radius:8px;padding:12px;display:inline-block">'
            f"{svg[svg.index('<svg') :]}</div>"
        )

    leaf_names = {id(w): "w", id(b): "b", id(x): "x"}
    return as_svg, backward_graph, leaf_names


@app.cell
def _(as_svg, backward_graph, leaf_names, loss):
    as_svg(backward_graph(loss, leaf_names))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Computing Gradients

    To optimize weights of parameters in the neural network, we need to compute the derivatives of
    our loss function with respect to parameters, namely, we need
    $`\frac{\partial loss}{\partial w}`$ and $`\frac{\partial loss}{\partial b}`$ under some fixed
    values of `x` and `y`. To compute those derivatives, we call `loss.backward()`, and then
    retrieve the values from `w.grad` and `b.grad`:
    """)
    return


@app.cell
def _(b, loss, w):
    loss.backward()
    print(w.grad)
    print(b.grad)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Explore — check that gradient by hand

    `backward()` filled in fifteen numbers and there is no reason yet to believe them.
    This model is small enough to differentiate on paper, so the cell below computes the
    same gradient from the closed form and compares.

    With $`z = xW + b`$ and $`L`$ the mean binary cross entropy over the three outputs, the
    logistic loss collapses to something short:

    $$\frac{\partial L}{\partial z_j} = \frac{\sigma(z_j) - y_j}{3}, \qquad
    \frac{\partial L}{\partial W_{ij}} = x_i \cdot \frac{\partial L}{\partial z_j}, \qquad
    \frac{\partial L}{\partial b_j} = \frac{\partial L}{\partial z_j}$$

    The sigmoid never appears in the code — `binary_cross_entropy_with_logits` fuses it
    into the loss for numerical stability — and it reappears here in the derivative.

    Look at the picture rather than the numbers: every row of `dL/dW` is identical.
    Nothing forced that, except `x = torch.ones(5)`. The gradient with respect to a weight
    is its input times the error arriving at its output, so an input of zero produces a
    gradient of zero no matter how wrong the prediction was — and five identical inputs
    produce five identical rows.
    """)
    return


@app.cell
def _(b, mo, torch, w, x, y, z):
    import altair as alt
    import pandas as pd

    def gradient_map(values, title):
        cells = pd.DataFrame(
            [
                {"column": j, "row": i, "value": float(v)}
                for i, line in enumerate(values.tolist())
                for j, v in enumerate(line)
            ]
        )
        limit = cells["value"].abs().max()
        position = {"x": alt.X("column:O", axis=None), "y": alt.Y("row:O", axis=None)}
        return (
            alt.Chart(cells)
            .mark_rect(stroke="white", strokeWidth=2)
            .encode(
                **position,
                color=alt.Color(
                    "value:Q", scale=alt.Scale(scheme="redblue", domain=[-limit, limit], reverse=True), legend=None
                ),
            )
            + alt.Chart(cells)
            .mark_text(fontSize=11)
            .encode(
                **position,
                text=alt.Text("value:Q", format=".3f"),
                color=alt.condition(f"abs(datum.value) > {0.6 * limit}", alt.value("white"), alt.value("#111111")),
            )
        ).properties(width=3 * 76, height=values.shape[0] * 40, title=title)

    if w.grad is None:
        _panel = mo.callout(
            mo.md(
                "`w.grad` is still empty. Run the `loss.backward()` cell above — it fills "
                "`.grad` in by mutation, which is invisible to the dependency graph, so "
                "marimo cannot run it for you."
            ),
            kind="warn",
        )
    else:
        _error = (torch.sigmoid(z.detach()) - y) / y.numel()
        _analytic_w = torch.outer(x, _error)
        _panel = mo.vstack(
            [
                mo.hstack(
                    [
                        gradient_map(w.grad, "w.grad, from backward()"),
                        gradient_map(_analytic_w, "x ⊗ (σ(z) − y)/3, by hand"),
                    ],
                    justify="start",
                    gap=1.5,
                    wrap=True,
                ),
                mo.md(
                    f"identical to floating-point tolerance: **{torch.allclose(w.grad, _analytic_w)}** for `w`, "
                    f"**{torch.allclose(b.grad, _error)}** for `b`; "
                    f"largest disagreement {(w.grad - _analytic_w).abs().max():.2e}"
                ),
            ]
        )
    _panel
    return alt, gradient_map, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > [!NOTE]
    > - We can only obtain the `grad` properties for the leaf nodes of the computational graph,
    > which have `requires_grad` property set to `True`. For all other nodes in our graph,
    > gradients will not be available. - We can only perform gradient calculations using `backward`
    > once on a given graph, for performance reasons. If we need to do several `backward` calls on
    > the same graph, we need to pass `retain_graph=True` to the `backward` call.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Disabling Gradient Tracking

    By default, all tensors with `requires_grad=True` are tracking their computational history and
    support gradient computation. However, there are some cases when we do not need to do that, for
    example, when we have trained the model and just want to apply it to some input data, i.e. we
    only want to do *forward* computations through the network. We can stop tracking computations
    by surrounding our computation code with `torch.no_grad()` block:
    """)
    return


@app.cell
def _(b, torch, w, x):
    z_1 = torch.matmul(x, w) + b
    print(z_1.requires_grad)
    with torch.no_grad():
        z_1 = torch.matmul(x, w) + b
    print(z_1.requires_grad)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Explore — the same three switches, drawn

    `True` then `False` is the whole result above, which understates it. The switches
    rebuild the identical forward pass and redraw the graph.

    Freeze `w` and more disappears than its own green node: the matrix multiply and the
    squeeze go with it, and the graph drops from six nodes to three. Autograd records an
    operation only when something it consumed needs a gradient, so with `x` and `w` both
    untracked there is nothing to record — the multiply still *runs*, it is simply not
    written down.

    That is what freezing a backbone actually buys. The saving is not one skipped
    accumulation; it is every backward operation below the highest frozen layer, and the
    activations they would have needed kept alive to do it.

    Freeze both and there is no graph at all. `no_grad()` gets to the same place by a
    different route: the tensors still require grad, but nothing is recorded while the
    block is open. The first is a property of the parameters and survives the function
    call; the second is a property of the moment. Inference wants the second, and a frozen
    layer wants the first.
    """)
    return


@app.cell
def _(mo):
    track_w = mo.ui.switch(True, label="`w.requires_grad`")
    track_b = mo.ui.switch(True, label="`b.requires_grad`")
    under_no_grad = mo.ui.switch(False, label="run the forward pass inside `torch.no_grad()`")
    mo.hstack([track_w, track_b, under_no_grad], justify="start", gap=2)
    return track_b, track_w, under_no_grad


@app.cell
def _(as_svg, backward_graph, mo, torch, track_b, track_w, under_no_grad, x, y):
    _w = torch.randn(5, 3, requires_grad=track_w.value)
    _b = torch.randn(3, requires_grad=track_b.value)
    with torch.set_grad_enabled(not under_no_grad.value):
        _z = torch.matmul(x, _w) + _b
        _loss = torch.nn.functional.binary_cross_entropy_with_logits(_z, y)

    if _loss.grad_fn is None:
        _drawing = mo.callout(
            mo.md(
                f"`loss.grad_fn` is `None`: nothing was recorded, so `loss.backward()` would "
                f"raise. `loss` is still a perfectly good number — {_loss.item():.4f} — it "
                "simply has no history."
            ),
            kind="neutral",
        )
    else:
        _drawing = as_svg(backward_graph(_loss, {id(_w): "w", id(_b): "b", id(x): "x"}))
    _drawing
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Another way to achieve the same result is to use the `detach()` method on the tensor:
    """)
    return


@app.cell
def _(b, torch, w, x):
    z_2 = torch.matmul(x, w) + b
    z_det = z_2.detach()
    print(z_det.requires_grad)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There are reasons you might want to disable gradient tracking:
    - To mark some parameters in your neural network as **frozen parameters**.
    - To **speed up computations** when you are only doing forward pass, because computations on
      tensors that do not track gradients would be more efficient.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # More on Computational Graphs

    Conceptually, autograd keeps a record of data (tensors) and all executed operations (along with
    the resulting new tensors) in a directed acyclic graph (DAG) consisting of
    [Function](https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function) objects. In
    this DAG, leaves are the input tensors, roots are the output tensors. By tracing this graph
    from roots to leaves, you can automatically compute the gradients using the chain rule.

    In a forward pass, autograd does two things simultaneously:

    - run the requested operation to compute a resulting tensor
    - maintain the operation’s *gradient function* in the DAG.

    The backward pass kicks off when `.backward()` is called on the DAG root. `autograd` then:

    - computes the gradients from each `.grad_fn`,
    - accumulates them in the respective tensor’s `.grad` attribute
    - using the chain rule, propagates all the way to the leaf tensors.

    > [!NOTE]
    > **DAGs are dynamic in PyTorch** An important thing to note is that the graph is recreated
    > from scratch; after each `.backward()` call, autograd starts populating a new graph. This is
    > exactly what allows you to use control flow statements in your model; you can change the
    > shape, size and operations at every iteration if needed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Optional Reading: Tensor Gradients and Jacobian Products

    In many cases, we have a scalar loss function, and we need to compute the gradient with respect
    to some parameters. However, there are cases when the output function is an arbitrary tensor.
    In this case, PyTorch allows you to compute so-called **Jacobian product**, and not the actual
    gradient.

    For a vector function $`\vec{y}=f(\vec{x})`$, where $`\vec{x}=\langle x_1,\dots,x_n\rangle`$
    and $`\vec{y}=\langle y_1,\dots,y_m\rangle`$, a gradient of $`\vec{y}`$ with respect to
    $`\vec{x}`$ is given by **Jacobian matrix**:

    ``` math
    \begin{aligned}
    J=\left(\begin{array}{ccc}
       \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{1}}{\partial x_{n}}\\
       \vdots & \ddots & \vdots\\
       \frac{\partial y_{m}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
       \end{array}\right)
    \end{aligned}
    ```

    Instead of computing the Jacobian matrix itself, PyTorch allows you to compute **Jacobian
    Product** $`v^T\cdot J`$ for a given input vector $`v=(v_1 \dots v_m)`$. This is achieved by
    calling `backward` with $`v`$ as an argument. The size of $`v`$ should be the same as the size
    of the original tensor, with respect to which we want to compute the product:
    """)
    return


@app.cell
def _(torch):
    inp = torch.eye(4, 5, requires_grad=True)
    out = (inp + 1).pow(2).t()
    out.backward(torch.ones_like(out), retain_graph=True)
    print(f"First call\n{inp.grad}")
    out.backward(torch.ones_like(out), retain_graph=True)
    print(f"\nSecond call\n{inp.grad}")
    inp.grad.zero_()
    out.backward(torch.ones_like(out), retain_graph=True)
    print(f"\nCall after zeroing gradients\n{inp.grad}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice that when we call `backward` for the second time with the same argument, the value of
    the gradient is different. This happens because when doing `backward` propagation, PyTorch
    **accumulates the gradients**, i.e. the value of computed gradients is added to the `grad`
    property of all leaf nodes of computational graph. If you want to compute the proper gradients,
    you need to zero out the `grad` property before. In real-life training an *optimizer* helps us
    to do this.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > [!NOTE]
    > Previously we were calling `backward()` function without parameters. This is essentially
    > equivalent to calling `backward(torch.tensor(1.0))`, which is a useful way to compute the
    > gradients in case of a scalar-valued function, such as loss during neural network training.
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
    # Further Reading

    - [Autograd Mechanics](https://pytorch.org/docs/stable/notes/autograd.html)
    """)
    return


if __name__ == "__main__":
    app.run()
