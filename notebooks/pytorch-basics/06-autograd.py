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
    *PyTorch basics, 6 of 8 — before this: [Build Model](05-build-model.py) · after:
    [Optimization](07-optimization-loop.py)*

    # Automatic Differentiation with `torch.autograd`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Today's target** — run the notebook; consuming it means having watched `requires_grad`,
    > `backward()` and `.grad` produce a gradient — and accumulate one when nothing resets it.
    >
    > **Marc's depth line** — the other half of the base Marc named: understanding how the gradient
    > works. His own checks: `zero_grad` leaves every gradient zero; two backward passes on one
    > batch without resetting is twice the gradient. Half a day here is fine.
    >
    > **Stop-line** — done means: ran it, could explain to Marc where a gradient lives and when it
    > accumulates, questions captured — close it.
    >
    > **Capture** — `scripts/q "your question"` appends it to Friday's file for Marc.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The last notebook ended on a flag: every parameter in its table carried `trainable = True` —
    the `requires_grad` property — and nothing had happened because of it yet. This notebook is
    where the flag does its work. Training adjusts parameters by **back propagation**: each weight
    moves according to the **gradient** of the loss with respect to that weight, and PyTorch
    computes those gradients with a built-in differentiation engine, `torch.autograd`, which can
    differentiate through any computational graph.

    The whole mechanism fits in a one-layer network — input `x`, parameters `w` and `b`, and a
    loss to differentiate:
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
    ## Tensors, Functions and Computational graph

    Running that code did more than produce a number. `w` and `b` are **parameters** — the tensors
    training will adjust, so the gradients of the loss with respect to them are what we are after,
    and setting their `requires_grad` property is how we ask for them. While the forward pass ran,
    autograd recorded every operation that touched a tracked tensor into a **computational
    graph** — the record it will differentiate, drawn from the live objects a few cells down.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            "You can set the value of `requires_grad` when creating a tensor, or later by "
            "using the `x.requires_grad_(True)` method."
        ),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A function that we apply to tensors to construct the computational graph is in fact an object
    of class `Function`. This object knows how to compute the function in the *forward* direction,
    and also how to compute its derivative during the *backward propagation* step. A reference to
    the backward propagation function is stored in the `grad_fn` property of a tensor. You can
    find more information on `Function` [in the
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
    ### The graph itself, walked and drawn

    The two printed nodes are the tip of something reachable: every tensor autograd produced
    carries `.grad_fn`, and every `grad_fn` carries `.next_functions` — the edges to whatever
    produced its inputs. Walking that from `loss` back to the leaves is a dozen lines; drawing
    what the walk finds is plumbing, folded below it.

    The double-outlined box at the top is the root, the tensor `backward()` gets called on. The
    ovals at the bottom are `AccumulateGrad`: the ends of the road, one per leaf tensor with
    `requires_grad=True`, and the only places a `.grad` is ever written. The plain boxes between
    them are the recorded operations. The forward pass built the picture bottom-up; `backward()`
    will run it top-down, root to leaves.

    One node is not in the code you wrote. `torch.matmul(x, w)` with a 1-D `x` becomes unsqueeze,
    matrix multiply, squeeze — so a `SqueezeBackward4` sits in the graph recording a reshape you
    never asked for. Its matching unsqueeze is absent: it touched only `x`, which tracks nothing,
    so it was never recorded — a rule the switches further down make vivid. The graph is a record
    of what *ran*, built during the forward pass, which is why it can follow an `if` statement and
    why it has to be rebuilt every iteration.
    """)
    return


@app.cell
def _():
    def graph_of(root):
        """Walk .grad_fn to the leaves: a node reaches its producers via .next_functions."""
        nodes, edges, frontier = {}, [], [root.grad_fn]
        while frontier:
            node = frontier.pop()
            if id(node) in nodes:
                continue
            nodes[id(node)] = node
            for producer, _ in node.next_functions:
                if producer is not None:
                    edges.append((producer, node))
                    frontier.append(producer)
        return nodes, edges

    return (graph_of,)


@app.cell(hide_code=True)
def _(graph_of, mo):
    # The viewing vocabulary is shared with the sibling notebooks and lives in _viz.py.
    # mo.mermaid is marimo's native graph surface: it sizes to its content and follows
    # the reader's theme, where raw graphviz SVG arrived at poster size (graphviz margins
    # are in INCHES) on a hand-rolled white card. Node fills stay computed from the
    # palette constants; each node carries its own light fill, so the label ink is the
    # calibrated dark regardless of the page.
    from _palette import tint
    from _viz import ACCENT, BASE, INK_DARK, OKABE_ITO, show

    def draw(root, named):
        """Hand graph_of's findings to mermaid.

        The three node kinds are told apart by shape first and color second: a rectangle,
        a stadium and a doubled outline survive color-vision deficiency, and survive
        printing in gray as well. Labels are the exact class names, so the picture
        matches what `type(node).__name__` prints.
        """
        nodes, edges = graph_of(root)
        green = OKABE_ITO["green"]
        lines = ["flowchart BT"]
        for node in nodes.values():
            kind = type(node).__name__
            nid = f"n{id(node)}"
            if kind == "AccumulateGrad":
                label = f"{named.get(id(node.variable), 'leaf')}<br/>{tuple(node.variable.shape)}"
                lines.append(f'{nid}(["{label}"]):::leaf')
            elif node is root.grad_fn:
                lines.append(f'{nid}[["{kind}"]]:::root')
            else:
                lines.append(f'{nid}["{kind}"]:::op')
        lines += [f"n{id(p)} --> n{id(c)}" for p, c in edges]
        lines += [
            f"classDef leaf fill:{tint(green, 0.82)},stroke:{green},color:{INK_DARK}",
            f"classDef root fill:{tint(ACCENT, 0.82)},stroke:{ACCENT},color:{INK_DARK}",
            f"classDef op fill:{tint(BASE, 0.82)},stroke:{BASE},color:{INK_DARK}",
            f"linkStyle default stroke:{tint(INK_DARK, 0.45)}",
        ]
        return mo.mermaid("\n".join(lines))

    return draw, show


@app.cell(hide_code=True)
def _(b, draw, loss, w):
    draw(loss, {id(w): "w", id(b): "b"})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Computing Gradients

    To optimize the parameters, we need the derivatives of the loss function with respect to
    them: $\frac{\partial loss}{\partial w}$ and $\frac{\partial loss}{\partial b}$ under the
    fixed values of `x` and `y`. To compute those derivatives, we call `loss.backward()`, and
    then retrieve the values from `w.grad` and `b.grad`:
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
    ### Check that gradient by hand

    `backward()` filled in fifteen numbers and there is no reason yet to believe them. This model
    is small enough to differentiate on paper, so the cell below computes the same gradient from
    the closed form and compares.

    With $z = xW + b$ and $L$ the mean binary cross entropy over the three outputs, the logistic
    loss collapses to something short:

    $$\frac{\partial L}{\partial z_j} = \frac{\sigma(z_j) - y_j}{3}, \qquad
    \frac{\partial L}{\partial W_{ij}} = x_i \cdot \frac{\partial L}{\partial z_j}, \qquad
    \frac{\partial L}{\partial b_j} = \frac{\partial L}{\partial z_j}$$

    The sigmoid never appears in the code — `binary_cross_entropy_with_logits` fuses it into the
    loss for numerical stability — and it reappears here in the derivative.
    """)
    return


@app.cell
def _(torch, x, y, z):
    error = (torch.sigmoid(z.detach()) - y) / y.numel()  # dL/dz, one entry per output
    analytic_w = torch.outer(x, error)  # each weight's gradient: its input times the error at its output
    return analytic_w, error


@app.cell(hide_code=True)
def _(analytic_w, b, error, mo, show, torch, w):
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
        _panel = mo.vstack(
            [
                mo.hstack(
                    [
                        show(w.grad, "w.grad, from backward()"),
                        show(analytic_w, "x ⊗ (σ(z) − y)/3, by hand"),
                    ],
                    justify="start",
                    gap=1.5,
                    wrap=True,
                ),
                mo.md(
                    f"identical to floating-point tolerance: **{torch.allclose(w.grad, analytic_w)}** for `w`, "
                    f"**{torch.allclose(b.grad, error)}** for `b`; "
                    f"largest disagreement {(w.grad - analytic_w).abs().max():.2e}"
                ),
            ]
        )
    _panel
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Look at the picture rather than the numbers: every row of `dL/dW` is identical. Nothing forced
    that, except `x = torch.ones(5)`. The gradient with respect to a weight is its input times the
    error arriving at its output, so an input of zero produces a gradient of zero no matter how
    wrong the prediction was — and five identical inputs produce five identical rows.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""
    - We can only obtain the `grad` properties for the leaf nodes of the computational graph,
      which have the `requires_grad` property set to `True`. For all other nodes in our graph,
      gradients will not be available.
    - We can only perform gradient calculations using `backward` once on a given graph, for
      performance reasons: the backward pass frees the graph's saved tensors as it consumes
      them, and a second call raises. If we do need several `backward` calls on the same graph,
      we pass `retain_graph=True` to the `backward` call — the optional-reading section at the
      bottom does exactly that.
    """),
        kind="warn",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Disabling Gradient Tracking

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
    ### The same three switches, drawn

    `True` then `False` is the whole result above, which understates it. The switches below
    rebuild the identical forward pass and redraw the graph — predict what each flip removes
    before flipping it.

    Freeze `w` and more disappears than its own oval: the matrix multiply and the squeeze go with
    it, and the graph drops from six nodes to three. Autograd records an operation only when
    something it consumed needs a gradient, so with `x` and `w` both untracked there is nothing to
    record — the multiply still *runs*, it is simply not written down.

    That is what freezing a backbone actually buys. The saving is not one skipped accumulation; it
    is every backward operation below the highest frozen layer, and the activations they would
    have needed kept alive to do it.

    Freeze both and there is no graph at all. `no_grad()` gets to the same place by a different
    route: the tensors still require grad, but nothing is recorded while the block is open. The
    first is a property of the parameters and survives the function call; the second is a property
    of the moment. Inference wants the second, and a frozen layer wants the first.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    track_w = mo.ui.switch(True, label="`w.requires_grad`")
    track_b = mo.ui.switch(True, label="`b.requires_grad`")
    under_no_grad = mo.ui.switch(False, label="run the forward pass inside `torch.no_grad()`")
    mo.hstack([track_w, track_b, under_no_grad], justify="start", gap=2)
    return track_b, track_w, under_no_grad


@app.cell
def _(torch, track_b, track_w, under_no_grad, x, y):
    w_live = torch.randn(5, 3, requires_grad=track_w.value)
    b_live = torch.randn(3, requires_grad=track_b.value)
    with torch.set_grad_enabled(not under_no_grad.value):
        loss_live = torch.nn.functional.binary_cross_entropy_with_logits(torch.matmul(x, w_live) + b_live, y)
    return b_live, loss_live, w_live


@app.cell(hide_code=True)
def _(b_live, draw, loss_live, mo, w_live):
    if loss_live.grad_fn is None:
        _drawing = mo.callout(
            mo.md(
                f"`loss.grad_fn` is `None`: nothing was recorded, so `loss.backward()` would "
                f"raise. `loss` is still a perfectly good number — {loss_live.item():.4f} — it "
                "simply has no history."
            ),
            kind="neutral",
        )
    else:
        _drawing = draw(loss_live, {id(w_live): "w", id(b_live): "b"})
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
    ## More on Computational Graphs

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
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            "**DAGs are dynamic in PyTorch.** The graph is recreated from scratch: after "
            "each `.backward()` call, autograd starts populating a new graph. This is "
            "exactly what allows you to use control flow statements in your model; you can "
            "change the shape, size and operations at every iteration if needed."
        ),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Optional Reading: Tensor Gradients and Jacobian Products

    In many cases, we have a scalar loss function, and we need to compute the gradient with respect
    to some parameters. However, there are cases when the output function is an arbitrary tensor.
    In this case, PyTorch allows you to compute a so-called **Jacobian product**, and not the
    actual gradient.

    For a vector function $\vec{y}=f(\vec{x})$, where $\vec{x}=\langle x_1,\dots,x_n\rangle$ and
    $\vec{y}=\langle y_1,\dots,y_m\rangle$, a gradient of $\vec{y}$ with respect to $\vec{x}$ is
    given by the **Jacobian matrix**:

    $$J=\left(\begin{array}{ccc}
    \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{1}}{\partial x_{n}}\\
    \vdots & \ddots & \vdots\\
    \frac{\partial y_{m}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
    \end{array}\right)$$

    Instead of computing the Jacobian matrix itself, PyTorch allows you to compute the **Jacobian
    product** $v^T\cdot J$ for a given input vector $v=(v_1 \dots v_m)$. This is achieved by
    calling `backward` with $v$ as an argument. The size of $v$ should be the same as the size of
    the original tensor, with respect to which we want to compute the product.

    The cell calls `backward` three times on one retained graph — guess what the second print
    shows before reading past it:
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
    **accumulates the gradients**: the value of computed gradients is added to the `grad` property
    of all leaf nodes of the computational graph. If you want to compute the proper gradients, you
    need to zero out the `grad` property before. This is Marc's second check, passed by exactly
    doubled numbers — and it is not a demo quirk: in real training an *optimizer* zeroes the
    gradients every iteration, and the next notebook's loop shows where that call sits.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            "Previously we were calling the `backward()` function without parameters. This "
            "is essentially equivalent to calling `backward(torch.tensor(1.0))`, which is a "
            "useful way to compute the gradients in case of a scalar-valued function, such "
            "as loss during neural network training."
        ),
        kind="info",
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
    ## Where to go next

    - [Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html) documents what
      this notebook skipped: how in-place writes on saved tensors are caught (a version counter
      the backward pass checks), what participates in the graph under mixed `requires_grad`, and
      hooks for watching gradients flow through any node.
    - `torch.autograd.grad(loss, [w, b])` returns the same gradients as values instead of
      accumulating them into `.grad` — the functional door to everything here, and the one
      higher-order derivatives go through.
    - Next: [Optimization](07-optimization-loop.py), where these pieces start running in a loop —
      forward, `backward()`, a step downhill — and `optimizer.zero_grad()` runs every iteration,
      because gradients accumulate and nothing else resets them.
    """)
    return


if __name__ == "__main__":
    app.run()
