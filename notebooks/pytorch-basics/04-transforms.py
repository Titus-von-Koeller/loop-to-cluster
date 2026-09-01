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
    *PyTorch basics, 4 of 8 — before this: [Datasets & DataLoaders](03-datasets-and-dataloaders.py)
    · after: [Build Model](05-build-model.py)*

    # Transforms
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Today's target** — run the notebook; consuming it means having seen what `transform` and
    > `target_transform` do to a sample on its way in — pixels to scaled floats, labels to one-hot.
    >
    > **Marc's depth line** — Marc priced this one at "could be interesting" and moved on; it is
    > not on his focus map — a skim is fine.
    >
    > **Stop-line** — done means: ran it, could explain to Marc why features and labels each get
    > their own transform, questions captured — close it.
    >
    > **Capture** — `scripts/q "your question"` appends it to Friday's file for Marc.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The last notebook passed `transform=v2.Compose([...])` to every dataset it built, and even
    measured what it costs per sample — without ever saying what it does. This notebook is about
    that argument. Training wants scaled float tensors; on disk sit PIL images and integer
    labels; **transforms** are the callables that close the gap, applied inside `__getitem__` on
    every access.

    Every TorchVision dataset takes two of them: `transform` for the features and
    `target_transform` for the labels. The
    [torchvision.transforms](https://pytorch.org/vision/stable/transforms.html) module ships the
    common ones. Below, the `torchvision.transforms.v2` API scales the images to `[0, 1]` floats,
    and `torch.nn.functional.one_hot` turns each integer label into a ten-float vector.
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn.functional as F
    from torchvision import datasets
    from torchvision.transforms import v2

    ds = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        target_transform=v2.Lambda(lambda y: F.one_hot(torch.tensor(y), num_classes=10).float()),
    )
    return F, datasets, ds, torch, v2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ToImage() and ToDtype()

    The `torchvision.transforms.v2` API replaces the legacy `ToTensor` transform with a two-step
    pipeline.
    [v2.ToImage](https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.ToImage.html)
    converts a PIL image or NumPy `ndarray` into a `torchvision.tv_tensors.Image` tensor, and
    [v2.ToDtype](https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.ToDtype.html)
    with `scale=True` casts it to `float32` and scales the pixel intensity values to the range
    \[0., 1.\].
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The same picture, three times, in different numbers

    Below is image 7 of the training set at each stage of that two-step pipeline, with what
    it actually *is* at each stage underneath.

    The three pictures are identical, and that is the point worth taking: rendering
    normalizes, so the eye cannot tell 0-255 from 0.0-1.0. Only the numbers can. Forgetting
    `scale=True` is therefore invisible everywhere except the loss — and there it is
    unmistakable if you know the number to expect. A ten-class model at random
    initialization should start at ln(10) = 2.303, because it is guessing uniformly. With
    `scale=True` this network starts at 2.305. Fed the same images unscaled, it starts at
    18.2: the first layer's outputs are 255 times larger, the logits are saturated, and
    the model spends its early steps climbing back to ignorance.
    """)
    return


@app.cell
def _(datasets, mo, torch, v2):
    _raw_image, _raw_label = datasets.FashionMNIST(root="data", train=True, download=False)[7]
    _as_tensor = v2.ToImage()(_raw_image)
    _as_float = v2.ToDtype(torch.float32, scale=True)(_as_tensor)

    _stages = [
        ("the file", _raw_image, f"`{type(_raw_image).__name__}`, mode {_raw_image.mode}, 28x28 pixels"),
        (
            "after `ToImage()`",
            _as_tensor.squeeze(0),
            f"`{_as_tensor.dtype}` {tuple(_as_tensor.shape)}, values {_as_tensor.min()}–{_as_tensor.max()}",
        ),
        (
            "after `ToDtype(scale=True)`",
            _as_float.squeeze(0),
            f"`{_as_float.dtype}` {tuple(_as_float.shape)}, values {_as_float.min():.1f}–{_as_float.max():.1f}",
        ),
    ]
    mo.hstack(
        [
            mo.vstack(
                [mo.md(f"**{name}**"), mo.image(image, width=140, rounded=True), mo.md(f"<small>{caption}</small>")],
                align="center",
                gap=0.3,
            )
            for name, image, caption in _stages
        ],
        justify="start",
        gap=2,
        wrap=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Lambda transforms

    A `v2.Lambda` wraps any user-defined callable into a transform. The `target_transform=` line
    in the loading cell at the top uses one to apply
    [torch.nn.functional.one_hot](https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html),
    turning the integer label into a one-hot tensor of size 10 (the number of classes), then
    casting it to `float` to match the expected dtype.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What the label became

    `ds` above was built with that `target_transform`, so its labels are not integers any
    more. Move the slider and watch the second half of the pair.
    """)
    return


@app.cell
def _(mo):
    label_sample = mo.ui.slider(0, 999, value=7, label="sample", show_value=True)
    label_sample
    return (label_sample,)


@app.cell(hide_code=True)
def _(ds, label_sample, mo):
    _image, _label = ds[label_sample.value]
    _hot = int(_label.argmax())
    _names = [
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
    _cells = " ".join(
        f"**`{value:.0f}`**" if index == _hot else f"`{value:.0f}`" for index, value in enumerate(_label.tolist())
    )
    mo.hstack(
        [
            mo.image(_image.squeeze(0), width=110, vmin=0, vmax=1, rounded=True),
            mo.vstack(
                [
                    mo.md(f"label as stored: `{_hot}` — {_names[_hot]}"),
                    mo.md(f"label as the model receives it: {_cells}"),
                    mo.md(f"`dtype={_label.dtype}`, `shape={tuple(_label.shape)}`"),
                ],
                gap=0.4,
            ),
        ],
        justify="start",
        align="center",
        gap=2,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ten floats where one integer would do, and `nn.CrossEntropyLoss` accepts either: given
    class indices it looks the right logit up, given a probability vector it takes the full
    cross entropy. For a one-hot vector those are the same arithmetic and the same number to
    the last decimal — so this transform costs a tensor per sample and buys nothing here. It
    earns its place when the target stops being one-hot: label smoothing, mixup and
    distillation all hand the loss a vector that is not a single 1.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A transform pipeline with the parameters exposed

    Every transform in the v2 API has a functional twin: `RandomRotation` picks an angle
    and calls `functional.rotate`, so driving `rotate` from a slider is the same operation
    with the die replaced by your hand. Drag the divider on the image to compare.

    Two things become obvious by moving these that no amount of reading gives you. The
    geometry transforms have to invent pixels at the edges, and they fill them with black
    — which on FashionMNIST is indistinguishable from background, and on a dataset with a
    light background would be a black frame the model could learn to read. And a
    28x28 image has so little to spare that a scale of 1.4 already pushes a boot out of
    frame: augmentation strength is bounded by resolution, not by taste.
    """)
    return


@app.cell
def _(mo):
    sample = mo.ui.slider(0, 999, value=7, label="image", show_value=True)
    angle = mo.ui.slider(-90.0, 90.0, step=1.0, value=0.0, label="rotate °", show_value=True)
    shift_x = mo.ui.slider(-10, 10, value=0, label="translate x", show_value=True)
    shift_y = mo.ui.slider(-10, 10, value=0, label="translate y", show_value=True)
    zoom = mo.ui.slider(0.4, 2.0, step=0.05, value=1.0, label="scale", show_value=True)
    slant = mo.ui.slider(-30.0, 30.0, step=1.0, value=0.0, label="shear °", show_value=True)
    brightness = mo.ui.slider(0.2, 2.0, step=0.05, value=1.0, label="brightness", show_value=True)
    contrast = mo.ui.slider(0.0, 2.0, step=0.05, value=1.0, label="contrast", show_value=True)
    blur = mo.ui.slider(0.0, 3.0, step=0.1, value=0.0, label="blur σ", show_value=True)
    flip = mo.ui.switch(False, label="horizontal flip")

    mo.hstack(
        [
            mo.vstack([sample, angle, shift_x, shift_y]),
            mo.vstack([zoom, slant, brightness, contrast]),
            mo.vstack([blur, flip]),
        ],
        justify="start",
        gap=2,
        wrap=True,
    )
    return angle, blur, brightness, contrast, flip, sample, shift_x, shift_y, slant, zoom


@app.cell
def _(
    angle,
    blur,
    brightness,
    contrast,
    ds,
    flip,
    mo,
    sample,
    shift_x,
    shift_y,
    slant,
    torch,
    zoom,
):
    from torchvision.transforms.v2 import functional as VF

    def as_pixels(image):
        """Freeze an image to uint8 so two renderings share one brightness scale.

        `mo.image` stretches a float array to fill 0-255, which would silently undo the
        brightness slider: the before and after would be normalized apart.
        """
        return (image.clamp(0, 1) * 255).to(torch.uint8).squeeze(0)

    original = ds[sample.value][0]
    transformed = VF.affine(
        original,
        angle=angle.value,
        translate=[shift_x.value, shift_y.value],
        scale=zoom.value,
        shear=[slant.value, 0.0],
    )
    if flip.value:
        transformed = VF.horizontal_flip(transformed)
    transformed = VF.adjust_contrast(VF.adjust_brightness(transformed, brightness.value), contrast.value)
    if blur.value > 0:
        transformed = VF.gaussian_blur(transformed, kernel_size=[5, 5], sigma=blur.value)

    mo.vstack(
        [
            mo.image_compare(as_pixels(original), as_pixels(transformed), width=320),
            mo.md(
                f"<small>left: untouched · right: transformed · "
                f"{(transformed - original).abs().mean():.3f} mean absolute change per pixel</small>"
            ),
        ],
        align="center",
    )
    return VF, as_pixels, original


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The same transform, applied eight times

    Nothing above is random: each slider fixes a parameter, so the output is a function of
    the input. An augmentation is the same operation with the parameter drawn fresh on
    every call — and `transform=` is called inside `__getitem__`, once per access, not
    once per dataset. So these eight are eight different images from *one* stored image,
    and a model training on this dataset never sees the same input twice.

    That is the whole mechanism of augmentation as regularization, and it is also the
    reason a dataset with a random transform cannot be cached naively, and the reason two
    processes with different seeds see different data for the same index — which becomes a
    correctness question the moment training is distributed.
    """)
    return


@app.cell
def _(as_pixels, mo, original, torch, v2):
    _augment = v2.Compose(
        [
            v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.85, 1.15)),
            v2.RandomHorizontalFlip(p=0.5),
        ]
    )
    torch.manual_seed(0)
    mo.hstack(
        [mo.image(as_pixels(_augment(original)), width=84, rounded=True) for _ in range(8)],
        justify="start",
        gap=0.6,
        wrap=True,
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

    - The [v2 gallery](https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_getting_started.html)
      and [v2 API](https://pytorch.org/vision/stable/transforms.html#v2-api-reference-recommended)
      cover what this notebook skipped: `Normalize` (mean/std standardization, the usual third
      step after scaling), `CutMix`/`MixUp`, and the same transforms on detection boxes and video.
    - v2 transforms are not tied to `__getitem__`: handed a batched tensor on the GPU —
      `aug(torch.rand(64, 1, 28, 28, device="cuda"))` — they transform the whole batch on the
      device. That is the escape hatch when per-sample CPU transforms become the loading
      bottleneck the last notebook measured.
    - Next: [Build Model](05-build-model.py), where the scaled tensors finally meet parameters.
    """)
    return


if __name__ == "__main__":
    app.run()
