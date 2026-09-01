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
    [Tensors](tensorqs_tutorial.html) \|\| **Datasets & DataLoaders** \|\|
    [Transforms](transforms_tutorial.html) \|\| [Build Model](buildmodel_tutorial.html) \|\|
    [Autograd](autogradqs_tutorial.html) \|\| [Optimization](optimization_tutorial.html) \|\| [Save
    & Load Model](saveloadrun_tutorial.html)

    # Datasets & DataLoaders
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Today's target** — run the notebook; consuming it means having watched a `Dataset` hand out
    > single samples and a `DataLoader` turn them into shuffled batches.
    >
    > **Marc's depth line** — Marc: "really, the part which is interesting is the data loader" —
    > worth real time here; the datasets half stays light, Hugging Face has its own.
    >
    > **Stop-line** — done means: ran it, could explain to Marc what `DataLoader` adds on top of a
    > `Dataset`, questions captured — close it.
    >
    > **Capture** — `scripts/q "your question"` appends it to Friday's file for Marc.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Code for processing data samples can get messy and hard to maintain; we ideally want our
    dataset code to be decoupled from our model training code for better readability and
    modularity. PyTorch provides two data primitives: `torch.utils.data.DataLoader` and
    `torch.utils.data.Dataset` that allow you to use pre-loaded datasets as well as your own data.
    `Dataset` stores the samples and their corresponding labels, and `DataLoader` wraps an iterable
    around the `Dataset` to enable easy access to the samples.

    PyTorch domain libraries provide a number of pre-loaded datasets (such as FashionMNIST) that
    subclass `torch.utils.data.Dataset` and implement functions specific to the particular data.
    They can be used to prototype and benchmark your model. You can find them here: [Image
    Datasets](https://pytorch.org/vision/stable/datasets.html), [Text
    Datasets](https://pytorch.org/text/stable/datasets.html), and [Audio
    Datasets](https://pytorch.org/audio/stable/datasets.html)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Loading a Dataset

    Here is an example of how to load the
    [Fashion-MNIST](https://research.zalando.com/project/fashion_mnist/fashion_mnist/) dataset from
    TorchVision. Fashion-MNIST is a dataset of Zalando’s article images consisting of 60,000
    training examples and 10,000 test examples. Each example comprises a 28×28 grayscale image and
    an associated label from one of 10 classes.

    We load the [FashionMNIST Dataset](https://pytorch.org/vision/stable/datasets.html#fashion-mnist)
    with the following parameters:
    - `root` is the path where the train/test data is stored,
    - `train` specifies training or test dataset,
    - `download=True` downloads the data from the internet if it's not available at `root`.
    - `transform` and `target_transform` specify the feature and label transformations
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import torch
    from torch.utils.data import Dataset
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
    return Dataset, plt, test_data, torch, training_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Iterating and Visualizing the Dataset

    We can index `Datasets` manually like a list: `training_data[index]`. We use `matplotlib` to
    visualize some samples in our training data.
    """)
    return


@app.cell
def _(plt, torch, training_data):
    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }
    figure = plt.figure(figsize=(8, 8))
    cols, rows = (3, 3)
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        _img, _label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[_label])
        plt.axis("off")
        plt.imshow(_img.squeeze(), cmap="gray")
    plt.show()
    return (labels_map,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The dataset as something you can page through

    Nine random samples per re-run is enough to prove the dataset loads, and not enough to
    give you a feel for it. Pick classes and page through them instead. Sandal, Sneaker
    and Ankle boot are worth putting side by side: at 28x28 in grayscale they are three
    silhouettes with the same footprint, and the model that struggles with them in the
    quickstart notebook is not being stupid.
    """)
    return


@app.cell
def _(labels_map, mo, training_data):
    wanted = mo.ui.multiselect(
        options=dict(sorted((name, index) for index, name in labels_map.items())),
        value=["Sandal", "Sneaker", "Ankle Boot"],
        label="classes",
    )
    page = mo.ui.slider(0, 40, value=0, label="page", show_value=True)
    mo.hstack([wanted, page, mo.md(f"{len(training_data):,} training images")], justify="start", gap=1)
    return page, wanted


@app.cell
def _(labels_map, mo, page, torch, training_data, wanted):
    _targets = training_data.targets
    _keep = torch.isin(_targets, torch.tensor(wanted.value or list(labels_map)))
    _matching = _keep.nonzero().flatten()
    _start = min(page.value * 12, max(len(_matching) - 12, 0))
    _cards = [
        mo.vstack(
            [
                mo.image(training_data[i][0].squeeze(0), width=84, vmin=0, vmax=1, rounded=True),
                mo.md(f"<small>{labels_map[int(_targets[i])]}</small>"),
            ],
            align="center",
            gap=0.2,
        )
        for i in _matching[_start : _start + 12].tolist()
    ]
    mo.vstack(
        [
            mo.md(f"**{len(_matching):,}** images in the selected classes."),
            mo.hstack(_cards, justify="start", wrap=True, gap=0.8),
        ]
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
    # Creating a Custom Dataset for your files

    A custom Dataset class must implement three functions:
    <span class="title-ref">\_\_init\_\_</span>, <span class="title-ref">\_\_len\_\_</span>, and
    <span class="title-ref">\_\_getitem\_\_</span>. Take a look at this implementation; the
    FashionMNIST images are stored in a directory `img_dir`, and their labels are stored separately
    in a CSV file `annotations_file`.

    In the next sections, we'll break down what's happening in each of these functions.
    """)
    return


@app.cell
def _(Dataset):
    import os

    import pandas as pd
    from torchvision.io import decode_image

    class CustomImageDataset(Dataset):
        def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
            self.img_labels = pd.read_csv(annotations_file)
            self.img_dir = img_dir
            self.transform = transform
            self.target_transform = target_transform

        def __len__(self):
            return len(self.img_labels)

        def __getitem__(self, idx):
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
            image = decode_image(img_path)
            _label = self.img_labels.iloc[idx, 1]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                _label = self.target_transform(_label)
            return (image, _label)

    return decode_image, os, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # `__init__`

    The \_\_init\_\_ function is run once when instantiating the Dataset object. We initialize the
    directory containing the images, the annotations file, and both transforms (covered in more
    detail in the next section).

    The labels.csv file looks like: :

        tshirt1.jpg, 0
        tshirt2.jpg, 0
        ......
        ankleboot999.jpg, 9
    """)
    return


@app.cell
def _(pd):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # `__len__`

    The \_\_len\_\_ function returns the number of samples in our dataset.

    Example:
    """)
    return


@app.cell
def _():
    def __len__(self):
        return len(self.img_labels)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # `__getitem__`

    The \_\_getitem\_\_ function loads and returns a sample from the dataset at the given index
    `idx`. Based on the index, it identifies the image's location on disk, converts that to a
    tensor using `decode_image`, retrieves the corresponding label from the csv data in
    `self.img_labels`, calls the transform functions on them (if applicable), and returns the
    tensor image and corresponding label in a tuple.
    """)
    return


@app.cell
def _(decode_image, os):
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        _label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            _label = self.target_transform(_label)
        return (image, _label)

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
    # Preparing your data for training with DataLoaders

    The `Dataset` retrieves our dataset's features and labels one sample at a time. While training
    a model, we typically want to pass samples in "minibatches", reshuffle the data at every epoch
    to reduce model overfitting, and use Python's `multiprocessing` to speed up data retrieval.

    `DataLoader` is an iterable that abstracts this complexity for us in an easy API.
    """)
    return


@app.cell
def _(test_data, training_data):
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    return DataLoader, train_dataloader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Iterate through the DataLoader

    We have loaded that dataset into the `DataLoader` and can iterate through the dataset as
    needed. Each iteration below returns a batch of `train_features` and `train_labels` (containing
    `batch_size=64` features and labels respectively). Because we specified `shuffle=True`, after
    we iterate over all batches the data is shuffled (for finer-grained control over the data
    loading order, take a look at
    [Samplers](https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler)).
    """)
    return


@app.cell
def _(plt, train_dataloader):
    # Display image and label.
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    _img = train_features[0].squeeze()
    _label = train_labels[0]
    plt.imshow(_img, cmap="gray")
    plt.show()
    print(f"Label: {_label}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The three arguments that decide what a batch is

    `DataLoader(training_data, batch_size=64, shuffle=True)` hides three separate
    decisions. Change them and watch what arrives.

    - **batch_size** sets the first dimension of every tensor the model sees. It is also
      the number that `len(dataloader)` counts *batches* of, not samples — an off-by-a-
      factor-of-64 error waiting to happen in a progress bar.
    - **shuffle** decides whether the sampler walks the dataset in file order or permutes
      it each epoch. FashionMNIST is already stored well mixed — all ten classes appear in
      the first 64 images — so the class counts below barely move when you turn it off.
      What does change is that every epoch then sees the identical sequence of batches,
      and consecutive gradients stay correlated in whatever way the file happens to be
      ordered. On a dataset stored grouped by class, and many are, that is the difference
      between training and not.
    - **drop_last** decides what happens to the remainder. 60,000 images in batches of 64
      leaves a final batch of 32, since 60,000 = 937 × 64 + 32. Keeping it means one batch
      has a different shape from all the others; dropping it means throwing away those 32
      images every epoch.

    That last one stops being cosmetic once there is more than one GPU: ranks that
    disagree about how many batches exist will hang at the collective that ends the epoch,
    which is why distributed training usually drops the remainder.
    """)
    return


@app.cell
def _(mo):
    batch_size_choice = mo.ui.slider(steps=[1, 8, 16, 32, 64, 128, 256], value=64, label="batch_size", show_value=True)
    shuffle_choice = mo.ui.switch(True, label="shuffle")
    drop_last_choice = mo.ui.switch(False, label="drop_last")
    mo.hstack([batch_size_choice, shuffle_choice, drop_last_choice], justify="start", gap=2)
    return batch_size_choice, drop_last_choice, shuffle_choice


@app.cell
def _(
    DataLoader,
    batch_size_choice,
    drop_last_choice,
    labels_map,
    mo,
    shuffle_choice,
    torch,
    training_data,
):
    explore_loader = DataLoader(
        training_data,
        batch_size=batch_size_choice.value,
        shuffle=shuffle_choice.value,
        drop_last=drop_last_choice.value,
        generator=torch.Generator().manual_seed(0),
    )
    _images, _labels = next(iter(explore_loader))
    _remainder = len(training_data) % batch_size_choice.value
    _last = "same as the rest" if _remainder == 0 or drop_last_choice.value else f"{_remainder} images"

    _strip = [
        mo.vstack(
            [
                mo.image(image.squeeze(0), width=52, vmin=0, vmax=1, rounded=True),
                mo.md(f"<small>{labels_map[int(label)][:8]}</small>"),
            ],
            align="center",
            gap=0.1,
        )
        for image, label in zip(_images[:24], _labels[:24], strict=True)
    ]
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(f"{len(explore_loader):,}", label="batches per epoch", bordered=True),
                    mo.stat(str(tuple(_images.shape)), label="shape of one batch", bordered=True),
                    mo.stat(_last, label="last batch", bordered=True),
                    mo.stat(f"{len(_labels.unique())}/10", label="classes in this batch", bordered=True),
                ],
                justify="start",
                gap=1,
            ),
            mo.md("The first batch, up to the first 24 images:"),
            mo.hstack(_strip, justify="start", wrap=True, gap=0.5),
        ],
        gap=1,
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
    # Further Reading

    - [torch.utils.data API](https://pytorch.org/docs/stable/data.html)
    """)
    return


if __name__ == "__main__":
    app.run()
