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
    *PyTorch basics, 3 of 8 — before this: [Tensors](02-tensors.py) · after:
    [Transforms](04-transforms.py)*

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
    In the quickstart, the training loop never touched a file: data arrived as ready-made
    `(64, 1, 28, 28)` batches, one `for` iteration at a time. On disk, FashionMNIST is 60,000
    individual samples. This notebook is about the two pieces PyTorch puts between those facts —
    and keeps deliberately separate from the model and its training loop, so that dataset code
    stays swappable without touching the loop:

    - `torch.utils.data.Dataset` stores the samples and their labels, and answers one question:
      *give me sample `i`*.
    - `torch.utils.data.DataLoader` wraps any `Dataset` and turns single samples into shuffled,
      batched, optionally parallel-loaded minibatches.

    TorchVision ships many ready-made `Dataset` subclasses to prototype and benchmark against;
    FashionMNIST below is one of its [image datasets](https://pytorch.org/vision/stable/datasets.html).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading a dataset

    Here is an example of how to load the
    [Fashion-MNIST](https://research.zalando.com/project/fashion_mnist/fashion_mnist/) dataset from
    TorchVision. Fashion-MNIST is a dataset of Zalando’s article images consisting of 60,000
    training examples and 10,000 test examples. Each example comprises a 28×28 grayscale image and
    an associated label from one of 10 classes.

    We load the [FashionMNIST Dataset](https://pytorch.org/vision/stable/datasets.html#fashion-mnist)
    with the following parameters:
    - `root` is the path where the train/test data is stored.
    - `train` selects the training or the test split.
    - `download=True` downloads the data from the internet if it is not available at `root`.
    - `transform` and `target_transform` specify the feature and label transformations.
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
    return Dataset, datasets, plt, test_data, torch, training_data, v2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Iterating and visualizing the dataset

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
    ### The dataset as something you can page through

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
    ## Creating a custom Dataset for your files

    A custom Dataset class must implement three functions:
    `__init__`, `__len__`, and
    `__getitem__`. Take a look at this implementation; the
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

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What each of the three methods does

    `__init__` runs once, when the dataset is constructed. Here it reads the annotations file
    into a DataFrame and stores the image directory and both transforms — no image is loaded
    yet. The labels.csv file looks like:

        tshirt1.jpg, 0
        tshirt2.jpg, 0
        ......
        ankleboot999.jpg, 9

    `__len__` returns the number of samples — here, the number of CSV rows. Everything that
    needs the dataset's size, from `len(dataset)` to the sampler deciding how many indices an
    epoch has, asks this.

    `__getitem__(idx)` does the real per-sample work: find the image's location on disk, decode
    the file to a tensor with `decode_image`, look up the label in the CSV data, apply both
    transforms if given, and return the `(image, label)` tuple. All the cost of data loading
    lives in this call — a fact the DataLoader section below comes back to when it puts worker
    processes behind exactly this method.
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
    ## Preparing your data for training with DataLoaders

    The `Dataset` retrieves our dataset's features and labels one sample at a time. While training
    a model, we typically want to pass samples in "minibatches", reshuffle the data at every epoch
    to reduce model overfitting, and use Python's `multiprocessing` to speed up data retrieval.

    `DataLoader` is an iterable that abstracts this complexity for us in an easy API.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Everything it needs is `__len__` and `__getitem__`

    `DataLoader` never learns what a sample *means*: it asks the dataset for its length, asks for
    samples by index, and stacks what comes back. Before pointing one at 60,000 images, hand it
    the smallest `Dataset` that satisfies the contract, and predict: ten samples in batches of
    four — how many batches, and what exactly arrives in each?
    """)
    return


@app.cell
def _(Dataset):
    from torch.utils.data import DataLoader

    class Squares(Dataset):
        """Sample i is the pair (i, i * i) -- every value names its own position."""

        def __len__(self):
            return 10

        def __getitem__(self, idx):
            return idx, idx * idx

    for _indices, _squares in DataLoader(Squares(), batch_size=4):
        print(_indices.tolist(), _squares.tolist(), "  <-", type(_indices).__name__, str(_indices.dtype))
    return DataLoader, Squares


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Three things happened without being asked for: the four `(index, square)` pairs came back as
    *one* pair of tensors, because the default **collate** step stacks samples column-wise; the
    plain Python ints were promoted to `torch.int64` tensors along the way; and the epoch ended
    with a short batch of two, because ten does not divide by four. All three reappear at scale
    below.
    """)
    return


@app.cell
def _(DataLoader, test_data, training_data):
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64)
    return (train_dataloader,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Iterating through the DataLoader

    The same call, pointed at the real dataset. The training loader shuffles; the test loader
    keeps file order, since evaluation reads every sample exactly once either way. Each iteration
    returns one batch — `train_features` and `train_labels`, 64 of each — and `shuffle=True`
    re-permutes the order every epoch (for finer-grained control over loading order, take a look
    at [Samplers](https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler)).
    """)
    return


@app.cell
def _(labels_map, plt, train_dataloader):
    # Display image and label.
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    _img = train_features[0].squeeze()
    _label = int(train_labels[0])
    plt.imshow(_img, cmap="gray")
    plt.show()
    print(f"Label: {_label} = {labels_map[_label]}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The three arguments that decide what a batch is

    `DataLoader(training_data, batch_size=64, shuffle=True)` hides three separate
    decisions. Change them and watch what arrives.

    - **batch_size** sets the first dimension of every tensor the model sees. It is also
      the number that `len(dataloader)` counts *batches* of, not samples — an off-by-a-
      factor-of-64 error waiting to happen in a progress bar.
    - **shuffle** decides whether the sampler walks the dataset in file order or permutes
      it each epoch. FashionMNIST is already stored well mixed — all ten classes appear in
      the first 64 images, which is also how the quickstart got away with never passing
      `shuffle` at all — so the class counts below barely move when you turn it off.
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
    _epoch1 = next(iter(explore_loader.batch_sampler))[:6]
    _epoch2 = next(iter(explore_loader.batch_sampler))[:6]
    _order = "a fresh permutation every epoch" if shuffle_choice.value else "file order, identical every epoch"

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
            mo.md(f"The sampler starts epoch 1 at `{_epoch1}…`, epoch 2 at `{_epoch2}…` — {_order}."),
            mo.md("The first batch, up to the first 24 images:"),
            mo.hstack(_strip, justify="start", wrap=True, gap=0.5),
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The argument that decides who does the work

    Every batch so far was assembled in this notebook's own process: `DataLoader` called
    `__getitem__` 64 times, collated the results, and only then handed the batch over — all time
    the training step spends waiting. `num_workers=n` moves that work into `n` child processes,
    which prefetch batches ahead of whoever is consuming them.

    Whether that pays depends on what `__getitem__` costs — guess before pressing. FashionMNIST
    sits in memory as uint8 tensors and its transform only converts one sample to float32: do
    four workers help? And a heavier pipeline — a rotation and a blur per sample, the shape of
    real augmentation — is the same question with a different answer.
    """)
    return


@app.cell
def _(mo):
    time_it = mo.ui.run_button(label="time four loaders — a few seconds")
    time_it
    return (time_it,)


@app.cell
def _(DataLoader, datasets, mo, time_it, torch, v2):
    mo.stop(not time_it.value, mo.md("*Nothing here is precomputed — press the button.*"))

    import time

    _pipelines = {
        "float32 only": v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        "rotation + blur": v2.Compose(
            [v2.ToImage(), v2.RandomRotation(15), v2.GaussianBlur(9), v2.ToDtype(torch.float32, scale=True)]
        ),
    }

    def _timed(transform, num_workers, batches=100):
        _ds = datasets.FashionMNIST(root="data", train=True, transform=transform)
        _batches = iter(DataLoader(_ds, batch_size=64, num_workers=num_workers))
        _start = time.perf_counter()
        for _ in range(batches):
            next(_batches)
        return time.perf_counter() - _start

    _boot = _timed(_pipelines["float32 only"], num_workers=4, batches=1)
    _rows = [mo.stat(f"{_boot:.2f} s", label="first parallel loader (forkserver boot), paid once", bordered=True)]
    for _name, _transform in _pipelines.items():
        _serial = _timed(_transform, num_workers=0)
        _parallel = _timed(_transform, num_workers=4)
        _rows.append(
            mo.hstack(
                [
                    mo.stat(f"{_serial:.2f} s", label=f"{_name} — workers=0", bordered=True),
                    mo.stat(f"{_parallel:.2f} s", label=f"{_name} — workers=4", bordered=True),
                    mo.stat(f"{_serial / _parallel:.1f}×", label="speedup", bordered=True),
                ],
                justify="start",
                gap=1,
            )
        )
    mo.vstack([mo.md("100 batches of 64 each, measured just now:"), *_rows], gap=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Two traps, met here rather than in the wild. The startup stat exists because Python 3.14
    boots workers through a *forkserver* by default — under the old Linux default, `fork`,
    `num_workers` felt free, which is why older tutorials never mention a cost. And everything a
    worker touches — the dataset object and its transforms — travels to it by pickle: `Squares`
    above, a class defined in a notebook cell, dies with `module '__mp_main__' has no attribute
    'Squares'` the moment it meets `num_workers > 0`, while the importable `FashionMNIST` travels
    fine.
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
    ## Where to go next

    - [torch.utils.data](https://pytorch.org/docs/stable/data.html) documents what this notebook
      skipped: `Sampler`s, custom `collate_fn`s, `IterableDataset` for data with no length,
      memory pinning for faster host-to-GPU copies.
    - The moment training spans more than one GPU, the loader is where it shows first: a
      `DistributedSampler` hands each rank its own disjoint shard of every epoch's indices, and
      `drop_last` stops being cosmetic (the hang described above). That is this repository's
      road; the tutorial's next stop is [Transforms](04-transforms.py).
    - The `datasets` library at Hugging Face serves the same two-method contract, so a
      `DataLoader` consumes its datasets directly.
    """)
    return


if __name__ == "__main__":
    app.run()
