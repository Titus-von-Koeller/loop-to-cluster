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

    We load the [FashionMNIST Dataset](https://pytorch.org/vision/stable/datasets.html#fashion-mnist) with the following parameters:
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
    return (train_dataloader,)


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
