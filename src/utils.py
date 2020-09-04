import json
import os
from typing import Callable

import imageio
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pathlib

import torch
from torchvision import datasets, transforms
from torchvision.datasets.mnist import read_label_file, read_image_file

from args import args


def clearline():
    CURSOR_UP_ONE = "\x1b[1A"
    ERASE_LINE = "\x1b[2K"
    print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)


def input2label(x: torch.Tensor) -> torch.LongTensor:
    """
    - Convert a torch array containing floats to contain ints
    - The continuous values of 'x' are binned based on n_bins set at args.py
    - This will turn our problem of predicting the next pixel value to 
     a classification problem (instead of regression)
    """
    return torch.squeeze(torch.round((args.n_bins - 1) * x).type(torch.LongTensor), 1)


def tile_images(images: np.array, n_rows=0) -> np.array:
    n_images = len(images)
    height = images[0].shape[1]
    width = images[0].shape[2]
    if n_rows == 0:
        n_rows = int(np.floor(np.sqrt(n_images)))

    while n_images % n_rows != 0:
        n_rows -= 1
    n_cols = n_images // n_rows

    images = np.squeeze(np.array(images), axis=1)
    images = np.transpose(images, (1, 2, 0))
    images = np.reshape(images, [height, width, n_rows, n_cols])
    images = np.transpose(images, (2, 3, 0, 1))
    images = np.concatenate(images, 1)
    images = np.concatenate(images, 1)
    return images


def plot_stats(stats, savepath: str) -> None:
    """
    Make all the plots in stats. Stats can be a dict or a path to json (str)
    """
    if type(stats) is str:
        assert os.path.isfile(stats)
        with open(stats, "r") as sf:
            stats = json.load(sf)

    assert type(stats) is dict, "stats must be a dictionary"

    if not os.path.isdir(savepath):
        os.makedirs(savepath)

    def _plot(y, title):
        plt.Figure()
        if type(y) is list:
            plt.plot(range(1, len(y) + 1), y)
        elif type(y) is dict:
            for key, z in y.items():
                plt.plot(range(1, len(z) + 1), z, label=key)
            plt.legend()
        else:
            raise ValueError
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.title(title)
        plt.savefig(os.path.join(savepath, title.replace(" ", "_") + ".png"))
        plt.close()

    # Loop over stats dict and plot. Dicts within stats get plotted together.
    for key, value in stats.items():
        _plot(value, key)


def get_label2onehot(n_classes: int) -> Callable:
    def label2onehot(target_class_index):
        one_hot_vector = np.zeros((n_classes), dtype="float32")
        one_hot_vector[target_class_index] = 1
        return one_hot_vector

    return label2onehot


def augment(rotate=5):
    return transforms.Compose(
        [transforms.RandomRotation(rotate), transforms.ToTensor()]
    )


def data_loader(dataset, batch_size, n_workers=8):
    assert dataset.lower() in ["mnist", "fashionmnist"]

    loader_args = {
        "batch_size": batch_size,
        "num_workers": n_workers,
        "pin_memory": True,
    }
    datapath = os.path.join(os.getenv("HOME"), "data", dataset.lower())
    dataset_args = {
        "root": datapath,
        "download": True,
        "transform": transforms.ToTensor(),
    }

    if dataset.lower() == "mnist":
        dataset_init = datasets.MNIST
        n_classes = 10
    else:
        dataset_init = datasets.FashionMNIST
        n_classes = 10
    label2onehot = get_label2onehot(n_classes)
    dataset_args.update({"target_transform": label2onehot})

    val_loader = torch.utils.data.DataLoader(
        dataset_init(train=False, **dataset_args), shuffle=False, **loader_args
    )
    dataset_args["transform"] = augment()
    train_loader = torch.utils.data.DataLoader(
        dataset_init(train=True, **dataset_args), shuffle=True, **loader_args
    )

    return train_loader, val_loader, label2onehot, n_classes
