import json
import os

import imageio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pathlib

import torch
from torchvision import datasets, transforms
from torchvision.datasets.mnist import read_label_file, read_image_file

def clearline():
    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)
    
def tile_images(images: np.array, n_rows=0):
    n_images = len(images)
    height = images[0].shape[1]
    width = images[0].shape[2]
    if n_rows == 0:
        n_rows = int(np.floor(np.sqrt(n_images)))
    
    while n_images % n_rows != 0:
        n_rows -= 1
    n_cols = n_images//n_rows
    
    images = np.squeeze(np.array(images), axis=1)
    images = np.transpose(images, (1, 2, 0))
    images = np.reshape(images,[height, width, n_rows, n_cols])
    images = np.transpose(images,(2, 3, 0, 1))
    images = np.concatenate(images, 1)
    images = np.concatenate(images, 1)
    return images

def plot_stats(stats, savepath: str):
    """
    Make all the plots in stats. Stats can be a dict or a path to json (str)
    """
    if type(stats) is str:
        assert os.path.isfile(stats)
        with open(stats,'r') as sf:
            stats = json.load(sf)
            
    assert type(stats) is dict, "stats must be a dictionary"
    
    if not os.path.isdir(savepath):
        os.makedirs(savepath)

    def _plot(y, title):
        plt.Figure()
        if type(y) is list:
            plt.plot(range(1, len(y)+1), y)
        elif type(y) is dict:
            for key, z in y.items():
                plt.plot(range(1, len(z)+1), z, label=key)
            plt.legend()
        else:
            raise ValueError
        plt.xlabel('Epoch')
        plt.ylabel(title)
        plt.title(title)
        plt.savefig(os.path.join(savepath, title.replace(' ', '_') + '.png'))
        plt.close()

    # Loop over stats dict and plot. Dicts within stats get plotted together.
    for key, value in stats.items():
        _plot(value, key)            
        
def onehot(n_classes):
    def onehot_fcn(x):
        y = np.zeros((n_classes), dtype='float32')
        y[x] = 1
        return y
    return onehot_fcn


def augment(rotate=5):
    return transforms.Compose([transforms.RandomRotation(rotate),
                               transforms.ToTensor()])
def data_loader(dataset, batch_size, n_workers=8):
    assert dataset.lower() in ['mnist','emnist','fashionmnist']

    loader_args = {'batch_size':batch_size,
                   'num_workers':n_workers,
                   'pin_memory':True}
    datapath = os.path.join(os.getenv('HOME'), 'data', dataset.lower())
    dataset_args = {'root':datapath,
                    'download':True,
                    'transform':transforms.ToTensor()}

    if dataset.lower()=='mnist':
        dataset_init = datasets.MNIST
        n_classes = 10
    elif dataset.lower()=='emnist':
        dataset_init = EMNIST
        n_classes = 37
        dataset_args.update({'split':'letters'})
    else:
        dataset_init = datasets.FashionMNIST
        n_classes = 10
    onehot_fcn = onehot(n_classes)
    dataset_args.update({'target_transform':onehot_fcn})

    val_loader = torch.utils.data.DataLoader(
        dataset_init(train=False, **dataset_args), shuffle=False, **loader_args)

    dataset_args['transform'] = augment()
    train_loader = torch.utils.data.DataLoader(
        dataset_init(train=True, **dataset_args), shuffle=True, **loader_args)


    return train_loader, val_loader, onehot_fcn, n_classes
        