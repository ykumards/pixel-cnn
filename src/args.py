import torch
from argparse import Namespace
from pathlib import Path

args = Namespace(
    exp_name="pixelCNN_v1",
    exp_dir = Path('../experiments/'),
    seed=123,
    disable_cuda=False,
    cuda=False,
    debug = False,
    num_workers = 1,
    dataset_name = 'mnist',

    ######### Training Params ######################
    num_epochs=50,
    es_criteria=8,
    reduce_lr_criteria=3,
    es_threshold=0.05,
    learning_rate=1e-4,
    batch_size=32,
    optimizer='adam',

    ######### Model Params #########################
    n_features=512,
    n_layers=6,
    n_bins=128,
    dropout=0.7,
)

if not args.disable_cuda and torch.cuda.is_available():
    print("using cuda")
    args.cuda = True
else:
    print("not using cuda")
    args.cuda = False

if args.debug:
    args.num_workers = 1
else:
    args.num_workers = 20
