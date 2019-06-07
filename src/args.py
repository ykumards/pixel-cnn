import torch
from argparse import Namespace
from pathlib import Path

args = Namespace(
    exp_name="pixelCNN_trial",
    exp_dir = Path('../experiments/'),
    seed=123,
    disable_cuda=False,
    cuda=False,
    debug = False,
    num_workers = 1,
    dataset_name = 'mnist',

    ######### Training Params ######################
    num_epochs=2,
    es_criteria=9,
    reduce_lr_criteria=3,
    es_threshold=0.05,
    learning_rate=1e-4,
    batch_size=8,
    optimizer='adam',

    ######### Model Params #########################
    n_features=2,
    n_layers=1,
    n_bins=4,
    dropout=0.3,
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
