"""
adapted from: https://github.com/jrbtaylor/conditional-pixelcnn
"""

import os
import json
from pathlib import Path

import torch
import trainer, utils, model
from args import args


def run(resume=False):
    exp_name = args.exp_name
    exp_name += "_%s_%ifeat_%ilayers_%ibins" % (
        args.dataset_name,
        args.n_features,
        args.n_layers,
        args.n_bins,
    )

    exp_dir = args.exp_dir / exp_name
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)

    # Data loaders
    train_loader, val_loader, label2onehot, n_classes = utils.data_loader(
        args.dataset_name, args.batch_size
    )

    if not resume:
        # Store experiment params in params.json
        params = {
            "batch_size": args.batch_size,
            "n_features": args.n_features,
            "n_layers": args.n_layers,
            "n_bins": args.n_bins,
            "optimizer": args.optimizer,
            "learning_rate": args.learning_rate,
            "dropout": args.dropout,
            "cuda": args.cuda,
        }

        with open(exp_dir / "params.json", "w") as f:
            json.dump(params, f)

        # Model
        net = model.PixelCNN(
            1, n_classes, args.n_features, args.n_layers, args.n_bins, args.dropout
        )

    else:
        # if resuming, need to have params, stats and checkpoint files
        if not (
            os.path.isfile(exp_dir / "params.json")
            and os.path.isfile(exp_dir / "stats.json")
            and os.path.isfile(exp_dir / "last_checkpoint.pth")
        ):
            raise Exception("Missing param, stats or checkpoint file on resume")
        net = torch.load(exp_dir / "last_checkpoint.pth")
    print("-" * 100)
    print("model loaded")
    print(net)
    print()
    print("-" * 100)

    # Define loss fcn
    loss_fn = torch.nn.NLLLoss()

    # Train
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        model=net,
        exp_path=exp_dir,
        label_preprocess=utils.input2label,
        loss_fn=loss_fn,
        label2onehot=label2onehot,
        n_classes=10,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        cuda=args.cuda,
        patience=args.es_criteria,
        max_epochs=args.num_epochs,
        resume=False,
    )

    # Generate some between-class examples
    trainer.generate_between_classes(
        net, [28, 28], [1, 7], os.path.join(exp_dir, "1-7.jpeg"), n_classes, args.cuda
    )
    trainer.generate_between_classes(
        net, [28, 28], [3, 8], os.path.join(exp_dir, "3-8.jpeg"), n_classes, args.cuda
    )
    trainer.generate_between_classes(
        net, [28, 28], [4, 9], os.path.join(exp_dir, "4-9.jpeg"), n_classes, args.cuda
    )
    trainer.generate_between_classes(
        net, [28, 28], [5, 6], os.path.join(exp_dir, "5-6.jpeg"), n_classes, args.cuda
    )


if __name__ == "__main__":
    run(resume=False)
