#!/usr/bin/env python3
from torch.utils.data import DataLoader
import numpy as np

from config import local_config, metacentrum_config, sge_config
from common import build_model, get_dataloaders
from parse_arguments import parse_args


def main():
    args = parse_args()

    config = sge_config if args.sge else metacentrum_config if args.metacentrum else local_config

    # Load the dataset first
    eval_dataloader = get_dataloaders(
        dataset=args.dataset,
        config=config,
        lstm=True if "LSTM" in args.classifier else False,
        eval_only=True,
    )
    assert isinstance( # Is here for type checking and hinting compliance
        eval_dataloader, DataLoader
    ), "Error type of eval_dataloader returned from get_dataloaders."

    # Build model with correct number of classes
    model, trainer = build_model(args, eval_dataloader.dataset.num_speakers)

    print(f"Trainer: {type(trainer).__name__}")
    print(f"Model: {type(model).__name__}")
    print(f"Extractor: {type(model.extractor).__name__}")
    print(f"Feature Processor: {type(model.feature_processor).__name__}")
    print(f"Dataset: {type(eval_dataloader.dataset).__name__}")

    # Load the model from the checkpoint
    if args.checkpoint:
        trainer.load_model(args.checkpoint)
    else:
        raise ValueError("Checkpoint must be specified when only evaluating.")

    print(
        f"Evaluating {args.checkpoint} {type(model).__name__} on "
        + f"{type(eval_dataloader.dataset).__name__} dataloader."
    )
    subtitle = str(args.checkpoint).split("/")[-1]
    trainer.eval(eval_dataloader, subtitle=subtitle)


if __name__ == "__main__":
    main()
