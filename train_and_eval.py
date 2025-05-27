#!/usr/bin/env python3
from common import build_model, get_dataloaders
from config import local_config, metacentrum_config, sge_config
from parse_arguments import parse_args

# trainers
from trainers.BaseFFTrainer import BaseFFTrainer
from trainers.BaseSklearnTrainer import BaseSklearnTrainer
import numpy as np


def main():
    args = parse_args()

    config = sge_config if args.sge else metacentrum_config if args.metacentrum else local_config

    train_dataloader, val_dataloader, eval_dataloader = get_dataloaders(
        dataset=args.dataset,
        config=config,
        lstm=True if "LSTM" in args.classifier else False,
        augment=args.augment,
    )

    model, trainer = build_model(args, num_classes=len(np.bincount(train_dataloader.dataset.get_labels())))
    # TODO: Implement training of MHFA and AASIST with SkLearn models

    print(f"Trainer: {type(trainer).__name__}")
    print(f"Model: {type(model).__name__}")
    print(f"Extractor: {type(model.extractor).__name__}")
    print(f"Feature Processor: {type(model.feature_processor).__name__}")
    print(f"Dataset: {type(train_dataloader.dataset).__name__}")

    # Train the model
    if isinstance(trainer, BaseFFTrainer):
        # Default value of numepochs = 20
        trainer.train(train_dataloader, val_dataloader, numepochs=args.num_epochs)
        #trainer.eval(eval_dataloader, subtitle=str(args.num_epochs))  # Eval after training

    elif isinstance(trainer, BaseSklearnTrainer):
        # Default value of variant = all
        trainer.train(train_dataloader, val_dataloader, variant=args.variant)
        trainer.eval(eval_dataloader)  # Eval after training

    else:
        # Should not happen, should inherit from BaseSklearnTrainer or BaseFFTrainer
        raise ValueError("Invalid trainer, should inherit from BaseSklearnTrainer or BaseFFTrainer.")


if __name__ == "__main__":
    main()
