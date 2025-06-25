#!/usr/bin/env python3
from common import build_model, get_dataloaders
from config import local_config, metacentrum_config, sge_config
from parse_arguments import parse_args
from trainers.BaseFFTrainer import BaseFFTrainer
import numpy as np


def main():
    args = parse_args()

    config = sge_config if args.sge else metacentrum_config if args.metacentrum else local_config

    # Load the datasets first
    train_dataloader, val_dataloader, eval_dataloader = get_dataloaders(
        dataset=args.dataset,
        config=config,
        lstm=True if "LSTM" in args.classifier else False,
        augment=args.augment,
    )

    # Build model with correct number of classes
    model, trainer = build_model(args, num_classes=len(np.bincount(train_dataloader.dataset.get_labels())))

    print(f"Trainer: {type(trainer).__name__}")

    # Load the model from the checkpoint
    if args.checkpoint:
        trainer.load_model(args.checkpoint)
        print(f"Loaded model from {args.checkpoint}.")
    else:
        raise ValueError("Checkpoint must be specified when only evaluating.")

    print(f"Fine-tuning {type(model).__name__} on {type(train_dataloader.dataset).__name__} dataloader.")

    # Fine-tune the model
    if isinstance(trainer, BaseFFTrainer):
        trainer.finetune(train_dataloader, val_dataloader, numepochs=args.num_epochs, finetune_ssl=True, start_epoch=args.start_epoch)
        trainer.eval(eval_dataloader, subtitle="finetune")
    else:
        raise NotImplementedError("Fine-tuning is only implemented for FF models.")
    

if __name__ == "__main__":
    main()
