#!/usr/bin/env python3
from common import build_model, get_dataloaders
import config # args restriction check and type check
from parse_arguments import parse_args

# trainers
from trainers.BaseFFTrainer import BaseFFTrainer
from trainers.BaseSklearnTrainer import BaseSklearnTrainer
import numpy as np

from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None,config_path="configs", config_name="default")
def main(args: DictConfig):
    print(OmegaConf.to_yaml(args))
    train_dataloader, val_dataloader, eval_dataloader = get_dataloaders(
        dataset=args.training.dataset,
        config=args.environment,
        lstm=True if "LSTM" in args.model.classifier else False,
        augment=args.training.augment,
    )

    model, trainer = build_model(args, num_classes=len(np.bincount(train_dataloader.dataset.get_labels())))

    if args.training.checkpoint:
        trainer.load_model(args.training.checkpoint)

    # TODO: Implement training of MHFA and AASIST with SkLearn models

    print(f"Trainer: {type(trainer).__name__}")
    print(f"Model: {type(model).__name__}")
    print(f"Extractor: {type(model.extractor).__name__}")
    print(f"Feature Processor: {type(model.feature_processor).__name__}")
    print(f"Dataset: {type(train_dataloader.dataset).__name__}")

    # Train the model
    if isinstance(trainer, BaseFFTrainer):
        # Default value of numepochs = 20
        trainer.train(train_dataloader, val_dataloader, numepochs=args.training.num_epochs, start_epoch=args.training.start_epoch)
        trainer.eval(eval_dataloader, subtitle=str(args.training.num_epochs))  # Eval after training

    elif isinstance(trainer, BaseSklearnTrainer):
        # Default value of variant = all
        trainer.train(train_dataloader, val_dataloader, variant=args.variant)
        trainer.eval(eval_dataloader)  # Eval after training

    else:
        # Should not happen, should inherit from BaseSklearnTrainer or BaseFFTrainer
        raise ValueError("Invalid trainer, should inherit from BaseSklearnTrainer or BaseFFTrainer.")


if __name__ == "__main__":
    main()
