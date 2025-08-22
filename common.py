# region Imports
from argparse import Namespace
from typing import Dict, Tuple
import os

import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# Classifiers
from classifiers.FFBase import FFBase
from classifiers.single_input.EmbeddingFF import EmbeddingFF

from datasets.SLTSSTC import SLTSSTCDataset_pair, SLTSSTCDataset_single, SLTSSTCDataset_eval
from datasets.utils import custom_pair_batch_create, custom_single_batch_create, custom_eval_batch_create

# Extractors
from extractors.HuBERT import HuBERT_base, HuBERT_extralarge, HuBERT_large
from extractors.Wav2Vec2 import Wav2Vec2_base, Wav2Vec2_large, Wav2Vec2_LV60k
from extractors.WavLM import WavLM_base, WavLM_baseplus, WavLM_large
from extractors.XLSR import XLSR_1B, XLSR_2B, XLSR_300M
from extractors.MelSpectrogram import MelSpectrogram
from extractors.FDLP import FDLP

# Feature processors
from feature_processors.AASIST import AASIST
from feature_processors.MeanProcessor import MeanProcessor
from feature_processors.MHFA import MHFA
from feature_processors.SLS import SLS
from feature_processors.ResNet import ResNet293
from feature_processors.ECAPA_TDNN import ECAPA_TDNN

# Trainers
from trainers.BaseTrainer import BaseTrainer
from trainers.FFDotTrainer import FFDotTrainer
from trainers.FFPairTrainer import FFPairTrainer
from trainers.EmbeddingFFTrainer import EmbeddingFFTrainer

# Loss functions
from losses.CrossEntropyLoss import CrossEntropyLoss
from losses.AdditiveAngularMarginLoss import AdditiveAngularMarginLoss

import sys
def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

def get_dataloaders(
    dataset: str,
    config: dict,
    lstm: bool = False,
    augment: bool = False,
    eval_only: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader] | DataLoader: # return training dataloader, validation dataloader, evaluation dataloader or just evaluation dataloader depending on the mode
    dataset_config = {}
    t = "pair" if "pair" in dataset else "single"
    if "SLTSSTC" in dataset:
        train_dataset_class = str_to_class(dataset)
        val_dataset_class = str_to_class("SLTSSTCDataset_eval")
        eval_dataset_class = str_to_class("SLTSSTCDataset_eval")
        dataset_config = config.sltsstc
    else:
        raise ValueError("Invalid dataset name.")
    
    # Common parameters
    collate_func = custom_single_batch_create
    collate_func_eval = custom_eval_batch_create
    bs = config["batch_size"] if not lstm else config["lstm_batch_size"]  # Adjust batch size for LSTM models

    # Load the datasets
    train_dataloader = DataLoader(Dataset())  # dummy dataloader for type hinting compliance
    val_dataloader = DataLoader(Dataset())  # dummy dataloader for type hinting compliance

    # always load training dataset to get the number of unique speakers
    print("Loading training datasets...")
    train_dataset = train_dataset_class(
        root_dir=config["data_dir"] + dataset_config["train_subdir"],
        protocol_file_name=dataset_config["train_protocol"],
        variant="train",
        augment=augment,
        rir_root=config["rir_root"],
    )

    dev_kwargs = {  # kwargs for the dataset class
        "root_dir": config["data_dir"] + dataset_config["dev_subdir"],
        "protocol_file_name": dataset_config["dev_protocol"],
        "variant": "dev",
    }
    val_dataset = val_dataset_class(**dev_kwargs)

    # there is about 90% of spoofed recordings in the dataset, balance with weighted random sampling
    # samples_weights = [train_dataset.get_class_weights()[i] for i in train_dataset.get_labels()]  # old and slow solution
    samples_weights = np.vectorize(train_dataset.get_class_weights().__getitem__)(
        train_dataset.get_labels()
    )  # blazing fast solution
    weighted_sampler = WeightedRandomSampler(samples_weights, len(train_dataset))

    # create dataloader, use custom collate_fn to pad the data to the longest recording in batch
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=bs,
        collate_fn=collate_func,
        sampler=weighted_sampler,
        drop_last=True,
        num_workers=int(os.environ.get("OMP_NUM_THREADS")),
    )
    if not eval_only:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=bs,
            collate_fn=collate_func_eval,
            shuffle=True,
            drop_last=True,
            num_workers=int(os.environ.get("OMP_NUM_THREADS")),
        )


    print("Loading eval dataset...")
    eval_kwargs = {  # kwargs for the dataset class
        "root_dir": config["data_dir"] + dataset_config["eval_subdir"],
        "protocol_file_name": dataset_config["eval_protocol"],
        "variant": "eval",
        "num_speakers": train_dataset.num_speakers,
    }

    # Create the dataset based on dynamically created eval_kwargs
    eval_dataset = eval_dataset_class(**eval_kwargs)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=bs,
        collate_fn=collate_func_eval,
        shuffle=True,
        num_workers=int(os.environ.get("OMP_NUM_THREADS")),
    )

    if eval_only:
        return eval_dataloader
    else:
        return train_dataloader, val_dataloader, eval_dataloader
    

def build_model(args: Namespace, num_classes: int = 2) -> Tuple[FFBase, BaseTrainer]:
    # Beware of MHFA or AASIST with SkLearn models, they are not implemented yet
    if args.model.processor in ["MHFA", "AASIST", "SLS"] and args.model.classifier in ["GMMDiff", "SVMDiff", "LDAGaussianDiff"]:
        raise NotImplementedError("Training of SkLearn models with MHFA, AASIST or SLS is not yet implemented.")
    # region Extractor
    extractor = str_to_class(args.model.extractor)()  # map the argument to the class and instantiate it
    # endregion

    # region Processor (pooling)
    processor = None
    if args.model.processor == "MHFA":
        input_transformer_nb = extractor.transformer_layers
        input_dim = extractor.feature_size

        processor_output_dim = (
            input_dim  # Output the same dimension as input - might want to play around with this
        )
        compression_dim = processor_output_dim // 8
        head_nb = round(
            input_transformer_nb * 4 / 3
        )  # Half random guess number, half based on the paper and testing

        processor = MHFA(
            head_nb=head_nb,
            input_transformer_nb=input_transformer_nb,
            inputs_dim=input_dim,
            compression_dim=compression_dim,
            outputs_dim=processor_output_dim,
        )
    elif args.model.processor == "AASIST":
        processor = AASIST(
            inputs_dim=extractor.feature_size,
            # compression_dim=extractor.feature_size // 8,  # compression dim is hardcoded at the moment
            outputs_dim=extractor.feature_size,  # Output the same dimension as input, might want to play around with this
        )
    elif args.model.processor == "SLS":
        processor = SLS(
            inputs_dim=extractor.feature_size,
            outputs_dim=extractor.feature_size,  # Output the same dimension as input, might want to play around with this
        )
    elif args.model.processor == "Mean":
        processor = MeanProcessor()  # default avg pooling along the transformer layers and time frames
    elif args.model.processor == "ResNet293":
        processor = ResNet293(
            input_dim=extractor.feature_size,
            output_dim=extractor.feature_size,
        )
    elif args.model.processor == "ECAPA_TDNN":
        processor = ECAPA_TDNN(
            input_dim=extractor.feature_size,
            output_dim=extractor.feature_size,
        )
    else:
        raise ValueError("Only AASIST, MHFA, Mean and SLS processors are currently supported.")
    # endregion

    # region Loss Function
    loss_class = str_to_class(args.model.loss.name)
    loss_params_dict = {}
    if args.model.loss.type == "embedding":
        loss_params_dict['in_features'] = extractor.feature_size
        loss_params_dict['out_features'] = num_classes
    loss_params_dict.update(args.model.loss.items())
    # Create the loss function instance
    loss_fn = loss_class(**loss_params_dict)
    # endregion

    # region Model and trainer
    model: FFBase
    trainer = None
    try:
        model = str_to_class(args.model.classifier)(
            extractor, processor, loss_fn=loss_fn, in_dim=extractor.feature_size, num_classes=num_classes
        )
        trainer_class = str_to_class(args.model.trainer)
        trainer = trainer_class(model, save_embeddings=args.training.save_embeddings)
    except KeyError:
        raise ValueError(f"Invalid classifier, check the classifier/ folder for available classifiers.")
    # endregion

    # Print model info
    print(f"Building {type(model).__name__} model with {type(model.extractor).__name__} extractor", end="")
    if isinstance(model, FFBase):
        print(f" and {type(model.feature_processor).__name__} processor.", end="")
    if loss_fn is not None:
        print(f" Using {type(loss_fn).__name__} loss function.")
    else:
        print(".")

    return model, trainer
