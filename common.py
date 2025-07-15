# region Imports
from argparse import Namespace
from typing import Dict, Tuple
import os

import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# Classifiers
from classifiers.BaseSklearnModel import BaseSklearnModel
from classifiers.differential.FFConcat import FFLSTM, FFLSTM2, FFConcat1, FFConcat2, FFConcat3
from classifiers.differential.FFDiff import FFDiff, FFDiffAbs, FFDiffQuadratic
from classifiers.differential.FFDot import FFDot
from classifiers.differential.GMMDiff import GMMDiff
from classifiers.differential.LDAGaussianDiff import LDAGaussianDiff
from classifiers.differential.SVMDiff import SVMDiff
from classifiers.FFBase import FFBase
from classifiers.single_input.EmbeddingFF import EmbeddingFF

# Datasets
from datasets.ASVspoof5 import ASVspoof5Dataset_pair, ASVspoof5Dataset_single
from datasets.ASVspoof2019 import ASVspoof2019LADataset_pair, ASVspoof2019LADataset_single
from datasets.ASVspoof2021 import (
    ASVspoof2021DFDataset_pair,
    ASVspoof2021DFDataset_single,
    ASVspoof2021LADataset_pair,
    ASVspoof2021LADataset_single,
)
from datasets.SLTSSTC import SLTSSTCDataset_pair, SLTSSTCDataset_single, SLTSSTCDataset_eval
from datasets.InTheWild import InTheWildDataset_pair, InTheWildDataset_single
from datasets.Morphing import MorphingDataset_pair, MorphingDataset_single
from datasets.utils import custom_pair_batch_create, custom_single_batch_create, custom_eval_batch_create

# Extractors
from extractors.HuBERT import HuBERT_base, HuBERT_extralarge, HuBERT_large
from extractors.Wav2Vec2 import Wav2Vec2_base, Wav2Vec2_large, Wav2Vec2_LV60k
from extractors.WavLM import WavLM_base, WavLM_baseplus, WavLM_large
from extractors.XLSR import XLSR_1B, XLSR_2B, XLSR_300M
from extractors.MelSpectrogram import MelSpectrogram

# Feature processors
from feature_processors.AASIST import AASIST
from feature_processors.MeanProcessor import MeanProcessor
from feature_processors.MHFA import MHFA
from feature_processors.SLS import SLS
from feature_processors.ResNet import ResNet293

# Trainers
from trainers.BaseTrainer import BaseTrainer
from trainers.FFDotTrainer import FFDotTrainer
from trainers.FFPairTrainer import FFPairTrainer
from trainers.EmbeddingFFTrainer import EmbeddingFFTrainer
from trainers.GMMDiffTrainer import GMMDiffTrainer
from trainers.LDAGaussianDiffTrainer import LDAGaussianDiffTrainer
from trainers.SVMDiffTrainer import SVMDiffTrainer

# Loss functions
from losses.CrossEntropyLoss import CrossEntropyLoss
from losses.AdditiveAngularMarginLoss import AdditiveAngularMarginLoss

# endregion

# region Constants
# map of argument names to the classes
EXTRACTORS: dict[str, type] = {
    "HuBERT_base": HuBERT_base,
    "HuBERT_large": HuBERT_large,
    "HuBERT_extralarge": HuBERT_extralarge,
    "Wav2Vec2_base": Wav2Vec2_base,
    "Wav2Vec2_large": Wav2Vec2_large,
    "Wav2Vec2_LV60k": Wav2Vec2_LV60k,
    "WavLM_base": WavLM_base,
    "WavLM_baseplus": WavLM_baseplus,
    "WavLM_large": WavLM_large,
    "XLSR_300M": XLSR_300M,
    "XLSR_1B": XLSR_1B,
    "XLSR_2B": XLSR_2B,
    "MelSpectrogram": MelSpectrogram,
}

CLASSIFIERS: Dict[str, Tuple[type, Dict[str, type]]] = {
    # Maps the classifier to tuples of the corresponding class and the initializable arguments
    "EmbeddingFF": (EmbeddingFF, {}),
    "FFConcat1": (FFConcat1, {}),
    "FFConcat2": (FFConcat2, {}),
    "FFConcat3": (FFConcat3, {}),
    "FFDiff": (FFDiff, {}),
    "FFDiffAbs": (FFDiffAbs, {}),
    "FFDiffQuadratic": (FFDiffQuadratic, {}),
    "FFLSTM": (FFLSTM, {}),
    "FFLSTM2": (FFLSTM2, {}),
    "GMMDiff": (GMMDiff, {"n_components": int, "covariance_type": str}),
    "LDAGaussianDiff": (LDAGaussianDiff, {}),
    "SVMDiff": (SVMDiff, {"kernel": str}),
}

# List of classifiers that are compatible with embedding-based losses
EMBEDDING_COMPATIBLE_CLASSIFIERS = [
    "EmbeddingFF",
    # Add any future embedding-compatible classifiers here
]

TRAINERS = {  # Maps the classifier to the trainer
    "EmbeddingFF": EmbeddingFFTrainer,
    "FFConcat1": FFPairTrainer,
    "FFConcat2": FFPairTrainer,
    "FFConcat3": FFPairTrainer,
    "FFDiff": FFPairTrainer,
    "FFDiffAbs": FFPairTrainer,
    "FFDiffQuadratic": FFPairTrainer,
    "FFLSTM": FFPairTrainer,
    "FFLSTM2": FFPairTrainer,
    "GMMDiff": GMMDiffTrainer,
    "LDAGaussianDiff": LDAGaussianDiffTrainer,
    "SVMDiff": SVMDiffTrainer,
}

# Define loss categories and metadata
LOSS_METADATA = {
    # Structure: "loss_name": {"type": "category", "parameters": {...}}
    "CrossEntropy": {
        "type": "standard", 
        "parameters": {}
    },
    "AdditiveAngularMargin": {
        "type": "embedding", 
        "parameters": {
            "in_features": int,
            "out_features": int,
            "margin": float,
            "s": float,
            "easy_margin": bool
        }
    },
    # For future implementation
    # "CosFace": {
    #     "type": "embedding",
    #     "parameters": {
    #         "in_features": int,
    #         "out_features": int,
    #         "margin": float,
    #         "s": float,
    #     }
    # },
    # "MagFace": {
    #     "type": "embedding",
    #     "parameters": {
    #         "in_features": int,
    #         "out_features": int,
    #         "l_a": float,
    #         "u_a": float,
    #         "l_margin": float,
    #         "u_margin": float,
    #         "scale": float,
    #     }
    # }
}

# Maps the loss name to the loss class
LOSSES = {
    "CrossEntropy": CrossEntropyLoss,
    "AdditiveAngularMargin": AdditiveAngularMarginLoss,
    # Future losses would be added here
    # "CosFace": CosFaceLoss,
    # "MagFace": MagFaceLoss,
}

DATASETS = {  # map the dataset name to the dataset class
    "ASVspoof2019LADataset_single": ASVspoof2019LADataset_single,
    "ASVspoof2019LADataset_pair": ASVspoof2019LADataset_pair,
    "ASVspoof2021LADataset_single": ASVspoof2021LADataset_single,
    "ASVspoof2021LADataset_pair": ASVspoof2021LADataset_pair,
    "ASVspoof2021DFDataset_single": ASVspoof2021DFDataset_single,
    "ASVspoof2021DFDataset_pair": ASVspoof2021DFDataset_pair,
    "InTheWildDataset_single": InTheWildDataset_single,
    "InTheWildDataset_pair": InTheWildDataset_pair,
    "MorphingDataset_single": MorphingDataset_single,
    "MorphingDataset_pair": MorphingDataset_pair,
    "ASVspoof5Dataset_single": ASVspoof5Dataset_single,
    "ASVspoof5Dataset_pair": ASVspoof5Dataset_pair,
    "SLTSSTCDataset_single": SLTSSTCDataset_single,
    "SLTSSTCDataset_pair": SLTSSTCDataset_pair,
    "SLTSSTCDataset_eval": SLTSSTCDataset_eval,
}
# endregion

def get_dataloaders(
    dataset: str,
    config: dict,
    lstm: bool = False,
    augment: bool = False,
    eval_only: bool = False,
    mode: str = "speaker_verification",
) -> Tuple[DataLoader, DataLoader, DataLoader] | DataLoader: # return training dataloader, validation dataloader, evaluation dataloader or just evaluation dataloader depending on the mode
    """Get dataloader"""
    return get_speaker_verification_dataloader(dataset, config, lstm, augment, eval_only)

def get_speaker_verification_dataloader(
    dataset: str,
    config: dict,
    lstm: bool = False,
    augment: bool = False,
    eval_only: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader] | DataLoader:
    """Get dataloaders for speaker verification mode."""
    # Get the dataset class and config
    dataset_config = {}
    t = "pair" if "pair" in dataset else "single"
    if "SLTSSTC" in dataset:
        train_dataset_class = DATASETS[dataset]
        val_dataset_class = DATASETS["SLTSSTCDataset_eval"]
        eval_dataset_class = DATASETS["SLTSSTCDataset_eval"]
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
    

def build_model(args: Namespace, num_classes: int = 2) -> Tuple[FFBase | BaseSklearnModel, BaseTrainer]:
    # Beware of MHFA or AASIST with SkLearn models, they are not implemented yet
    if args.model.processor in ["MHFA", "AASIST", "SLS"] and args.model.classifier in ["GMMDiff", "SVMDiff", "LDAGaussianDiff"]:
        raise NotImplementedError("Training of SkLearn models with MHFA, AASIST or SLS is not yet implemented.")
    # region Extractor
    extractor = EXTRACTORS[args.model.extractor]()  # map the argument to the class and instantiate it
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
    else:
        raise ValueError("Only AASIST, MHFA, Mean and SLS processors are currently supported.")
    # endregion

    # region Loss Function
    loss_fn = None
    if not args.model.loss.name in LOSSES: # invalid loss function
        raise ValueError(f"Invalid loss function, should be one of: {list(LOSSES.keys())}")
    else: # valid loss function
        # Get the loss class and metadata
        loss_class = LOSSES[args.model.loss.name]
        loss_metadata = LOSS_METADATA[args.model.loss.name]
        loss_params_dict = {}
        
        # Set parameters based on loss type
        if loss_metadata["type"] == "embedding":
            # Set input feature dimension based on feature processor
            # i.e. all processors currently maintain the extractor's dimension
            loss_params_dict['in_features'] = extractor.feature_size
            
            # Set output feature dimension (number of classes)
            loss_params_dict['out_features'] = num_classes
        
        # Add any loss-specific parameters from args
        for param_name, param_type in loss_metadata["parameters"].items():
            if hasattr(args, param_name) and getattr(args, param_name) is not None:
                param_value = getattr(args, param_name)
                loss_params_dict[param_name] = param_type(param_value)
        
        # Create the loss function instance
        loss_fn = loss_class(**loss_params_dict)
        
        # Warn if using an embedding-based loss with standard model
        if loss_metadata["type"] == "embedding" and args.model.classifier not in EMBEDDING_COMPATIBLE_CLASSIFIERS:
            print(f"WARNING: You are using an embedding-based loss ({args.model.loss}) with a non-embedding classifier ({args.model.classifier}).")
            print(f"For proper functionality, consider using one of the embedding-compatible classifiers: {EMBEDDING_COMPATIBLE_CLASSIFIERS}")
    # endregion

    # region Model and trainer
    model: FFBase | BaseSklearnModel
    trainer = None
    match args.model.classifier:
        # region Special case Sklearn models
        case "GMMDiff":
            gmm_params = {  # Dict comprehension, get gmm parameters from args and remove None values
                k: v for k, v in args.model.items() if (k in ["n_components", "covariance_type"] and k is not None)
            }
            model = GMMDiff(extractor, processor, **gmm_params if gmm_params else {})  # pass as kwargs
            trainer = GMMDiffTrainer(model)
        case "SVMDiff":
            model = SVMDiff(extractor, processor, kernel=args.model.kernel if args.model.kernel else "rbf")
            trainer = SVMDiffTrainer(model)
        case "LDAGaussianDiff":
            model = LDAGaussianDiff(extractor, processor)
            trainer = LDAGaussianDiffTrainer(model)
        # endregion
        case _: # Everything else that doesn't require special handling
            try:
                # Users must explicitly choose EmbeddingFF if they want to use embedding-based losses
                model = CLASSIFIERS[str(args.model.classifier)][0](
                    extractor, processor, loss_fn=loss_fn, in_dim=extractor.feature_size, num_classes=num_classes
                )
                trainer_class = TRAINERS[str(args.model.classifier)]
                trainer = trainer_class(model, save_embeddings=args.training.save_embeddings)
            except KeyError:
                raise ValueError(f"Invalid classifier, should be one of: {list(CLASSIFIERS.keys())}")
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
