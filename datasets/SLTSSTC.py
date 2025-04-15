from typing import Literal
import torch
from torch.utils.data import Dataset
from torchaudio import load
import os
import pandas as pd
import numpy as np

from augmentation.Augment import Augmentor


class SLTSSTCDataset_base(Dataset):
    """
    Base class for the SLT Source Speaker Tracing Challenge dataset. This class should not be used directly, but rather subclassed.

    param root_dir: Path to the dataset root folder
    param protocol_file_name: Name of the CSV protocol file to use
    param variant: One of "train", "dev", "eval" to specify the dataset variant
    param augment: Whether to apply data augmentation (for training)
    param rir_root: Path to the RIR dataset for RIR augmentation
    """

    def __init__(
        self,
        root_dir,
        protocol_file_name,
        variant: Literal["train", "dev", "eval"] = "train",
        augment=False,
        rir_root="",
    ):
        # Enable data augmentation base on the argument passed, but only for training
        self.augment = False if variant != "train" else augment
        if self.augment:
            self.augmentor = Augmentor(rir_root=rir_root)

        self.root_dir = root_dir
        self.variant = variant

        # Load the CSV protocol file
        protocol_file = os.path.join(self.root_dir, protocol_file_name)
        self.protocol_df = pd.read_csv(protocol_file)
    
    def __len__(self):
        return len(self.protocol_df)

    def __getitem__(self, idx):
        raise NotImplementedError("This method should be implemented in a specific subclass")

    def get_labels(self) -> np.ndarray:
        """
        Returns an array of source speaker labels for the dataset
        Used for computing class weights for the loss function and weighted random sampling (see train.py)
        """
        return self.protocol_df["source_speaker_label"].to_numpy()

    def get_class_weights(self):
        """Returns an array of class weights for the dataset based on source speaker labels"""
        labels = self.get_labels()
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        return torch.FloatTensor(class_weights)


class SLTSSTCDataset_pair(SLTSSTCDataset_base):
    """
    Dataset class for SLT Source Speaker Tracing Challenge that provides pairs of audio.
    """

    def __init__(
        self,
        root_dir,
        protocol_file_name,
        variant: Literal["train", "dev", "eval"] = "train",
        augment=False,
        rir_root="",
    ):
        super().__init__(root_dir, protocol_file_name, variant, augment, rir_root)
        self.num_speakers = len(self.protocol_df["source_speaker_label"].unique())

    def __getitem__(self, idx):
        """
        Returns a pair of audio samples along with their paths and same-speaker label
        
        Args:
            idx (int): Index of the pair in the dataset
            
        Returns:
            tuple: (
                (str, str): Paths of source and target audio files
                (torch.Tensor, torch.Tensor): Waveforms of source and target audio
                int: Label (1 if same speaker, 0 if different speakers)
            )
        """
        # Get the row from the protocol DataFrame
        row = self.protocol_df.iloc[idx]
        
        # Get the paths for source and target files
        source_path = os.path.join(self.root_dir, row['source_file'])
        target_path = os.path.join(self.root_dir, row['target_file'])
        
        # Load the waveforms
        source_wav, _ = load(source_path)
        target_wav, _ = load(target_path)
        
        # Get the label (assuming it's already 1 or 0 in the CSV)
        label = int(row['label'])
        
        # If augmentation is enabled and we're in training mode
        if self.augment and self.variant == "train":
            source_wav = self.augmentor.augment(source_wav)
            target_wav = self.augmentor.augment(target_wav)
        
        return source_path, target_path, source_wav, target_wav, label


class SLTSSTCDataset_single(SLTSSTCDataset_base):
    """
    Dataset class for SLT Source Speaker Tracing Challenge that provides single audio files.
    """

    def __init__(
        self,
        root_dir,
        protocol_file_name,
        variant: Literal["train", "dev", "eval"] = "train",
        augment=False,
        rir_root="",
    ):
        super().__init__(root_dir, protocol_file_name, variant, augment, rir_root)
        self.num_speakers = len(self.protocol_df["source_speaker_label"].unique())

    def __getitem__(self, idx):
        """
        Returns tuples of the form (audio_file_name, waveform, source_speaker_label)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_path = self.protocol_df.loc[idx, "file_path"]
        audio_path = os.path.join(self.root_dir, file_path)
        waveform, _ = load(audio_path)

        source_speaker_label = self.protocol_df.loc[idx, "source_speaker_label"]

        if self.augment:
            waveform = self.augmentor.augment(waveform)

        return file_path, waveform, source_speaker_label


class SLTSSTCDataset_eval(SLTSSTCDataset_base):
    """
    Dataset class for SLT Source Speaker Tracing Challenge that provides pairs of audio for evaluation.
    This class is specifically designed for evaluation purposes and returns data in a format compatible
    with the evaluation pipeline.
    """

    def __init__(
        self,
        root_dir,
        protocol_file_name,
        variant: Literal["train", "dev", "eval"] = "eval",
        augment=False,
        rir_root="",
        num_speakers: int = 0,
    ):
        super().__init__(root_dir, protocol_file_name, variant, augment, rir_root)
        self.num_speakers = num_speakers

    def __getitem__(self, idx):
        """
        Returns evaluation pairs in a format compatible with the evaluation pipeline.
        
        Args:
            idx (int): Index of the pair in the dataset
            
        Returns:
            tuple: (
                (str, str): Paths of source and target audio files
                (torch.Tensor, torch.Tensor): Waveforms of source and target audio
                int: Label (1 if same speaker, 0 if different speakers)
            )
        """
        # Get the row from the protocol DataFrame
        row = self.protocol_df.iloc[idx]
        
        # Get the paths for source and target files
        source_path = os.path.join(self.root_dir, row['source_file'])
        target_path = os.path.join(self.root_dir, row['target_file'])
        
        # Load the waveforms
        source_wav, _ = load(source_path)
        target_wav, _ = load(target_path)
        
        # Get the label (assuming it's already 1 or 0 in the CSV)
        label = int(row['label'])
        
        # Return data in the format expected by the evaluation pipeline
        return (source_path, target_path), (source_wav, target_wav), label