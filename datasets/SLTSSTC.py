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

    def __getitem__(self, idx):
        """
        To be implemented for pair-based approach
        """
        raise NotImplementedError("Pair implementation not provided")


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
