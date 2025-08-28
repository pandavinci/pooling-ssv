from typing import Literal
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np

from augmentation.Augment import Augmentor


class SLTSSTCDataset_preextracted_base(Dataset):
    """
    Base class for SLT Source Speaker Tracing Challenge dataset with preextracted features.
    This class loads features from compressed numpy files instead of raw audio.

    param root_dir: Path to the dataset root folder (where the .npz file is located)
    param protocol_file_name: Name of the .npz file containing preextracted features
    param variant: One of "train", "dev", "eval" to specify the dataset variant
    param augment: Whether to apply data augmentation (for training) - Note: augmentation is not supported with preextracted features
    param rir_root: Not used for preextracted features (kept for compatibility)
    param feature_transform: Not used for preextracted features (kept for compatibility)
    """

    def __init__(
        self,
        root_dir,
        protocol_file_name,
        variant: Literal["train", "dev", "eval"] = "train",
        augment=False,
        rir_root="",
        feature_transform=None
    ):
        if augment:
            print("Warning: Data augmentation is not supported with preextracted features. Disabling augmentation.")
        self.augment = False  # Disable augmentation for preextracted features
        
        self.root_dir = root_dir
        self.variant = variant
        self.feature_transform = feature_transform  # Not used but kept for compatibility
        
        # Load the preextracted features from the .npz file
        features_file = os.path.join(root_dir, protocol_file_name)
        if not os.path.exists(features_file):
            raise FileNotFoundError(f"Features file not found: {features_file}")
        
        self.data = np.load(features_file, allow_pickle=True)
        
        # Determine format based on available keys
        self.is_pair_format = 'source_features' in self.data.files
        
        if self.is_pair_format:
            self.source_features = self.data['source_features']
            self.target_features = self.data['target_features']
            self.file_paths = self.data['file_paths']
            self.labels = self.data['labels']
            self.num_samples = len(self.source_features)
        else:
            self.features = self.data['features']
            self.file_paths = self.data['file_paths']
            self.labels = self.data['labels']
            self.num_samples = len(self.features)
        
        # For compatibility with existing code
        self.num_speakers = 9027  # hardcoded - this is the number of speakers in the dataset

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        raise NotImplementedError("This method should be implemented in a specific subclass")

    def get_labels(self) -> np.ndarray:
        """
        Get the labels for all samples in the dataset.
        """
        return self.labels

    def get_class_weights(self):
        """Calculate class weights for imbalanced datasets."""
        labels = self.get_labels()
        unique_labels, counts = np.unique(labels, return_counts=True)
        total_samples = len(labels)
        class_weights = total_samples / (len(unique_labels) * counts)
        
        # Create a weight array where each index corresponds to a class
        weights = np.zeros(len(unique_labels))
        for i, label in enumerate(unique_labels):
            weights[int(label)] = class_weights[i]
        
        return torch.FloatTensor(weights)


class SLTSSTCDataset_preextracted_pair(SLTSSTCDataset_preextracted_base):
    """
    Dataset class for SLT Source Speaker Tracing Challenge with preextracted features that provides pairs of audio features.

    param root_dir: Path to the dataset root folder (where the .npz file is located)
    param protocol_file_name: Name of the .npz file containing preextracted features
    param variant: One of "train", "dev", "eval" to specify the dataset variant
    param augment: Whether to apply data augmentation (for training) - Note: augmentation is not supported with preextracted features
    param rir_root: Not used for preextracted features (kept for compatibility)
    param feature_transform: Not used for preextracted features (kept for compatibility)
    """

    def __init__(
        self,
        root_dir,
        protocol_file_name,
        variant: Literal["train", "dev", "eval"] = "train",
        augment=False,
        rir_root="",
        feature_transform=None
    ):
        super().__init__(root_dir, protocol_file_name, variant, augment, rir_root, feature_transform)
        
        if not self.is_pair_format:
            raise ValueError("Features file does not contain pair format data (missing 'source_features' or 'target_features')")

    def __getitem__(self, idx):
        """
        Returns a pair of preextracted features along with their paths and same-speaker label
        
        Args:
            idx (int): Index of the pair in the dataset
            
        Returns:
            tuple: (
                (str, str): Paths of source and target audio files
                (torch.Tensor, torch.Tensor): Preextracted features of source and target audio
                int: Label (1 if same speaker, 0 if different speakers)
            )
        """
        # Get the features and metadata
        source_features = self.source_features[idx]
        target_features = self.target_features[idx]
        file_paths = self.file_paths[idx]
        label = int(self.labels[idx])
        
        # Convert numpy arrays to torch tensors
        source_features_tensor = torch.from_numpy(source_features).float()
        target_features_tensor = torch.from_numpy(target_features).float()
        
        # Add batch dimension if not present (for compatibility with collate functions)
        if len(source_features_tensor.shape) == 2:
            source_features_tensor = source_features_tensor.unsqueeze(0)  # (1, T, F)
        if len(target_features_tensor.shape) == 2:
            target_features_tensor = target_features_tensor.unsqueeze(0)  # (1, T, F)
        
        # Extract paths
        source_path = file_paths['source'] if isinstance(file_paths, dict) else file_paths[0]
        target_path = file_paths['target'] if isinstance(file_paths, dict) else file_paths[1]
        
        return source_path, target_path, source_features_tensor, target_features_tensor, label


class SLTSSTCDataset_preextracted_single(SLTSSTCDataset_preextracted_base):
    """
    Dataset class for SLT Source Speaker Tracing Challenge with preextracted features that provides single audio features.

    param root_dir: Path to the dataset root folder (where the .npz file is located)
    param protocol_file_name: Name of the .npz file containing preextracted features
    param variant: One of "train", "dev", "eval" to specify the dataset variant
    param augment: Whether to apply data augmentation (for training) - Note: augmentation is not supported with preextracted features
    param rir_root: Not used for preextracted features (kept for compatibility)
    param feature_transform: Not used for preextracted features (kept for compatibility)
    """

    def __init__(
        self,
        root_dir,
        protocol_file_name,
        variant: Literal["train", "dev", "eval"] = "train",
        augment=False,
        rir_root="",
        feature_transform=None
    ):
        super().__init__(root_dir, protocol_file_name, variant, augment, rir_root, feature_transform)
        
        if self.is_pair_format:
            raise ValueError("Features file contains pair format data but single format dataset was requested")

    def __getitem__(self, idx):
        """
        Returns tuples of the form (audio_file_name, preextracted_features, source_speaker_label)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_path = self.file_paths[idx]
        features = self.features[idx]
        source_speaker_label = self.labels[idx]
        
        # Convert numpy array to torch tensor
        features_tensor = torch.from_numpy(features).float()
        
        # Add batch dimension if not present (for compatibility with collate functions)
        if len(features_tensor.shape) == 2:
            features_tensor = features_tensor.unsqueeze(0)  # (1, T, F)

        return file_path, features_tensor, source_speaker_label


class SLTSSTCDataset_preextracted_eval(SLTSSTCDataset_preextracted_base):
    """
    Evaluation dataset class for SLT Source Speaker Tracing Challenge with preextracted features.
    Provides pairs of preextracted features in a format compatible with the evaluation pipeline.

    param root_dir: Path to the dataset root folder (where the .npz file is located)
    param protocol_file_name: Name of the .npz file containing preextracted features
    param variant: One of "train", "dev", "eval" to specify the dataset variant
    param augment: Whether to apply data augmentation (for training) - Note: augmentation is not supported with preextracted features
    param rir_root: Not used for preextracted features (kept for compatibility)
    param feature_transform: Not used for preextracted features (kept for compatibility)
    """

    def __init__(
        self,
        root_dir,
        protocol_file_name,
        variant: Literal["train", "dev", "eval"] = "eval",
        augment=False,
        rir_root="",
        feature_transform=None
    ):
        super().__init__(root_dir, protocol_file_name, variant, augment, rir_root, feature_transform)
        
        if not self.is_pair_format:
            raise ValueError("Features file does not contain pair format data (missing 'source_features' or 'target_features')")

    def __getitem__(self, idx):
        """
        Returns evaluation pairs in a format compatible with the evaluation pipeline.
        
        Args:
            idx (int): Index of the pair in the dataset
            
        Returns:
            tuple: (
                (str, str): Paths of source and target audio files
                (torch.Tensor, torch.Tensor): Preextracted features of source and target audio
                int: Label (1 if same speaker, 0 if different speakers)
            )
        """
        # Get the features and metadata
        source_features = self.source_features[idx]
        target_features = self.target_features[idx]
        file_paths = self.file_paths[idx]
        label = int(self.labels[idx])
        
        # Convert numpy arrays to torch tensors
        source_features_tensor = torch.from_numpy(source_features).float()
        target_features_tensor = torch.from_numpy(target_features).float()
        
        # Add batch dimension if not present (for compatibility with collate functions)
        if len(source_features_tensor.shape) == 2:
            source_features_tensor = source_features_tensor.unsqueeze(0)  # (1, T, F)
        if len(target_features_tensor.shape) == 2:
            target_features_tensor = target_features_tensor.unsqueeze(0)  # (1, T, F)
        
        # Extract paths
        source_path = file_paths['source'] if isinstance(file_paths, dict) else file_paths[0]
        target_path = file_paths['target'] if isinstance(file_paths, dict) else file_paths[1]
        
        # Return data in the format expected by the evaluation pipeline
        return (source_path, target_path), (source_features_tensor, target_features_tensor), label 