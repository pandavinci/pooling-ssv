import torch
import numpy as np

def custom_pair_batch_create(batch: list):
    """
    Custom collate_fn for the dataloader to create batches for batch training.

    Creates batches of pairs of genuine and spoofing speech for differential-based detection.
    Shorter waveforms are padded with zeros to match the length of the longest waveform in the batch.
    """
    
    # Free unused memory before creating the new batch
    # This is necessary because PyTorch has trouble with dataloader memory management
    if torch.cuda.is_available() and torch.rand(1).item() < 0.05:  # 5% chance
        torch.cuda.empty_cache()

    # Get the lengths of all tensors in the batch
    batch_size = len(batch)
    lengths_gt = torch.tensor([item[1].size(1) for item in batch])
    lengths_test = torch.tensor([item[2].size(1) for item in batch])

    # Find the maximum length
    max_length_gt = int(torch.max(lengths_gt))
    max_length_test = int(torch.max(lengths_test))

    # Pad the tensors to have the maximum length
    file_names = []
    padded_gts = torch.zeros(batch_size, max_length_gt)
    padded_tests = torch.zeros(batch_size, max_length_test)
    labels = torch.zeros(batch_size)
    for i, item in enumerate(batch):
        file_names.append(item[0])
        waveform_gt = item[1]
        waveform_test = item[2]
        padded_waveform_gt = torch.nn.functional.pad(
            waveform_gt, (0, max_length_gt - waveform_gt.size(1))
        ).squeeze(0)
        padded_waveform_test = torch.nn.functional.pad(
            waveform_test, (0, max_length_test - waveform_test.size(1))
        ).squeeze(0)
        try:  # If the label is not available (or is None), set it to np.nan
            label = torch.tensor(item[3])
        except:
            label = np.nan

        padded_gts[i] = padded_waveform_gt
        padded_tests[i] = padded_waveform_test
        labels[i] = label

    return file_names, padded_gts, padded_tests, labels

def custom_single_batch_create(batch: list):
    """
    Custom collate_fn for the dataloader to create batches for batch training.

    Creates batches of single recordings for "normal" detection.
    Shorter waveforms are padded with zeros to match the length of the longest waveform in the batch.
    """
    # Free unused memory before creating the new batch
    # This is necessary because PyTorch has trouble with dataloader memory management
    if torch.cuda.is_available() and torch.rand(1).item() < 0.05:  # 5% chance
        torch.cuda.empty_cache()

    # Get the lengths of all tensors in the batch
    batch_size = len(batch)
    lengths = torch.tensor([item[1].size(1) for item in batch])

    # Find the maximum length
    max_length = int(torch.max(lengths))

    # Pad the tensors to have the maximum length
    file_names = []
    padded_waveforms = torch.zeros(batch_size, max_length)
    labels = torch.zeros(batch_size)
    for i, item in enumerate(batch):
        file_names.append(item[0])
        waveform = item[1]
        padded_waveform = torch.nn.functional.pad(
            waveform, (0, max_length - waveform.size(1))
        ).squeeze(0)
        try:  # If the label is not available (or is None), set it to np.nan
            label = torch.tensor(item[2])
        except:
            label = np.nan

        padded_waveforms[i] = padded_waveform
        labels[i] = label

    return file_names, padded_waveforms, labels

def custom_eval_batch_create(batch: list):
    """
    Custom collate_fn for the dataloader to create batches for evaluation.
    Creates batches of pairs of recordings for evaluation.
    Shorter waveforms are padded with zeros to match the length of the longest waveform in the batch.
    """
    # Free unused memory before creating the new batch
    if torch.cuda.is_available() and torch.rand(1).item() < 0.05:  # 5% chance
        torch.cuda.empty_cache()

    # Get the lengths of all tensors in the batch
    batch_size = len(batch)
    source_lengths = torch.tensor([item[1][0].size(1) for item in batch])
    target_lengths = torch.tensor([item[1][1].size(1) for item in batch])

    # Find the maximum lengths for source and target
    max_source_length = int(torch.max(source_lengths))
    max_target_length = int(torch.max(target_lengths))

    # Pad the tensors to have the maximum lengths
    source_paths = []
    target_paths = []
    padded_source_waveforms = torch.zeros(batch_size, max_source_length)
    padded_target_waveforms = torch.zeros(batch_size, max_target_length)
    labels = torch.zeros(batch_size)

    for i, item in enumerate(batch):
        source_paths.append(item[0][0])
        target_paths.append(item[0][1])
        
        # Pad source waveform
        source_waveform = item[1][0]
        padded_source_waveform = torch.nn.functional.pad(
            source_waveform, (0, max_source_length - source_waveform.size(1))
        ).squeeze(0)
        
        # Pad target waveform
        target_waveform = item[1][1]
        padded_target_waveform = torch.nn.functional.pad(
            target_waveform, (0, max_target_length - target_waveform.size(1))
        ).squeeze(0)

        try:
            label = torch.tensor(item[2])
        except:
            label = np.nan

        padded_source_waveforms[i] = padded_source_waveform
        padded_target_waveforms[i] = padded_target_waveform
        labels[i] = label

    return (source_paths, target_paths), (padded_source_waveforms, padded_target_waveforms), labels


def custom_feature_pair_batch_create(batch: list):
    """
    Custom collate_fn for the dataloader to create batches for batch training with extracted features.

    Creates batches of pairs of genuine and spoofing speech features for differential-based detection.
    Shorter feature sequences are padded with zeros to match the length of the longest sequence in the batch.
    Expected input shape: (1, T, F) where T is time dimension and F is feature dimension.
    """
    
    # Free unused memory before creating the new batch
    # This is necessary because PyTorch has trouble with dataloader memory management
    if torch.cuda.is_available() and torch.rand(1).item() < 0.05:  # 5% chance
        torch.cuda.empty_cache()

    # Get the lengths of all feature tensors in the batch
    batch_size = len(batch)
    lengths_gt = torch.tensor([item[1].size(1) for item in batch])  # Time dimension
    lengths_test = torch.tensor([item[2].size(1) for item in batch])  # Time dimension

    # Find the maximum length and feature dimension
    max_length_gt = int(torch.max(lengths_gt))
    max_length_test = int(torch.max(lengths_test))
    feature_dim_gt = item[1].size(2)  # Feature dimension from first item
    feature_dim_test = item[2].size(2)  # Feature dimension from first item

    # Pad the tensors to have the maximum length
    file_names = []
    padded_gts = torch.zeros(batch_size, max_length_gt, feature_dim_gt)
    padded_tests = torch.zeros(batch_size, max_length_test, feature_dim_test)
    labels = torch.zeros(batch_size)
    
    for i, item in enumerate(batch):
        file_names.append(item[0])
        features_gt = item[1]  # Shape: (1, T, F)
        features_test = item[2]  # Shape: (1, T, F)
        
        # Pad along the time dimension (dimension 1)
        padded_features_gt = torch.nn.functional.pad(
            features_gt, (0, 0, 0, max_length_gt - features_gt.size(1))
        ).squeeze(0)  # Remove batch dimension
        padded_features_test = torch.nn.functional.pad(
            features_test, (0, 0, 0, max_length_test - features_test.size(1))
        ).squeeze(0)  # Remove batch dimension
        
        try:  # If the label is not available (or is None), set it to np.nan
            label = torch.tensor(item[3])
        except:
            label = np.nan

        padded_gts[i] = padded_features_gt
        padded_tests[i] = padded_features_test
        labels[i] = label

    return file_names, padded_gts, padded_tests, labels


def custom_feature_single_batch_create(batch: list):
    """
    Custom collate_fn for the dataloader to create batches for batch training with extracted features.

    Creates batches of single recordings features for "normal" detection.
    Shorter feature sequences are padded with zeros to match the length of the longest sequence in the batch.
    Expected input shape: (1, T, F) where T is time dimension and F is feature dimension.
    """
    # Free unused memory before creating the new batch
    # This is necessary because PyTorch has trouble with dataloader memory management
    if torch.cuda.is_available() and torch.rand(1).item() < 0.05:  # 5% chance
        torch.cuda.empty_cache()

    # Get the lengths of all feature tensors in the batch
    batch_size = len(batch)
    lengths = torch.tensor([item[1].size(1) for item in batch])  # Time dimension

    # Find the maximum length and feature dimension
    max_length = int(torch.max(lengths))
    feature_dim = batch[0][1].size(2)  # Feature dimension from first item

    # Pad the tensors to have the maximum length
    file_names = []
    padded_features = torch.zeros(batch_size, max_length, feature_dim)
    labels = torch.zeros(batch_size)
    
    for i, item in enumerate(batch):
        file_names.append(item[0])
        features = item[1]  # Shape: (1, T, F)
        
        # Pad along the time dimension (dimension 1)
        padded_feature = torch.nn.functional.pad(
            features, (0, 0, 0, max_length - features.size(1))
        ).squeeze(0)  # Remove batch dimension
        
        try:  # If the label is not available (or is None), set it to np.nan
            label = torch.tensor(item[2])
        except:
            label = np.nan

        padded_features[i] = padded_feature
        labels[i] = label

    return file_names, padded_features, labels


def custom_feature_eval_batch_create(batch: list):
    """
    Custom collate_fn for the dataloader to create batches for evaluation with extracted features.
    Creates batches of pairs of recordings features for evaluation.
    Shorter feature sequences are padded with zeros to match the length of the longest sequence in the batch.
    Expected input shape: (1, T, F) where T is time dimension and F is feature dimension.
    """
    # Free unused memory before creating the new batch
    if torch.cuda.is_available() and torch.rand(1).item() < 0.05:  # 5% chance
        torch.cuda.empty_cache()

    # Get the lengths of all feature tensors in the batch
    batch_size = len(batch)
    source_lengths = torch.tensor([item[1][0].size(1) for item in batch])  # Time dimension
    target_lengths = torch.tensor([item[1][1].size(1) for item in batch])  # Time dimension

    # Find the maximum lengths and feature dimensions for source and target
    max_source_length = int(torch.max(source_lengths))
    max_target_length = int(torch.max(target_lengths))
    source_feature_dim = batch[0][1][0].size(2)  # Feature dimension from first source item
    target_feature_dim = batch[0][1][1].size(2)  # Feature dimension from first target item

    # Pad the tensors to have the maximum lengths
    source_paths = []
    target_paths = []
    padded_source_features = torch.zeros(batch_size, max_source_length, source_feature_dim)
    padded_target_features = torch.zeros(batch_size, max_target_length, target_feature_dim)
    labels = torch.zeros(batch_size)

    for i, item in enumerate(batch):
        source_paths.append(item[0][0])
        target_paths.append(item[0][1])
        
        # Pad source features along time dimension
        source_features = item[1][0]  # Shape: (1, T, F)
        padded_source_feature = torch.nn.functional.pad(
            source_features, (0, 0, 0, max_source_length - source_features.size(1))
        ).squeeze(0)  # Remove batch dimension
        
        # Pad target features along time dimension
        target_features = item[1][1]  # Shape: (1, T, F)
        padded_target_feature = torch.nn.functional.pad(
            target_features, (0, 0, 0, max_target_length - target_features.size(1))
        ).squeeze(0)  # Remove batch dimension

        try:
            label = torch.tensor(item[2])
        except:
            label = np.nan

        padded_source_features[i] = padded_source_feature
        padded_target_features[i] = padded_target_feature
        labels[i] = label

    return (source_paths, target_paths), (padded_source_features, padded_target_features), labels
