"""
Dataset utilities for multimodal learning.

This module provides dataset classes and data loading utilities for
handling multimodal data.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class MultimodalDataset(Dataset):
    """Base dataset class for multimodal data."""
    
    def __init__(self, data, labels, transform=None):
        """
        Initialize multimodal dataset.
        
        Args:
            data: List of data for each modality
            labels: Ground truth labels
            transform: Optional transform to be applied on samples
        """
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        """Return the size of the dataset."""
        return len(self.labels)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        sample = [modality[idx] for modality in self.data]
        label = self.labels[idx]
        
        if self.transform:
            sample = [self.transform(s) for s in sample]
        
        return sample, label


def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    """
    Create a DataLoader for the given dataset.
    
    Args:
        dataset: Dataset to load
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
    
    Returns:
        DataLoader object
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1):
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        dataset: Dataset to split
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    return torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
