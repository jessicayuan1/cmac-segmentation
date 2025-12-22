"""
Function for building PyTorch DataLoaders for fundus image segmentation.
The function is imported into the main training file `train.py`.

This module reads preprocessed dataset splits stored as pickle files
(train / validation / test) and constructs corresponding PyTorch
DataLoader objects. Training data is augmented via multiple transform
variants using ConcatDataset.

Expected files in `data_csv_dir`:
- train_df.pkl
- val_df.pkl
- test_df.pkl
"""
import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset
from dataset_definition import FundusSegmentationDataset

def get_fundus_dataloaders(
    resolution,
    batch_size = 16,
    data_csv_dir = "data_csv",
    pin_memory = False,
    num_workers = 1
):
    """
    Build PyTorch DataLoaders for fundus image segmentation.

    Loads train, validation, and test dataset splits and constructs DataLoader objects at a fixed image resolution.
    Training data is augmented by concatenating multiple transform variants, while validation and test sets use deterministic
    transforms only.
    Arguments:
        resolution (int):
            Target image resolution (either 512, 768, or 1024).
        batch_size (int):
            Number of samples per batch. Defaults to 16.
        data_csv_dir (str, optional):
            Directory containing dataset DataFrames as .csv files.
            Defaults to "data_csv".
        pin_memory (bool, optional):
            Whether to enable pinned memory for faster CPU → GPU transfers.
            Should be True when training on CUDA. Defaults to False.
        num_workers (int, optional):
            Number of subprocesses used for data loading. Defaults to 1.
    Returns (in order):
        train_loader (DataLoader): DataLoader for training data.
        val_loader   (DataLoader): DataLoader for validation data.
        test_loader  (DataLoader): DataLoader for test data.
    """
    train_df = pd.read_pickle(f"{data_csv_dir}/train_df.pkl")
    val_df   = pd.read_pickle(f"{data_csv_dir}/val_df.pkl")
    test_df  = pd.read_pickle(f"{data_csv_dir}/test_df.pkl")

    train_transforms = ["t1", "t2", "t3", "t4", "t5", "t6", "t7"]
    train_ds = ConcatDataset(
        [
            FundusSegmentationDataset(train_df, resolution, transform_type = t)
            for t in train_transforms
        ]
    )
    val_ds = FundusSegmentationDataset(val_df, resolution, transform_type = "test")
    test_ds = FundusSegmentationDataset(test_df, resolution, transform_type = "test")

    train_loader = DataLoader(
        train_ds,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        pin_memory = pin_memory,
        persistent_workers = num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = pin_memory,
        persistent_workers = num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = pin_memory,
        persistent_workers = num_workers > 0,
    )
    return train_loader, val_loader, test_loader

