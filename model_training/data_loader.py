"""
Function for building PyTorch DataLoaders for fundus image segmentation.
The function is imported into the main training file `train.py`.

This module reads preprocessed dataset splits stored as csv files
(train / validation / test) and constructs corresponding PyTorch
DataLoader objects. Training data is augmented via multiple transform
variants using ConcatDataset.

Expected files in `data_csv`:
- train_df.csv
- val_df.csv
- test_df.csv
"""
import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset
from model_training.dataset_definition import FundusSegmentationDataset
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CSV_DIR = REPO_ROOT / "data_csv"

def get_fundus_dataloaders(
    resolution,
    batch_size = 16,
    data_csv_dir = DATA_CSV_DIR,
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
    train_df = pd.read_csv(f"{data_csv_dir}/train_df.csv")
    val_df   = pd.read_csv(f"{data_csv_dir}/val_df.csv")
    test_df  = pd.read_csv(f"{data_csv_dir}/test_df.csv")

    assert set(train_df.columns) == set(val_df.columns) == set(test_df.columns)

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

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_fundus_dataloaders(
        resolution = 512,
        batch_size = 16,
        data_csv_dir = "data_csv",
        pin_memory = False,
        num_workers = 1,
    )

    print("Dataset sizes:")
    print("Train:", len(train_loader.dataset))
    print("Val:  ", len(val_loader.dataset))
    print("Test: ", len(test_loader.dataset))

    print("\nDataloader sizes (batches):")
    print("Train:", len(train_loader))
    print("Val:  ", len(val_loader))
    print("Test: ", len(test_loader))