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
from torch.utils.data import DataLoader
from pathlib import Path

from model_training.dataset_definition import FundusSegmentationDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CSV_DIR = REPO_ROOT / "data_csv"

def get_fundus_dataloaders(
    resolution,
    batch_size = 16,
    data_csv_dir = DATA_CSV_DIR,
    use_clahe = False,
    clahe_clip = 2.5,
    clahe_tile = (8, 8),
    clahe_mode = "lab",
    pin_memory = False,
    num_workers = 1,
):
    """
    Build PyTorch DataLoaders for fundus image segmentation
    using precomputed CSV dataset splits.
    """

    train_df = pd.read_csv(Path(data_csv_dir) / "train_df.csv")
    val_df   = pd.read_csv(Path(data_csv_dir) / "val_df.csv")
    test_df  = pd.read_csv(Path(data_csv_dir) / "test_df.csv")

    DATASET_ROOT = REPO_ROOT  # or wherever your datasets are stored

    train_df["image_path"] = train_df["image_path"].apply(lambda x: str(DATASET_ROOT / x))
    val_df["image_path"]   = val_df["image_path"].apply(lambda x: str(DATASET_ROOT / x))
    test_df["image_path"]  = test_df["image_path"].apply(lambda x: str(DATASET_ROOT / x))

    assert set(train_df.columns) == set(val_df.columns) == set(test_df.columns)

    train_ds = FundusSegmentationDataset(
        train_df,
        dimensions = resolution,
        transform_type = "train",
        use_clahe = use_clahe,
        clahe_clip = clahe_clip,
        clahe_tile = clahe_tile,
        clahe_mode = clahe_mode,
    )
    val_ds = FundusSegmentationDataset(
        val_df,
        dimensions = resolution,
        transform_type = "eval",
        use_clahe = use_clahe,
        clahe_clip = clahe_clip,
        clahe_tile = clahe_tile,
        clahe_mode = clahe_mode,
    )
    test_ds = FundusSegmentationDataset(
        test_df,
        dimensions = resolution,
        transform_type = "eval",
        use_clahe = use_clahe,
        clahe_clip = clahe_clip,
        clahe_tile = clahe_tile,
        clahe_mode = clahe_mode,
    )
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
        use_clahe = True,
        clahe_clip = 2.0,
        clahe_tile = (8, 8),
        clahe_mode = "lab",
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