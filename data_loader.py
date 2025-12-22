import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset
from dataset_definition import FundusSegmentationDataset

def get_fundus_dataloaders(
    resolution,
    batch_size = 16,
    data_csv_dir = "data_csv",
):
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

    train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_ds, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(test_ds, batch_size = batch_size, shuffle = False)

    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = get_fundus_dataloaders(
    resolution = 1024,
    batch_size = 16,
)

