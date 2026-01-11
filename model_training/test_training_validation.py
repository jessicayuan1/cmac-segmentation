"""
Test script to run one training and validation epoch with CMAC or Swin model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Local Imports
from swin_unet_definition.model.swin_unet import SwinUNet
from CMAC_net_definition.model.CMAC import CMACNet
from model_training.data_loader import get_fundus_dataloaders
from model_training.loss_functions import (
    FocalTverskyLoss,
    DualLoss,
    BCEwithLogitsLossMultiLabel,
)
from model_training.training_loop import train_one_epoch
from model_training.valid_loop import valid_one_epoch

# Constants
IMG_SIZE = 512
BATCH_SIZE = 2 
OUT_CHANNELS = 5
IN_CHANNELS = 3
WINDOW_SIZE = 8
PATCH_SIZE = 4

def test_training_validation():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Choose model: SwinUNet or CMACNet
    model = SwinUNet(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=WINDOW_SIZE,
        mlp_ratio=4,
        out_channels=OUT_CHANNELS,
        in_channels=IN_CHANNELS
    ).to(device)

    # Alternative: CMACNet
    #model = CMACNet(
         #in_channels=IN_CHANNELS,
         #out_channels=OUT_CHANNELS,
         #embed_dim=96,
         #depths=[2, 2, 6, 2],
         #img_size=IMG_SIZE
    #).to(device)
    #Just testing params
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5)
    criterion = FocalTverskyLoss(alpha=0.5, beta=0.5, gamma=1.3)

    # Get dataloaders (using smaller resolution and batch size for testing)
    train_dataloader, val_dataloader, _ = get_fundus_dataloaders(
        resolution=IMG_SIZE,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        num_workers=0  # For testing, avoid multiprocessing issues
    )

    print(f"Train batches: {len(train_dataloader)}, Val batches: {len(val_dataloader)}")

    # Run one training epoch
    print("Running one training epoch...")
    train_result = train_one_epoch(model, train_dataloader, optimizer, criterion, device, n_classes=OUT_CHANNELS)
    print(f"Training result: {train_result}")

    # Run one validation epoch
    print("Running one validation epoch...")
    val_loss = valid_one_epoch(model, val_dataloader, criterion, device)
    print(f"Validation loss: {val_loss:.4f}")

    print("Test completed!")

if __name__ == "__main__":
    test_training_validation()