"""
This is the main training entry point.
- Global constants
- Builds dataloaders
- Initializes model, loss, optimizer
- Runs training/validation loop
"""
# Standard Imports
import os
import sys
import time
import argparse
from pathlib import Path

# Third-Party Imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

# Local Imports
from swin_unet_definition.model.swin_unet import SwinUNet
from CMAC_net_definition.model.CMAC import CMACNet
from data_loader import get_fundus_dataloaders
from loss_functions import FocalTverskyLoss
from training_loop import train_one_epoch
from valid_loop import valid_one_epoch

# =============== Global Constants (Adjusted depending on the experiment) =================
IMG_SIZE = 1024
DEFAULT_EPOCHS = 100
LEARNING_RATE = 1e-5
DEFAULT_SEED = 42
NUM_WORKERS = 4
SCHEDULER_FACTOR = 0.5
SCHEDULER_EPOCHS = 10
TVERSKY_ALPHA = 0.5
TVERSKY_BETA = 0.5
TVERSKY_GAMMA = 1.3
BATCH_SIZE = 16
OUT_CHANNELS = 5
IN_CHANNELS = 3
WINDOW_SIZE = 8
PATCH_SIZE = 4

# =============== Initializations/Main Function ==================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SwinUNet(
        img_size = IMG_SIZE,
        patch_size = PATCH_SIZE,
        embed_dim = 96,
        depths = [2, 2, 6, 2], 
        num_heads = [3, 6, 12, 24], 
        window_size = WINDOW_SIZE,
        mlp_ratio = 4,
        out_channels = OUT_CHANNELS,
        in_channels = 3
    ).to(device = device)
    """
    model = CMACNet(
        in_channels = IN_CHANNELS,
        out_channels = OUT_CHANNELS,
        embed_dim = 96,
        depths = [2, 2, 6, 2],
        img_size = IMG_SIZE
    ).to(device = device)
    """
    optimizer = torch.optim.Adam(params = model.parameters(), lr = LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode = "min",
        factor = SCHEDULER_FACTOR,
        patience = SCHEDULER_EPOCHS,
        verbose = False,
    )
    loss_function = FocalTverskyLoss(
        alpha = TVERSKY_ALPHA, 
        beta = TVERSKY_BETA, 
        gamma = TVERSKY_GAMMA)

    train_dataloader, val_dataloader, test_dataloader = get_fundus_dataloaders(
        resolution = IMG_SIZE,
        batch_size = BATCH_SIZE,
        pin_memory = True,
        num_workers = NUM_WORKERS
    )


if __name__ == "__main__":
    main()