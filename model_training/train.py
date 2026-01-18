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
import csv
from pathlib import Path

# Third-Party Imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

# Local Imports
from CMAC_net_definition.model.CMAC import CMACNet
from model_training.data_loader import get_fundus_dataloaders
from model_training.loss_functions import (
    FocalTverskyLoss,
    DualLoss,
    BCEwithLogitsLossMultiLabel,
)
from model_training.training_loop import train_one_epoch
from model_training.valid_loop import valid_one_epoch

#Help SLURM find external files
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# =============== Global Constants  =================
MODEL_NAME = ""

IMG_SIZE = 512
DEFAULT_EPOCHS = 150
LEARNING_RATE = 1e-5
DEFAULT_SEED = 42

BATCH_SIZE = 16
NUM_WORKERS = 4

SCHEDULER_FACTOR = 0.5
SCHEDULER_EPOCHS = 10

TVERSKY_ALPHA = 0.5
TVERSKY_BETA = 0.5
TVERSKY_GAMMA = 1.3

OUT_CHANNELS = 5
IN_CHANNELS = 3

WINDOW_SIZE = 8
PATCH_SIZE = 4

OUTPUT_DIR = Path(MODEL_NAME)

# =============== Main ================
def main():
    torch.manual_seed(DEFAULT_SEED)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    OUTPUT_DIR.mkdir(parents = True, exist_ok = True)

    # ============== Model ===============
    model = CMACNet(
        in_channels = IN_CHANNELS,
        out_channels = OUT_CHANNELS,
        embed_dim = 96,
        depths = [2, 2, 6, 2],
        img_size = IMG_SIZE
    ).to(device = device)
    
    # ============ Optimizer / Scheduler =================
    optimizer = torch.optim.Adam(params = model.parameters(), lr = LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode = "min",
        factor = SCHEDULER_FACTOR,
        patience = SCHEDULER_EPOCHS,
    )

    # ============== Loss =================
    loss_function = FocalTverskyLoss(
        alpha = TVERSKY_ALPHA, 
        beta = TVERSKY_BETA, 
        gamma = TVERSKY_GAMMA)
    

    # ================ Data =================
    train_dataloader, val_dataloader, test_dataloader = get_fundus_dataloaders(
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
    # ================= Metric Storage =================
    train_losses = []
    val_losses = []

    train_ious = []
    train_f1s = []
    train_recalls = []

    val_ious = []
    val_f1s = []
    val_recalls = []

    best_val_f1 = -1.0

    # ================= Training Loop =================
    for epoch in range(DEFAULT_EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{DEFAULT_EPOCHS}]")

        train_loss, tr_iou, tr_f1, tr_rec = train_one_epoch(
            model = model,
            dataloader = train_dataloader,
            optimizer = optimizer,
            criterion = loss_function,
            device = device,
            n_classes = OUT_CHANNELS,
        )

        val_loss, v_iou, v_f1, v_rec = valid_one_epoch(
            model = model,
            dataloader = val_dataloader,
            criterion = loss_function,
            device = device,
            n_classes = OUT_CHANNELS,
        )

        scheduler.step(val_loss)

        # ===== Store Metrics =====
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_ious.append(tr_iou)
        val_ious.append(v_iou)

        train_f1s.append(tr_f1)
        val_f1s.append(v_f1)

        train_recalls.append(tr_rec)
        val_recalls.append(v_rec)

        # ===== Save Best Model (Done if Validation F1 > Best F1 so far) =====
        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            torch.save(
                model.state_dict(),
                OUTPUT_DIR / "best_model.pt"
            )
            print(f"Saved new best model (val_f1 = {v_f1:.4f})")
    # ================= Save Metrics CSV =================
    csv_path = OUTPUT_DIR / f"{MODEL_NAME}.csv"
    with open(csv_path, mode = "w", newline = "") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "val_loss",
            "train_iou", "val_iou",
            "train_f1", "val_f1",
            "train_recall", "val_recall"
        ])

        for epoch in range(DEFAULT_EPOCHS):
            writer.writerow([
                epoch + 1,
                train_losses[epoch], val_losses[epoch],
                train_ious[epoch],   val_ious[epoch],
                train_f1s[epoch],    val_f1s[epoch],
                train_recalls[epoch], val_recalls[epoch]
            ])

    print(f"Training complete.")

if __name__ == "__main__":
    main()