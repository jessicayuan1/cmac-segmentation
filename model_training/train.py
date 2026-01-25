"""
This is the main training entry point.
This file:
- Defines global constants
- Builds dataloaders
- Initializes model, loss function, and optimizer
- Runs training/validation loop
- Returns model parameters and metric statistics
"""
# Standard Library Imports
import os
import sys
import time
import csv
from pathlib import Path

# Third-Party Library Imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np

# Local Imports
from HydraLANet_Definition.model.hydralanet import HydraLANet
from model_training.data_loader import get_fundus_dataloaders
from model_training.loss_functions import (
    FocalTverskyLoss,
    DualLoss,
    BCELossMultiLabel,
)
from model_training.training_loop import train_one_epoch
from model_training.valid_loop import valid_one_epoch

# =============== Global Constants =================
MODEL_NAME = "test1"

IMG_SIZE = 512
DEFAULT_EPOCHS = 75
LEARNING_RATE = 1e-5
DEFAULT_SEED = 42

BATCH_SIZE = 8
NUM_WORKERS = 6

W_FTL = 0.8
W_BCE = 0.2

TVERSKY_ALPHA = 0.5
TVERSKY_BETA = 0.5
TVERSKY_GAMMA = 2
SMOOTH = 1e-6

CLAHE_ON = False
CLAHE_CLIP = 1.5
CLAHE_MODE = 'green'

CLASS_WEIGHTS = [1.0, 1.0, 1.0, 1.0]
THRESHOLDS = [0.35, 0.35, 0.35, 0.35]

OUT_CHANNELS = 4
IN_CHANNELS = 3

OUTPUT_DIR = Path("runs") / Path(MODEL_NAME)

# =============== Main ================
def main():
    torch.manual_seed(DEFAULT_SEED)
    torch.cuda.manual_seed_all(DEFAULT_SEED)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    OUTPUT_DIR.mkdir(parents = True, exist_ok = True)

    print("CUDA available:", torch.cuda.is_available())
    print("Device:", torch.cuda.get_device_name(0))

    # ============== Model ===============
    model = HydraLANet()
    model = model.to(device)
    # ===== Freeze BatchNorm running stats (Due to low batch size) =====
    def freeze_bn(module):
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()

    model.apply(freeze_bn)
    
    # ============ Optimizer =============
    optimizer = torch.optim.AdamW(
        params = model.parameters(),
        lr = LEARNING_RATE,
        weight_decay = 1e-4
    )

    # ============== Loss =================
    loss_function = DualLoss(
        class_weights = CLASS_WEIGHTS,
        w_ft = W_FTL,
        w_bce = W_BCE,
        alpha = TVERSKY_ALPHA, 
        beta = TVERSKY_BETA, 
        gamma = TVERSKY_GAMMA,
        smooth = SMOOTH,).to(device)
    

    # ================ Data =================
    train_dataloader, val_dataloader, test_dataloader = get_fundus_dataloaders(
        resolution = IMG_SIZE,
        batch_size = BATCH_SIZE,
        data_csv_dir = "data_csv",
        use_clahe = CLAHE_ON,
        clahe_clip = CLAHE_CLIP,
        clahe_tile = (8, 8),
        clahe_mode = CLAHE_MODE,
        pin_memory = True,
        num_workers = NUM_WORKERS,
    )
    print("Train samples:", len(train_dataloader.dataset))
    print("Train batches:", len(train_dataloader))
    print("Val samples:", len(val_dataloader.dataset))
    print("Val batches:", len(val_dataloader))

    # ========== Sanity Check ==========
    images, targets = next(iter(train_dataloader))
    images, targets = images.to(device), targets.to(device)

    print("\n[TRAIN SANITY CHECK]")
    print("Targets shape:", targets.shape)
    print("Positive pixels per class:", targets.sum(dim = (0,2,3)))
    print("Unique target values:", torch.unique(targets))

    with torch.no_grad():
        probs = model(images)

        assert probs.min() >= 0.0 and probs.max() <= 1.0, \
            "Model output must be probabilities in [0, 1]"

        print(f"Train probs range: [{probs.min():.3f}, {probs.max():.3f}]")
        print(
            "Train max prob per class:",
            probs.max(dim = 0).values.max(dim = 1).values
        )
        print("Train pixels > 0.5:", (probs > 0.5).sum())

    images_v, targets_v = next(iter(val_dataloader))
    images_v, targets_v = images_v.to(device), targets_v.to(device)

    print("\n[VAL SANITY CHECK]")
    print("Targets shape:", targets_v.shape)
    print("Positive pixels per class:", targets_v.sum(dim = (0,2,3)))
    print("Unique target values:", torch.unique(targets_v))

    with torch.no_grad():
        probs_v = model(images_v)

        assert probs_v.min() >= 0.0 and probs_v.max() <= 1.0, \
            "Model output must be probabilities in [0, 1]"

        print(f"Val probs range: [{probs_v.min():.3f}, {probs_v.max():.3f}]")
        print(
            "Val max prob per class:",
            probs_v.max(dim = 0).values.max(dim = 1).values
        )
        print("Val pixels > 0.5:", (probs_v > 0.5).sum())

    # ================= Metric Storage =================
    train_losses = []
    val_losses = []

    train_ious = []
    train_f1s = []
    train_recalls = []

    val_ious = []
    val_f1s = []
    val_recalls = []

    train_mean_f1s = []
    val_mean_f1s = []
    train_mean_ious = []
    val_mean_ious = []
    train_mean_recalls = []
    val_mean_recalls = []

    best_val_f1 = -1.0

    # ===== Open CSV files =====
    CLASSES = ["EX", "HE", "MA", "SE"]

    mean_csv_path = OUTPUT_DIR / f"{MODEL_NAME}_mean.csv"
    mean_f = open(mean_csv_path, "w", newline = "")
    mean_writer = csv.writer(mean_f)
    mean_writer.writerow([
        "epoch",
        "train_loss", "val_loss",
        "train_mean_iou", "val_mean_iou",
        "train_mean_f1", "val_mean_f1",
        "train_mean_recall", "val_mean_recall"
    ])

    class_csv_path = OUTPUT_DIR / f"{MODEL_NAME}_per_class.csv"
    class_f = open(class_csv_path, "w", newline = "")
    class_writer = csv.writer(class_f)
    class_writer.writerow([
        "epoch", "split", "class",
        "iou", "f1", "recall"
    ])

    # ================= Training Loop =================
    for epoch in range(DEFAULT_EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{DEFAULT_EPOCHS}]")

        model.train()
        model.apply(freeze_bn)

        train_loss, tr_iou, tr_f1, tr_rec, tr_mean_iou, tr_mean_f1, tr_mean_rec = train_one_epoch(
            model = model,
            dataloader = train_dataloader,
            optimizer = optimizer,
            criterion = loss_function,
            device = device,
            thresholds = THRESHOLDS,
            n_classes = OUT_CHANNELS,
        )

        val_loss, v_iou, v_f1, v_rec, v_mean_iou, v_mean_f1, v_mean_rec = valid_one_epoch(
            model = model,
            dataloader = val_dataloader,
            criterion = loss_function,
            device = device,
            thresholds = THRESHOLDS,
            n_classes = OUT_CHANNELS,
        )

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
        if v_mean_f1 > best_val_f1:
            best_val_f1 = v_mean_f1
            torch.save(
                model.state_dict(),
                OUTPUT_DIR / "best_model.pt"
            )
            print(f"Saved new best model (mean val_f1 = {v_mean_f1:.4f})")

        # ===== Store mean metrics =====
        train_mean_f1s.append(tr_mean_f1)
        val_mean_f1s.append(v_mean_f1)

        train_mean_ious.append(tr_mean_iou)
        val_mean_ious.append(v_mean_iou)

        train_mean_recalls.append(tr_mean_rec)
        val_mean_recalls.append(v_mean_rec)

        # ================= Save Metrics CSV =================
        mean_writer.writerow([
            epoch + 1,
            train_loss, val_loss,
            tr_mean_iou, v_mean_iou,
            tr_mean_f1,   v_mean_f1,
            tr_mean_rec,  v_mean_rec
        ])
        mean_f.flush()
        for c, cls in enumerate(CLASSES):
            class_writer.writerow([
                epoch + 1, "train", cls,
                tr_iou[c], tr_f1[c], tr_rec[c]
            ])
            class_writer.writerow([
                epoch + 1, "val", cls,
                v_iou[c], v_f1[c], v_rec[c]
            ])
        class_f.flush()
    
    mean_f.close()
    class_f.close()
    print(f"Training complete.")

if __name__ == "__main__":
    main()