"""
This file contains functionality to run one training epoch given a model.
The model must already have applied the sigmoid function.
"""

import torch
import tqdm
import numpy as np
from model_training.utils.multilabel_metrics import (
    calculate_iou_per_class,
    calculate_f1_per_class,
    calculate_recall_per_class,
)

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    thresholds,
    n_classes = 4,
):
    model.train()
    running_loss = 0.0
    total_samples = 0

    all_outputs = []
    all_targets = []

    for imgs, masks in dataloader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        batch_size = imgs.size(0)
        total_samples += batch_size

        optimizer.zero_grad()

        outputs = model(imgs)  # (B, C, H, W), probs in [0, 1]

        # Ensure all outputs are between 0 and 1
        assert outputs.min() >= 0.0 and outputs.max() <= 1.0, \
            "Model outputs must be probabilities in [0, 1]"

        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size

        all_outputs.append(outputs.detach().cpu())
        all_targets.append(masks.detach().cpu())

    # Dataset-level aggregation
    preds = torch.cat(all_outputs, dim = 0) # probs
    targets = torch.cat(all_targets, dim = 0) # binary masks

    epoch_loss = running_loss / total_samples

    # Probabilities + thresholds for metrics
    ious, mean_iou = calculate_iou_per_class(preds, targets, thresholds)
    f1s, mean_f1 = calculate_f1_per_class(preds, targets, thresholds)
    recalls, mean_recall = calculate_recall_per_class(preds, targets, thresholds)

    return (
        epoch_loss,
        ious,
        f1s,
        recalls,
        mean_iou,
        mean_f1,
        mean_recall,
    )
