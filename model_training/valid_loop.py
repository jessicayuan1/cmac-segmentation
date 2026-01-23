import torch
import numpy as np
from model_training.utils.multilabel_metrics import (
    calculate_iou_per_class,
    calculate_f1_per_class,
    calculate_recall_per_class,
)

def valid_one_epoch(
    model,
    dataloader,
    criterion,
    device,
    thresholds,
    n_classes = 4,
):
    model.eval()
    val_loss = 0.0
    total_samples = 0

    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            batch_size = imgs.size(0)
            total_samples += batch_size

            # -------------------------
            # Forward: model returns PROBABILITIES
            # -------------------------
            outputs = model(imgs)  # (B, C, H, W), probs in [0, 1]

            # Safety check (debug-stage)
            assert outputs.min() >= 0.0 and outputs.max() <= 1.0, \
                "Model outputs must be probabilities in [0, 1]"

            loss = criterion(outputs, masks)

            val_loss += loss.item() * batch_size

            all_outputs.append(outputs.cpu())
            all_targets.append(masks.cpu())

    # -------------------------
    # Dataset-level aggregation
    # -------------------------
    preds = torch.cat(all_outputs, dim = 0)
    targets = torch.cat(all_targets, dim = 0)

    epoch_loss = val_loss / total_samples

    # Metrics expect probabilities + thresholds
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
