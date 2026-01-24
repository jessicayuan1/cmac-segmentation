import torch
import tqdm
import numpy as np
from model_training.utils.multilabel_metrics import (
    calculate_iou_per_class,
    calculate_f1_per_class,
    calculate_recall_per_class,
)

#Train for one epoch
def train_one_epoch(model, dataloader, optimizer, criterion, device, thresholds, n_classes = 4):
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
        outputs = model(imgs)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size

        all_outputs.append(outputs.detach().cpu())
        all_targets.append(masks.detach().cpu())

    # Concatenate all predictions and targets for full-dataset metrics
    preds = torch.cat(all_outputs, dim = 0)
    targets = torch.cat(all_targets, dim = 0)

    # Compute loss
    epoch_loss = running_loss / total_samples

    # Compute dataset-level metrics
    ious, mean_iou = calculate_iou_per_class(preds, targets, thresholds)
    f1s, mean_f1 = calculate_f1_per_class(preds, targets, thresholds)
    recalls, mean_recall = calculate_recall_per_class(preds, targets, thresholds)

    return epoch_loss, ious, f1s, recalls, mean_iou, mean_f1, mean_recall
