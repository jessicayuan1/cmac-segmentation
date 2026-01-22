import torch
import numpy as np
from model_training.utils.multilabel_metrics import (
    calculate_iou_per_class,
    calculate_f1_per_class,
    calculate_recall_per_class,
)

# Validate for one epoch
def valid_one_epoch(model, dataloader, criterion, device, thresholds, n_classes = 4):
    model.eval()
    val_loss = 0.0
    total_samples = 0

    # Store metrics per class as lists to accumulate across batches
    class_ious = [[] for _ in range(n_classes)]
    class_f1s = [[] for _ in range(n_classes)]
    class_recalls = [[] for _ in range(n_classes)]
    
    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)

            batch_size = imgs.size(0)
            total_samples += batch_size
            
            outputs = model(imgs)
            loss = criterion(outputs, masks)

            val_loss += loss.item() * imgs.size(0)
            
            batch_ious = calculate_iou_per_class(outputs, masks, thresholds)
            batch_f1s = calculate_f1_per_class(outputs, masks, thresholds)
            batch_recalls = calculate_recall_per_class(outputs, masks, thresholds)

            for i in range(n_classes):
                if batch_ious[i] is not None:
                    class_ious[i].append(batch_ious[i])

                if batch_f1s[i] is not None:
                    class_f1s[i].append(batch_f1s[i])

                if batch_recalls[i] is not None:
                    class_recalls[i].append(batch_recalls[i])
    
    # Aggregate results
    epoch_loss = val_loss / total_samples

    epoch_ious = [
        sum(class_ious[i]) / len(class_ious[i]) if len(class_ious[i]) > 0 else 0.0
        for i in range(n_classes)
    ]

    epoch_f1s = [
        sum(class_f1s[i]) / len(class_f1s[i]) if len(class_f1s[i]) > 0 else 0.0
        for i in range(n_classes)
    ]

    epoch_recalls = [
        sum(class_recalls[i]) / len(class_recalls[i]) if len(class_recalls[i]) > 0 else 0.0
        for i in range(n_classes)
    ]
    
    return epoch_loss, epoch_ious, epoch_f1s, epoch_recalls