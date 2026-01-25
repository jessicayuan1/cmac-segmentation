"""
This file contains implementation for calculating IoU, F1 and Recall Metrics for Multi-Label Binary Semantic Segmentation.
All 3 functions expect prediction probabilities and binary targets.
Thresholds for rounding up to 1 and down to 0 can also be set.
"""

import torch

def calculate_iou_per_class(predictions, targets, thresholds):
    """
    Args:
        predictions: (N, C, H, W) prediction probabilities in [0, 1]
        targets:     (N, C, H, W) binary masks {0,1}
        thresholds:  list/tuple of C floats
    Returns:
        class_ious: list of C floats
        mean_iou: scalar mean IoU over valid classes
    """
    # Safety Check
    assert predictions.min() >= 0.0 and predictions.max() <= 1.0, \
        "Predictions must be probabilities in [0, 1]"
    
    targets = targets.bool()
    probs = predictions

    C = probs.shape[1]
    class_ious = []
    valid_ious = []

    for c in range(C):
        thresh = thresholds[c] if isinstance(thresholds, (list, tuple)) else thresholds

        pred = probs[:, c] > thresh
        gt = targets[:, c]

        tp = (pred & gt).float().sum()
        fp = (pred & ~gt).float().sum()
        fn = (~pred & gt).float().sum()

        denom = tp + fp + fn

        if denom > 0:
            iou = tp / denom
            class_ious.append(iou.item())
            valid_ious.append(iou)
        else:
            class_ious.append(None)
    mean_iou = torch.stack(valid_ious).mean().item() if valid_ious else None

    return class_ious, mean_iou


def calculate_f1_per_class(predictions, targets, thresholds):
    """
    Args:
        predictions: (N, C, H, W) prediction probabilities in [0, 1]
        targets:     (N, C, H, W) binary masks {0,1}
        thresholds:  list/tuple of C floats
    Returns:
        class_f1s: list of C floats
        mean_f1: scalar mean F1 over valid classes
    """
    # Safety Check
    assert predictions.min() >= 0.0 and predictions.max() <= 1.0, \
        "Predictions must be probabilities in [0, 1]"

    targets = targets.bool()
    probs = predictions

    C = probs.shape[1]
    class_f1s = []
    valid_f1s = []

    for c in range(C):
        thresh = thresholds[c] if isinstance(thresholds, (list, tuple)) else thresholds

        pred = probs[:, c] > thresh
        gt = targets[:, c]

        tp = (pred & gt).float().sum()
        fp = (pred & ~gt).float().sum()
        fn = (~pred & gt).float().sum()

        denom = 2 * tp + fp + fn

        if denom > 0:
            f1 = 2 * tp / denom
            class_f1s.append(f1.item())
            valid_f1s.append(f1)
        else:
            class_f1s.append(None)

    mean_f1 = torch.stack(valid_f1s).mean().item() if valid_f1s else None
    return class_f1s, mean_f1

def calculate_recall_per_class(predictions, targets, thresholds):
    """
    Args:
        predictions: (N, C, H, W) prediction probabilities in [0, 1]
        targets:     (N, C, H, W) binary masks {0,1}
        thresholds:  list/tuple of C floats
    Returns:
        class_recalls: list of C floats
        mean_recall: scalar mean Recall over valid classes
    """
    # Safety Check
    assert predictions.min() >= 0.0 and predictions.max() <= 1.0, \
        "Predictions must be probabilities in [0, 1]"

    targets = targets.bool()
    probs = predictions

    C = probs.shape[1]
    class_recalls = []
    valid_recalls = []

    for c in range(C):
        thresh = thresholds[c] if isinstance(thresholds, (list, tuple)) else thresholds

        pred = probs[:, c] > thresh
        gt = targets[:, c]

        tp = (pred & gt).float().sum()
        fn = (~pred & gt).float().sum()

        denom = tp + fn

        if denom > 0:
            recall = tp / denom
            class_recalls.append(recall.item())
            valid_recalls.append(recall)
        else:
            class_recalls.append(None)

    mean_recall = torch.stack(valid_recalls).mean().item() if valid_recalls else None
    return class_recalls, mean_recall


def print_segmentation_metrics(predictions, targets, thresholds = 0.5):
    """
    Print only mean IoU, F1, and Recall.
    Assumes predictions are probabilities in [0, 1].
    """
    _, mean_iou = calculate_iou_per_class(predictions, targets, thresholds)
    _, mean_f1 = calculate_f1_per_class(predictions, targets, thresholds)
    _, mean_recall = calculate_recall_per_class(predictions, targets, thresholds)

    print("Segmentation Metrics:")
    print(f"Mean IoU: {mean_iou:.4f}" if mean_iou is not None else "Mean IoU: N/A")
    print(f"Mean F1: {mean_f1:.4f}" if mean_f1 is not None else "Mean F1: N/A")
    print(f"Mean Recall: {mean_recall:.4f}" if mean_recall is not None else "Mean Recall: N/A")
