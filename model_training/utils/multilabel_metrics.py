import torch

# Multilabel segmentation metrics
# predictions: (N, 4, H, W) raw logits
# targets:     (N, 4, H, W) binary masks
# Class order: ['EX', 'HE', 'MA', 'SE']

def calculate_iou_per_class(predictions, targets, thresholds):
    """
    Args:
        predictions: (N, 4, H, W) raw logits
        targets:     (N, 4, H, W) binary masks
        thresholds:  float or list of 4 floats

    Returns:
        class_ious: list of 4 floats (or None if undefined)
        mean_iou: average IoU across all valid pixels and classes
    """
    targets = targets.bool()
    probs = torch.sigmoid(predictions)

    C = predictions.shape[1]
    class_ious = []
    all_ious = []

    for c in range(C):
        thresh = thresholds[c] if isinstance(thresholds, (list, tuple)) else thresholds
        pred = (probs[:, c] > thresh)
        gt = targets[:, c]

        tp = (pred & gt).float().sum()
        fp = (pred & ~gt).float().sum()
        fn = (~pred & gt).float().sum()

        denom = tp + fp + fn

        if denom > 0:
            iou = tp / denom
            class_ious.append(iou.item())
            all_ious.append(iou)
        else:
            class_ious.append(None)

    if all_ious:
        mean_iou = torch.stack(all_ious).mean().item()
    else:
        mean_iou = None

    return class_ious, mean_iou

def calculate_f1_per_class(predictions, targets, thresholds):
    """
    Computes per-class and mean F1 (Dice) score over the entire dataset.

    Returns:
        class_f1s: list of 4 floats or None
        mean_f1: scalar mean F1 over valid classes
    """
    targets = targets.bool()
    probs = torch.sigmoid(predictions)

    C = predictions.shape[1]
    class_f1s = []
    valid_f1s = []

    for c in range(C):
        thresh = thresholds[c] if isinstance(thresholds, (list, tuple)) else thresholds
        pred = (probs[:, c] > thresh)
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
    Computes per-class and mean Recall over the entire dataset.

    Returns:
        class_recalls: list of 4 floats or None
        mean_recall: scalar mean Recall over valid classes
    """
    targets = targets.bool()
    probs = torch.sigmoid(predictions)

    C = predictions.shape[1]
    class_recalls = []
    valid_recalls = []

    for c in range(C):
        thresh = thresholds[c] if isinstance(thresholds, (list, tuple)) else thresholds
        pred = (probs[:, c] > thresh)
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

def print_segmentation_metrics(predictions, targets, thresholds=0.5):
    """
    Pretty-print IoU, F1, Recall per class with both class and overall averages.
    """
    classes = ["EX", "HE", "MA", "SE"]
    
    ious, sample_ious = calculate_iou_per_class(predictions, targets, thresholds)
    f1s, sample_f1s = calculate_f1_per_class(predictions, targets, thresholds)
    recalls, sample_recalls = calculate_recall_per_class(predictions, targets, thresholds)
    
    # Overall mean (across all samples and classes)
    mean_iou = sample_ious[~torch.isnan(sample_ious)].mean().item()
    mean_f1 = sample_f1s[~torch.isnan(sample_f1s)].mean().item()
    mean_recall = sample_recalls[~torch.isnan(sample_recalls)].mean().item()
    
    print("IoU per class:")
    for name, v in zip(classes, ious):
        if v is not None:
            print(f"  {name}: {v:.4f}")
        else:
            print(f"  {name}: N/A")
    print(f"  Mean: {mean_iou:.4f}")
    
    print("\nF1 (Dice) per class:")
    for name, v in zip(classes, f1s):
        if v is not None:
            print(f"  {name}: {v:.4f}")
        else:
            print(f"  {name}: N/A")
    print(f"  Mean: {mean_f1:.4f}")
    
    print("\nRecall per class:")
    for name, v in zip(classes, recalls):
        if v is not None:
            print(f"  {name}: {v:.4f}")
        else:
            print(f"  {name}: N/A")
    print(f"  Mean: {mean_recall:.4f}")