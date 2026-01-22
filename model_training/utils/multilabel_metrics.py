import torch

# Multilabel segmentation metrics
# predictions: (N, 4, H, W) raw logits
# targets:     (N, 4, H, W) binary masks
# Class order: ['EX', 'HE', 'MA', 'SE']

def calculate_iou_per_class(predictions, targets, thresholds):
    targets = targets.bool()
    probs = torch.sigmoid(predictions)

    ious = []

    for c in range(4):
        thresh = thresholds[c]
        pred = (probs[:, c] > thresh)
        gt = targets[:, c]

        tp = (pred & gt).float().sum()
        fp = (pred & ~gt).float().sum()
        fn = (~pred & gt).float().sum()

        denom = tp + fp + fn

        if denom == 0:
            ious.append(None)
        else:
            ious.append((tp / denom).item())

    return ious

def calculate_f1_per_class(predictions, targets, thresholds):
    targets = targets.bool()
    probs = torch.sigmoid(predictions)

    f1s = []

    for c in range(4):
        thresh = thresholds[c]
        pred = (probs[:, c] > thresh)
        gt = targets[:, c]

        tp = (pred & gt).float().sum()
        fp = (pred & ~gt).float().sum()
        fn = (~pred & gt).float().sum()

        denom = 2 * tp + fp + fn

        if denom == 0:
            f1s.append(None)
        else:
            f1s.append((2 * tp / denom).item())

    return f1s

def calculate_recall_per_class(predictions, targets, thresholds):
    targets = targets.bool()
    probs = torch.sigmoid(predictions)

    recalls = []

    for c in range(4):
        thresh = thresholds[c]
        pred = (probs[:, c] > thresh)
        gt = targets[:, c]

        tp = (pred & gt).float().sum()
        fn = (~pred & gt).float().sum()

        denom = tp + fn

        if denom == 0:
            recalls.append(None)
        else:
            recalls.append((tp / denom).item())

    return recalls


def print_segmentation_metrics(predictions, targets, threshold = 0.5):
    """
    Pretty-print IoU, F1, Recall per class.
    """
    classes = ["EX", "HE", "MA", "SE"]

    ious = calculate_iou_per_class(predictions, targets, threshold)
    f1s = calculate_f1_per_class(predictions, targets, threshold)
    recalls = calculate_recall_per_class(predictions, targets, threshold)

    print("IoU:")
    for name, v in zip(classes, ious):
        print(f"{name}: {v:.4f}")

    print("\nF1 (Dice):")
    for name, v in zip(classes, f1s):
        print(f"{name}: {v:.4f}")

    print("\nRecall:")
    for name, v in zip(classes, recalls):
        print(f"{name}: {v:.4f}")
