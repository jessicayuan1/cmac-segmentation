import torch

# Multilabel segmentation metrics
# predictions: (N, 4, H, W) raw logits
# targets:     (N, 4, H, W) binary masks
# Class order: ['EX', 'HE', 'MA', 'SE']


def calculate_iou_per_class(predictions, targets, threshold = 0.5):
    """
    IoU per class for multilabel segmentation.
    """
    assert predictions.shape == targets.shape, "Predictions and targets must have same shape"
    assert predictions.shape[1] == 4, "Expected 4 classes: ['EX', 'HE', 'MA', 'SE']"

    preds = (torch.sigmoid(predictions) > threshold)
    targets = targets.bool()

    ious = []

    for c in range(4):
        pred_mask = preds[:, c]
        target_mask = targets[:, c]

        intersection = (pred_mask & target_mask).float().sum()
        union = (pred_mask | target_mask).float().sum()

        iou = intersection / union if union > 0 else torch.tensor(0.0, device = predictions.device)
        ious.append(iou.item())

    return ious


def calculate_f1_per_class(predictions, targets, threshold = 0.5):
    """
    F1 / Dice per class.
    """
    assert predictions.shape == targets.shape, "Predictions and targets must have same shape"
    assert predictions.shape[1] == 4, "Expected 4 classes: ['EX', 'HE', 'MA', 'SE']"

    preds = (torch.sigmoid(predictions) > threshold)
    targets = targets.bool()

    f1s = []

    for c in range(4):
        pred_mask = preds[:, c]
        target_mask = targets[:, c]

        intersection = (pred_mask & target_mask).float().sum()
        denom = pred_mask.float().sum() + target_mask.float().sum()

        f1 = 2 * intersection / denom if denom > 0 else torch.tensor(0.0, device = predictions.device)
        f1s.append(f1.item())

    return f1s


def calculate_recall_per_class(predictions, targets, threshold = 0.5):
    """
    Recall per class.
    """
    assert predictions.shape == targets.shape, "Predictions and targets must have same shape"
    assert predictions.shape[1] == 4, "Expected 4 classes: ['EX', 'HE', 'MA', 'SE']"

    preds = (torch.sigmoid(predictions) > threshold)
    targets = targets.bool()

    recalls = []

    for c in range(4):
        pred_mask = preds[:, c]
        target_mask = targets[:, c]

        intersection = (pred_mask & target_mask).float().sum()
        target_sum = target_mask.float().sum()

        recall = intersection / target_sum if target_sum > 0 else torch.tensor(0.0, device = predictions.device)
        recalls.append(recall.item())

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
