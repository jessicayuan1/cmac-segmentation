import torch

# All metrics functions take in raw logit output of shape [batch_size, 5, dim, dim]
# Also takes in target output of same shape
# Order is ['MA', 'HE', 'EX', 'SE', 'OD']

def iou_per_class(predictions, targets, num_classes, threshold = 0.5):
    """
    Calculate IoU for each class independently (multi-label segmentation).
    predictions: [B, C, H, W] raw logits
    targets:     [B, C, H, W] binary masks
    returns:     tensor of shape [C] (one IoU per class)
    """
    preds = torch.sigmoid(predictions) > threshold
    targets = targets.bool()

    assert preds.shape[1] == num_classes
    assert targets.shape[1] == num_classes

    ious = []
    eps = 1e-6

    for c in range(num_classes):
        pred_mask = preds[:, c]
        target_mask = targets[:, c]

        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()

        iou = intersection / (union + eps)
        ious.append(iou)

    return torch.stack(ious)

def f1_per_class(predictions, targets, num_classes, threshold = 0.5):
    """
    Calculate F1 (Dice) score for each class independently (multi-label segmentation).
    predictions: [B, C, H, W] raw logits
    targets:     [B, C, H, W] binary masks
    returns:     tensor of shape [C] (one F1 per class)
    """
    preds = torch.sigmoid(predictions) > threshold
    targets = targets.bool()

    assert preds.shape[1] == num_classes
    assert targets.shape[1] == num_classes

    f1s = []
    eps = 1e-6

    for c in range(num_classes):
        pred_mask = preds[:, c]
        target_mask = targets[:, c]

        intersection = (pred_mask & target_mask).sum().float()
        pred_sum = pred_mask.sum().float()
        target_sum = target_mask.sum().float()

        f1 = (2 * intersection) / (pred_sum + target_sum + eps)
        f1s.append(f1)

    return torch.stack(f1s)

def recall_per_class(predictions, targets, num_classes, threshold = 0.5):
    """
    Calculate recall for each class independently (multi-label segmentation).
    predictions: [B, C, H, W] raw logits
    targets:     [B, C, H, W] binary masks
    returns:     tensor of shape [C] (one recall per class)
    """
    preds = torch.sigmoid(predictions) > threshold
    targets = targets.bool()

    assert preds.shape[1] == num_classes
    assert targets.shape[1] == num_classes

    recalls = []
    eps = 1e-6

    for c in range(num_classes):
        pred_mask = preds[:, c]
        target_mask = targets[:, c]

        tp = (pred_mask & target_mask).sum().float()
        fn = (~pred_mask & target_mask).sum().float()

        recall = tp / (tp + fn + eps)
        recalls.append(recall)

    return torch.stack(recalls)


def print_segmentation_metrics(predictions, targets, classes = ['MA', 'HE', 'EX', 'SE', 'OD']):
    """
    Print IoU, F1 (Dice), and Recall metrics for each class.
    Args:
        predictions: Tensor of shape (B, C, H, W) — raw logits
        targets:     Tensor of shape (B, C, H, W) — binary masks
        classes:     List of class names in order
                    ['MA', 'HE', 'EX', 'SE', 'OD']
    """
    ious = iou_per_class(predictions, targets, num_classes = 5)
    f1s = f1_per_class(predictions, targets, num_classes = 5)
    recalls = recall_per_class(predictions, targets, num_classes = 5)
    print("IoU:")
    for name, iou in zip(classes, ious):
        print(f"{name}: {iou.item():.4f}")
    print("\nF1 (Dice):")
    for name, f1 in zip(classes, f1s):
        print(f"{name}: {f1.item():.4f}")
    print("\nRecall:")
    for name, recall in zip(classes, recalls):
        print(f"{name}: {recall.item():.4f}")





