import torch
# Important make sure that predictions and targets have order:[MA, HE, EX, SE, OD]
def calculate_iou_per_class(predictions, targets, n_classes=5):
    """Calculate IoU for each class separately
    Order is same as in dataset
    """
    ious = []
    predictions = torch.argmax(predictions, dim=1)
    
    for class_id in range(0, n_classes):
        pred_mask = (predictions == class_id)
        target_mask = (targets == class_id)
        
        intersection = (pred_mask & target_mask).float().sum()
        union = (pred_mask | target_mask).float().sum()
        
        if union > 0:
            iou = intersection / union
        else:
            iou = torch.tensor(0.0)
        ious.append(iou.item())
    
    return ious


def calculate_f1_per_class(predictions, targets, n_classes=6):
    """Calculate F1 (Dice) score for each class separately
    Order is same as in dataset
    """
    f1s = []
    predictions = torch.argmax(predictions, dim=1)
    
    for class_id in range(0, n_classes):
        pred_mask = (predictions == class_id)
        target_mask = (targets == class_id)
        
        intersection = (pred_mask & target_mask).float().sum()
        pred_sum = pred_mask.float().sum()
        target_sum = target_mask.float().sum()
        
        if pred_sum + target_sum > 0:
            f1 = 2 * intersection / (pred_sum + target_sum)
        else:
            f1 = torch.tensor(0.0)
        f1s.append(f1.item())
    
    return f1s


def calculate_recall_per_class(predictions, targets, n_classes=6):
    """Calculate Recall for each class separately
    Order is same as in dataset
    """
    recalls = []
    predictions = torch.argmax(predictions, dim=1)
    
    for class_id in range(0, n_classes):
        pred_mask = (predictions == class_id)
        target_mask = (targets == class_id)
        
        intersection = (pred_mask & target_mask).float().sum()
        target_sum = target_mask.float().sum()
        
        if target_sum > 0:
            recall = intersection / target_sum
        else:
            recall = torch.tensor(0.0)
        recalls.append(recall.item())
    
    return recalls


def print_segmentation_metrics(predictions, targets, classes, n_classes=6):
    """Print IoU, F1, and Recall metrics for each class in the specified order.
    
    Args:
        predictions: Model predictions (logits or probabilities)
        targets: Ground truth labels
        classes: List of class names in order ['MA', 'HE', 'EX', 'SE', 'OD']
        n_classes: Number of classes including background (default 6)
    """
    ious = calculate_iou_per_class(predictions, targets, n_classes)
    f1s = calculate_f1_per_class(predictions, targets, n_classes)
    recalls = calculate_recall_per_class(predictions, targets, n_classes)
    
    print("IoU:")
    for name, iou in zip(classes, ious):
        print(f"{name}: {iou:.4f}")
    
    print("\nF1:")
    for name, f1 in zip(classes, f1s):
        print(f"{name}: {f1:.4f}")
    
    print("\nRecall:")
    for name, recall in zip(classes, recalls):
        print(f"{name}: {recall:.4f}")