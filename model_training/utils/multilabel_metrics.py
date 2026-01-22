import torch

# Multilabel segmentation metrics
# predictions: (N, 4, H, W) raw logits
# targets:     (N, 4, H, W) binary masks
# Class order: ['EX', 'HE', 'MA', 'SE']

def calculate_iou_per_class(predictions, targets, thresholds):
    """
    Returns:
        class_ious: list of 4 IoU values (one per class)
        sample_ious: (N, 4) tensor of per-sample IoUs
    """
    targets = targets.bool()
    probs = torch.sigmoid(predictions)
    
    N = predictions.shape[0]
    sample_ious = torch.zeros(N, 4)
    
    for c in range(4):
        thresh = thresholds[c] if isinstance(thresholds, (list, tuple)) else thresholds
        pred = (probs[:, c] > thresh)
        gt = targets[:, c]
        
        for i in range(N):
            pred_i = pred[i]
            gt_i = gt[i]
            
            tp = (pred_i & gt_i).float().sum()
            fp = (pred_i & ~gt_i).float().sum()
            fn = (~pred_i & gt_i).float().sum()
            
            denom = tp + fp + fn
            
            if denom > 0:
                sample_ious[i, c] = tp / denom
            else:
                # No ground truth and no prediction for this sample/class
                sample_ious[i, c] = float('nan')
    
    # Class-wise IoU: average across samples (ignoring NaNs)
    class_ious = []
    for c in range(4):
        valid_ious = sample_ious[:, c][~torch.isnan(sample_ious[:, c])]
        if len(valid_ious) > 0:
            class_ious.append(valid_ious.mean().item())
        else:
            class_ious.append(None)
    
    return class_ious, sample_ious


def calculate_f1_per_class(predictions, targets, thresholds):
    targets = targets.bool()
    probs = torch.sigmoid(predictions)
    
    N = predictions.shape[0]
    sample_f1s = torch.zeros(N, 4)
    
    for c in range(4):
        thresh = thresholds[c] if isinstance(thresholds, (list, tuple)) else thresholds
        pred = (probs[:, c] > thresh)
        gt = targets[:, c]
        
        for i in range(N):
            pred_i = pred[i]
            gt_i = gt[i]
            
            tp = (pred_i & gt_i).float().sum()
            fp = (pred_i & ~gt_i).float().sum()
            fn = (~pred_i & gt_i).float().sum()
            
            denom = 2 * tp + fp + fn
            
            if denom > 0:
                sample_f1s[i, c] = 2 * tp / denom
            else:
                sample_f1s[i, c] = float('nan')
    
    class_f1s = []
    for c in range(4):
        valid_f1s = sample_f1s[:, c][~torch.isnan(sample_f1s[:, c])]
        if len(valid_f1s) > 0:
            class_f1s.append(valid_f1s.mean().item())
        else:
            class_f1s.append(None)
    
    return class_f1s, sample_f1s


def calculate_recall_per_class(predictions, targets, thresholds):
    targets = targets.bool()
    probs = torch.sigmoid(predictions)
    
    N = predictions.shape[0]
    sample_recalls = torch.zeros(N, 4)
    
    for c in range(4):
        thresh = thresholds[c] if isinstance(thresholds, (list, tuple)) else thresholds
        pred = (probs[:, c] > thresh)
        gt = targets[:, c]
        
        for i in range(N):
            pred_i = pred[i]
            gt_i = gt[i]
            
            tp = (pred_i & gt_i).float().sum()
            fn = (~pred_i & gt_i).float().sum()
            
            denom = tp + fn
            
            if denom > 0:
                sample_recalls[i, c] = tp / denom
            else:
                sample_recalls[i, c] = float('nan')
    
    class_recalls = []
    for c in range(4):
        valid_recalls = sample_recalls[:, c][~torch.isnan(sample_recalls[:, c])]
        if len(valid_recalls) > 0:
            class_recalls.append(valid_recalls.mean().item())
        else:
            class_recalls.append(None)
    
    return class_recalls, sample_recalls


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