import torch
import sys
from pathlib import Path

# ---- Force repo root ----
REPO_ROOT = Path("/Users/liu_michael/Documents/fundus-image-segmentation").resolve()
sys.path.insert(0, str(REPO_ROOT))

from model_training.utils.multilabel_metrics import (
    calculate_iou_per_class,
    calculate_f1_per_class,
    calculate_recall_per_class,
    print_segmentation_metrics
)

# ================= Tests =================

def main():
    torch.manual_seed(0)

    N, C, H, W = 2, 4, 8, 8
    thresholds = [0.5] * C

    targets = torch.zeros(N, C, H, W)
    targets[:, :, 2:6, 2:6] = 1.0

    print("\n=== Perfect Prediction ===")
    preds = targets.clone()
    print_segmentation_metrics(preds, targets, thresholds)

    print("\n=== All-Zero Prediction ===")
    preds = torch.zeros_like(targets)
    print_segmentation_metrics(preds, targets, thresholds)

    print("\n=== All-One Prediction ===")
    preds = torch.ones_like(targets)
    print_segmentation_metrics(preds, targets, thresholds)

    print("\n=== Random Uniform Prediction ===")
    preds = torch.rand_like(targets)
    print_segmentation_metrics(preds, targets, thresholds)

    print("\n=== Biased Random (Weak Model) ===")
    preds = torch.rand_like(targets) * 0.4
    preds[targets == 1] += 0.3
    preds = preds.clamp(0, 1)
    print_segmentation_metrics(preds, targets, thresholds)


if __name__ == "__main__":
    main()