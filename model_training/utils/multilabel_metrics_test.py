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

CLASSES = ["EX", "HE", "MA", "SE"]


def test_metrics_random():
    """
    Sanity test with random data.
    Just checks code runs and shapes align.
    """
    print("Running random metrics test...")

    B, C, H, W = 2, 4, 32, 32

    predictions = torch.randn(B, C, H, W)
    targets = torch.randint(0, 2, (B, C, H, W))

    print("Predictions shape:", predictions.shape)
    print("Targets shape:", targets.shape)

    print("\nMetrics output:")
    print_segmentation_metrics(predictions, targets)

    print("\nRandom metrics test passed.\n")


def test_metrics_known_small():
    """
    Deterministic test with hand-constructed masks.
    """
    print("Running known-value metrics test...")

    B, C, H, W = 1, 4, 4, 4

    predictions = torch.full((B, C, H, W), -10.0)
    targets = torch.zeros((B, C, H, W))

    # ----- EX (class 0): perfect overlap -----
    predictions[0, 0, 0:2, 0:2] = 10.0
    targets[0, 0, 0:2, 0:2] = 1

    # ----- HE (class 1): partial overlap -----
    predictions[0, 1, 0:2, 0:2] = 10.0
    targets[0, 1, 1:3, 1:3] = 1

    print("Predicted binary masks:")
    print((torch.sigmoid(predictions) > 0.5).int())

    print("Target masks:")
    print(targets.int())

    print("\nMetrics output:")
    print_segmentation_metrics(predictions, targets)

    # ---- Direct metric checks ----
    ious = calculate_iou_per_class(predictions, targets)
    f1s = calculate_f1_per_class(predictions, targets)
    recalls = calculate_recall_per_class(predictions, targets)

    assert len(ious) == 4
    assert len(f1s) == 4
    assert len(recalls) == 4

    # Perfect EX class
    assert abs(ious[0] - 1.0) < 1e-4
    assert abs(f1s[0] - 1.0) < 1e-4
    assert abs(recalls[0] - 1.0) < 1e-4

    print("\nKnown-value metrics test passed.\n")


if __name__ == "__main__":
    test_metrics_random()
    test_metrics_known_small()
