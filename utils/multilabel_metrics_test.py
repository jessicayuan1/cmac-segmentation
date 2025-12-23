import torch
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.multilabel_metrics import (
    iou_per_class,
    f1_per_class,
    recall_per_class,
    print_segmentation_metrics
)

CLASSES = ['MA', 'HE', 'EX', 'SE', 'OD']

def test_metrics_random():
    """
    Sanity test with random data.
    Checks that shapes line up and code runs without error.
    """
    print("Running random metrics test...")

    B, C, H, W = 2, 5, 32, 32

    predictions = torch.randn(B, C, H, W)
    targets = torch.randint(0, 2, (B, C, H, W))

    print("Predictions shape:", predictions.shape)
    print("Targets shape:", targets.shape)

    print("\nMetrics output:")
    print_segmentation_metrics(predictions, targets, CLASSES)

    print("\nRandom metrics test passed.\n")

def test_metrics_known_small():
    """
    Deterministic test with hand-constructed masks.
    Uses 5 classes but small spatial dimensions.
    """
    print("Running known-value metrics test...")

    B, C, H, W = 1, 5, 4, 4

    # Initialize everything to background / negative
    predictions = torch.full((B, C, H, W), -10.0)
    targets = torch.zeros((B, C, H, W))

    # ----- Class 0 (MA): perfect overlap -----
    predictions[0, 0, 0:2, 0:2] = 10.0
    targets[0, 0, 0:2, 0:2] = 1

    # ----- Class 1 (HE): partial overlap -----
    predictions[0, 1, 0:2, 0:2] = 10.0
    targets[0, 1, 1:3, 1:3] = 1

    # ----- Other classes empty -----

    print("Predicted binary masks:")
    print((torch.sigmoid(predictions) > 0.5).int())

    print("Target masks:")
    print(targets.int())

    print("\nMetrics output:")
    print_segmentation_metrics(predictions, targets, CLASSES)

    # Optional hard assertions (lightweight)
    ious = iou_per_class(predictions, targets)
    f1s = f1_per_class(predictions, targets)
    recalls = recall_per_class(predictions, targets)

    assert ious.shape == (5,)
    assert f1s.shape == (5,)
    assert recalls.shape == (5,)

    # Perfect class should be exactly 1
    assert torch.isclose(ious[0], torch.tensor(1.0), atol = 1e-4)
    assert torch.isclose(f1s[0], torch.tensor(1.0), atol = 1e-4)
    assert torch.isclose(recalls[0], torch.tensor(1.0), atol = 1e-4)

    print("\nKnown-value metrics test passed.\n")


if __name__ == "__main__":
    test_metrics_random()
    test_metrics_known_small()
