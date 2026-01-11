import torch
import sys
import os

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import print_segmentation_metrics

def test_metrics():
    """Test the print_segmentation_metrics function with dummy data"""
    print("Testing segmentation metrics...")

    # Create dummy data
    batch_size = 2
    height, width = 32, 32
    n_classes = 5 #5 classes

    # Random predictions (logits)
    predictions = torch.randn(batch_size, n_classes, height, width)

    # Random targets (class indices)
    targets = torch.randint(0, n_classes, (batch_size, height, width))

    # Class names
    classes = ['MA', 'HE', 'EX', 'SE', 'OD']

    print(f"Testing with batch_size={batch_size}, height={height}, width={width}, n_classes={n_classes}")

    # Test print function
    print("\nTesting print_segmentation_metrics:")
    print_segmentation_metrics(predictions, targets, classes, n_classes)

    print("\nTest passed!")


def test_known_values():
    """Test with known constant values where expected metrics can be calculated manually"""
    print("\nTesting with known values...")

    # Expected metrics for this test case:
    # Predictions: MA, HE; HE, MA
    # Targets: MA, MA; HE, HE
    # For MA (class 0):
    # IoU = 1/3 ≈ 0.3333, F1 = 0.5, Recall = 0.5
    # For HE (class 1):
    # IoU = 1/3 ≈ 0.3333, F1 = 0.5, Recall = 0.5

    batch_size = 1
    height, width = 2, 2
    n_classes = 2  # MA + HE

    # Create predictions (logits) that will argmax to specific classes
    predictions = torch.zeros(batch_size, n_classes, height, width)
    # Position (0,0): MA (class 0)
    predictions[0, 0, 0, 0] = 100.0
    # Position (0,1): HE (class 1)
    predictions[0, 1, 0, 1] = 100.0
    # Position (1,0): HE (class 1)
    predictions[0, 1, 1, 0] = 100.0
    # Position (1,1): MA (class 0)
    predictions[0, 0, 1, 1] = 100.0

    # Create targets
    targets = torch.tensor([[[0, 0], [1, 1]]])  # MA, MA; HE, HE

    classes = ['MA', 'HE']

    print(f"Testing with batch_size={batch_size}, height={height}, width={width}, n_classes={n_classes}")
    print("Predictions argmax:", torch.argmax(predictions, dim=1).squeeze())
    print("Targets:", targets.squeeze())

    print("\nTesting print_segmentation_metrics with known values:")
    print_segmentation_metrics(predictions, targets, classes, n_classes)

    print("\nKnown values test completed!")


if __name__ == "__main__":
    test_metrics()
    test_known_values()
