"""
This file is used for testing models that have completed training.
Models weights must be stored in a folder called runs.
This file prints out the per class and average metrics for F1, IoU, and Recall.
"""
import torch
from torch import nn
from torch.utils.data import DataLoader

from HydraLANet_Definition.model.hydralanet import HydraLANet
from model_training.data_loader import get_fundus_dataloaders
from model_training.utils.multilabel_metrics import (
    calculate_f1_per_class,
    calculate_iou_per_class,
    calculate_recall_per_class
)
from model_training.valid_loop import valid_one_epoch

device = 'cuda' if torch.cuda.is_available() else "cpu"

model = HydraLANet(snapshot = "../runs/baseline/best_model.pt").to(device)

_, _, test_dataloader = get_fundus_dataloaders(
    resolution = 1024,
    batch_size = 2,
    data_csv_dir = "data_csv",
    use_clahe = False,
    clahe_clip = 1.5,
    clahe_mode = 'lab',
    pin_memory = False,
    num_workers = 1,
)

model.eval()
total_samples = 0
thresholds = [0.35, 0.35, 0.35, 0.35]

all_outputs = []
all_targets = []

with torch.no_grad():
    for imgs, masks in test_dataloader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        batch_size = imgs.size(0)
        total_samples += batch_size

        outputs = model(imgs)

        assert outputs.min() >= 0.0 and outputs.max() <= 1.0

        all_outputs.append(outputs.cpu())
        all_targets.append(masks.cpu())
preds = torch.cat(all_outputs, dim = 0)
targets = torch.cat(all_targets, dim = 0)

ious, mean_iou = calculate_iou_per_class(preds, targets, thresholds)
f1s, mean_f1 = calculate_f1_per_class(preds, targets, thresholds)
recalls, mean_recall = calculate_recall_per_class(preds, targets, thresholds)

print(f"Mean F1: {mean_f1}")
print(f"Mean IoU: {mean_iou}")
print(f"Mean Recall: {mean_recall}")
print(f"F1 per Class - EX: {f1s[0]}, HE: {f1s[1]}, MA: {f1s[2]}, SE: {f1s[3]}")
print(f"IoU per Class - EX: {ious[0]}, HE: {ious[1]}, MA: {ious[2]}, SE: {ious[3]}")
print(f"Recall per Class - EX: {recalls[0]}, HE: {recalls[1]}, MA: {recalls[2]}, SE: {recalls[3]}")






