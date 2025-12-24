import torch
import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from utils.metrics import calculate_iou_per_class, calculate_f1_per_class, calculate_recall_per_class
#Train for one epoch
def train_one_epoch(model, dataloader, optimizer, criterion, device, n_classes=5):
    model.train()
    running_loss = 0.0
    class_ious = [[] for _ in range(n_classes)]
    class_f1s = [[] for _ in range(n_classes)]
    class_recalls = [[] for _ in range(n_classes)]

    #loop = tqdm(dataloader, desc="Training", leave=False) Optional for viewing
    for imgs, masks in dataloader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

        # Calculate metrics per batch
        preds = torch.sigmoid(outputs)
        batch_ious = calculate_iou_per_class(preds, masks, n_classes=n_classes)
        batch_f1s = calculate_f1_per_class(preds, masks, n_classes=n_classes)
        batch_recalls = calculate_recall_per_class(preds, masks, n_classes=n_classes)

        for i in range(n_classes):
            class_ious[i].append(batch_ious[i])
            class_f1s[i].append(batch_f1s[i])
            class_recalls[i].append(batch_recalls[i])

    #Aggregate losses and metrics
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_ious = [sum(cls) / len(cls) for cls in class_ious]
    epoch_f1s = [sum(cls) / len(cls) for cls in class_f1s]
    epoch_recalls = [sum(cls) / len(cls) for cls in class_recalls]
    #Return metrics, loss for the epoch
    
    return epoch_loss, epoch_ious, epoch_f1s, epoch_recalls