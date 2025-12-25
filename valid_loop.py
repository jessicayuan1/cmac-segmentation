import torch
from utils.multilabel_metrics import iou_per_class, f1_per_class, recall_per_class

# Validate for one epoch
def valid_one_epoch(model, dataloader, criterion, device, n_classes = 5):
    model.eval()
    val_loss = 0.0
    total_samples = 0

    class_ious = [0.0 for _ in range(n_classes)]
    class_f1s = [0.0 for _ in range(n_classes)]
    class_recalls = [0.0 for _ in range(n_classes)]
    
    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)

            batch_size = imgs.size(0)
            total_samples += batch_size
            
            outputs = model(imgs)
            loss = criterion(outputs, masks)

            val_loss += loss.item() * imgs.size(0)
            
            batch_ious = iou_per_class(outputs, masks, num_classes = 5)
            batch_f1s = f1_per_class(outputs, masks, num_classes = 5)
            batch_recalls = recall_per_class(outputs, masks, num_classes = 5)

            # Accumulate metrics
            for i in range(n_classes):
                class_ious[i] += batch_ious[i] * batch_size
                class_f1s[i] += batch_f1s[i] * batch_size
                class_recalls[i] += batch_recalls[i] * batch_size
    
    # Aggregate results
    epoch_loss = val_loss / total_samples
    epoch_ious = [v / total_samples for v in class_ious]
    epoch_f1s = [v / total_samples for v in class_f1s]
    epoch_recalls = [v / total_samples for v in class_recalls]
    
    return epoch_loss, epoch_ious, epoch_f1s, epoch_recalls