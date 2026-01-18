import torch
from model_training.utils.multilabel_metrics import (
    calculate_iou_per_class,
    calculate_f1_per_class,
    calculate_recall_per_class,
)

#Train for one epoch
def train_one_epoch(model, dataloader, optimizer, criterion, device, n_classes = 5):
    model.train()
    running_loss = 0.0
    total_samples = 0
    class_ious = [0.0 for _ in range(n_classes)]
    class_f1s = [0.0 for _ in range(n_classes)]
    class_recalls = [0.0 for _ in range(n_classes)]

    #loop = tqdm(dataloader, desc = "Training", leave = False) Optional for viewing
    #Replace dataloader with loop if using tqdm
    for imgs, masks in dataloader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        batch_size = imgs.size(0)
        total_samples += batch_size

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size

        # Calculate metrics per batch
        with torch.no_grad():
            batch_ious = calculate_iou_per_class(outputs, masks, num_classes = 5)
            batch_f1s = calculate_f1_per_class(outputs, masks, num_classes = 5)
            batch_recalls = calculate_recall_per_class(outputs, masks, num_classes = 5)

            for i in range(n_classes):
                class_ious[i] += batch_ious[i] * batch_size
                class_f1s[i] += batch_f1s[i] * batch_size
                class_recalls[i] += batch_recalls[i] * batch_size

    # Aggregate losses and metrics
    epoch_loss = running_loss / total_samples
    epoch_ious = [v / total_samples for v in class_ious]
    epoch_f1s = [v / total_samples for v in class_f1s]
    epoch_recalls = [v / total_samples for v in class_recalls]

    # Return metrics, loss for the epoch
    return epoch_loss, epoch_ious, epoch_f1s, epoch_recalls