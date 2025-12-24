import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
def valid_one_epoch(model, dataloader, criterion, device):
    #Evaluate model performance
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * imgs.size(0)
    return val_loss / len(dataloader.dataset)