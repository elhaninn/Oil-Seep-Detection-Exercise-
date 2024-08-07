import torch
from torch.utils.data import DataLoader
from typing import Dict
from dataset import SARImageDataset
from transform import transform
from model import UNet

def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_iou = 0.0
    num_samples = 0

    with torch.no_grad():
        for inputs, masks in dataloader:
            inputs = inputs.to(device)
            masks = masks.to(device)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            iou = MeanIoU(num_classes=8)(preds, masks)
            total_iou += iou.item() * inputs.size(0)
            num_samples += inputs.size(0)

    mean_iou = total_iou / num_samples
    return mean_iou

