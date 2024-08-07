import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict
from datasets import SARImageDataset
from model import UNet
from transform import transform

def train_model(model: nn.Module, dataloaders: Dict[str, DataLoader], criterion: nn.Module, optimizer: optim.Optimizer, num_epochs: int = 25) -> nn.Module:
    best_model_wts = model.state_dict()
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            
            for inputs, masks in dataloaders[phase]:
                inputs = inputs.to(device)
                masks = masks.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, masks)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f}')
            
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()
    
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    image_dataset = SARImageDataset('seep_detection/train_images_256/', 'seep_detection/train_masks_256/', transform=transform)
    dataset_size = len(image_dataset)
    val_split = 0.2
    val_size = int(np.floor(val_split * dataset_size))
    train_size = dataset_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(image_dataset, [train_size, val_size])

    dataloaders: Dict[str, DataLoader] = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=32, shuffle=False)
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=1, out_channels=8).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    trained_model = train_model(model, dataloaders, criterion, optimizer, num_epochs=20)
