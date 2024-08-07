import os
from typing import Callable, Optional, Tuple
from skimage.io import imread
from torch.utils.data import Dataset
import numpy as np
import torch

class SARImageDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, transform: Optional[Callable[[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]] = None):
        """
        Args:
            image_dir (str): Directory with all the images.
            mask_dir (str): Directory with all the masks.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_names = os.listdir(image_dir)

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name = self.image_names[idx]
        image = imread(os.path.join(self.image_dir, img_name))
        mask = imread(os.path.join(self.mask_dir, img_name))
        
        if self.transform:
            image, mask = self.transform(image, mask)
        
        return image, mask
