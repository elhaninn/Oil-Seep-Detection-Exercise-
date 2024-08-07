import torch
import numpy as np
from torchvision.transforms import functional as F
from typing import Tuple

def transform(image: np.ndarray, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply transformations to the image and mask.

    Args:
        image (numpy.ndarray): Input image.
        mask (numpy.ndarray): Corresponding mask.

    Returns:
        tuple: Transformed image and mask.
    """
    image = F.to_tensor(image).float()
    mask = torch.tensor(mask, dtype=torch.long)
    return image, mask
