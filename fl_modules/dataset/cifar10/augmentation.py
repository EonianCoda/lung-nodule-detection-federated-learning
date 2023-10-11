from torchvision import transforms
import torch

from PIL import Image
import random
from .randaug import RandAugment


def weak_augment(image: torch.Tensor) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.25),
    ])
    image = transform(image)
    image = RandomShift(p=1.0)(image)
    return image

def strong_augment(image: torch.Tensor) -> torch.Tensor:
    image = Image.fromarray(image)
    image = RandAugment()(image)
    image = transforms.ToTensor()(image)
    return image

class RandomShift():
    def __init__(self,
                 p = 0.5, 
                 y_range = [-2,2], 
                 x_range = [-2,2]):
        self.p = p
        self.y_range = y_range
        self.x_range = x_range
        
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Tensor 
                Image to be shifted. Shape (C, H, W)
        """
        if random.random() > self.p:
            return image
        
        shift_y = random.randint(self.y_range[0], self.y_range[1])
        shift_x = random.randint(self.x_range[0], self.x_range[1])
        
        out = torch.zeros_like(image)
        
        src_start_y = max(0, shift_y)
        src_end_y = min(image.shape[1], image.shape[1] + shift_y)
        src_start_x = max(0, shift_x)
        src_end_x = min(image.shape[2], image.shape[2] + shift_x)
        
        dst_start_y = max(0, -shift_y)
        dst_end_y = min(image.shape[1], image.shape[1] - shift_y)
        dst_start_x = max(0, -shift_x)
        dst_end_x = min(image.shape[2], image.shape[2] - shift_x)
        
        out[:, dst_start_y:dst_end_y, dst_start_x:dst_end_x] = image[:, src_start_y:src_end_y, src_start_x:src_end_x]
        return out 