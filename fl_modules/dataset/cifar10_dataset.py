import numpy as np
from typing import Dict, Tuple, List, Union

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
from .randaug import RandAugmentMC

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

def normalize() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])

def weak_augment() -> transforms.Compose:
    return transforms.Compose([transforms.RandomHorizontalFlip(),
                                 transforms.RandomCrop(size=32,
                                                    padding=int(32*0.125),
                                                    padding_mode='reflect')])

def strong_augment(img_size: int = 32) -> transforms.Compose:
    return transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(size=img_size,
                                                    padding=int(img_size*0.125),
                                                    padding_mode='reflect'),
                                RandAugmentMC(n=2, m=10)])

class Cifar10Dataset(Dataset):
    def __init__(self,
                data: Union[Dict[str, np.ndarray], str],
                batch_size: int = 64,
                targets: List[str] = ['none']) -> None:
        """
        Args:
            data: A dictionary with keys 'x' and 'y' or a path to the numpy array
            targets: A list of augment_type, where augment_type is either 'strong', 'weak', 'none', default is ['none']
        """
        super(Cifar10Dataset, self).__init__()
        
        if isinstance(data, str):
            data = np.load(data)
        
        # Convert to PIL Image
        self.x = []
        self.y = data['y'].astype(np.int64)
        for i in range(len(data['x'])):
            self.x.append(Image.fromarray(data['x'][i]))
        
        self.batch_size = batch_size    
        self.targets = targets
        self.weak = weak_augment()
        self.strong = strong_augment()
        self.normalize = normalize()
        
    def __getitem__(self, idx: int) -> List[torch.Tensor]:
        x = self.x[idx]
        y = self.y[idx]
        
        if len(self.targets) == 1:
            return self.normalize(self.augment(x, self.targets[0])), y
        else:
            images = []
            for augment_type in self.targets:
                images.append(self.normalize(self.augment(x, augment_type)))
            
            return images, y
    
    def augment(self, image: Image, augment_type: str) -> np.ndarray:
        if augment_type == 'strong':
            image = self.strong(image)
        elif augment_type == 'weak':
            image = self.weak(image)
        elif augment_type == 'none':
            pass
        return image
    
    def __len__(self) -> int:
        return len(self.x)