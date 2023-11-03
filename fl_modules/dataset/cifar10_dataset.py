import numpy as np
from typing import Dict, Tuple, List, Union
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
from .randaug import RandAugmentMC
from .utils import load_pickle

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
                data: Union[Dict[str, Union[List[Image.Image], np.ndarray]], str],
                batch_size: int = 64,
                targets: List[str] = ['none'],
                iters: int = None
                ) -> None:
        """
        Args:
            data: A dictionary with keys 'x' and 'y' or a path to the numpy array
            targets: A list of augment_type, where augment_type is either 'strong', 'weak', 'none', default is ['none']
        """
        super(Cifar10Dataset, self).__init__()
        
        if isinstance(data, str):
            data = load_pickle(data)
        
        # Convert to PIL Image
        self.x = data['x']
        self.y = data['y'].astype(np.int64)
        
        self.batch_size = batch_size    
        self.targets = targets
        self.weak = weak_augment()
        self.strong = strong_augment()
        self.normalize = normalize()
        
        if iters != None:
            self.set_iters(iters)
        else:
            self.idx_mapping = list(range(0, len(self.x)))

    def set_iters(self, iters: int) -> None:
        self.iters = iters
        num_samples = iters * self.batch_size
        self.idx_mapping = []
        for _ in range(num_samples // len(self.x)):
            self.idx_mapping.extend(list(range(0, len(self.x))))
        
        # Add the remaining samples
        if num_samples % len(self.x) != 0:
            idxs = list(range(0, len(self.x)))
            random.shuffle(idxs)
            self.idx_mapping.extend(idxs[0:num_samples % len(self.x)])
        
    def __getitem__(self, idx: int) -> List[torch.Tensor]:
        idx = self.idx_mapping[idx]
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
        if hasattr(self, 'iters'):
            return self.iters * self.batch_size
        else:
            return len(self.x)