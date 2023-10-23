import numpy as np
from numpy.typing import NDArray
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
        # transforms.Normalize(cifar10_mean, cifar10_std)
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

class Cifar10UnsupervisedDataset(Dataset):
    def __init__(self,
                dataset_type: str,
                data: Union[Dict[str, np.ndarray], str],
                targets: Tuple[str, str]):
        """
        Args:
            dataset_type:
                'train' or 'test'
            x:
                A numpy array of shape (N, 32, 32, 3) or a path to the numpy array
            targets:
                A tuple of (augment_type_1, augment_type_2), where augment_type_i is either 'strong', 'weak', 'none'
        """
        super(Cifar10UnsupervisedDataset, self).__init__()
        self.dataset_type = dataset_type
        
        if isinstance(data, str):
            data = np.load(data)
        
        # Convert to PIL Image
        images = []
        for i in range(len(data['x'])):
            images.append(Image.fromarray(data['x'][i]))
        self.x = images
        self.targets = targets
        
        self.weak = weak_augment()
        self.strong = strong_augment()
        self.normalize = normalize()
        
    def __getitem__(self, index: int) -> List[torch.Tensor]:
        x = self.x[index]
        data = []
        for augment_type in self.targets:
            data.append(self.normalize(self.augment(x, augment_type)))
        return data
    
    def augment(self, image: Image, augment_type: str) -> np.ndarray:
        image = image.copy()
        if augment_type == 'strong':
            image = self.strong(image)
        elif augment_type == 'weak':
            image = self.weak(image)
        return image
    
    def __len__(self) -> int:
        return len(self.x)

class Cifar10SupervisedDataset(Dataset):
    def __init__(self,
                dataset_type: str,
                data: Union[Dict[str, np.ndarray], str],
                do_augment: bool = True) -> None:
        super(Cifar10SupervisedDataset, self).__init__()
        self.dataset_type = dataset_type
        if isinstance(data, str):
            data = np.load(data)
        
        # Convert to PIL Image
        images = []
        for i in range(len(data['x'])):
            images.append(Image.fromarray(data['x'][i]))
        self.x = images
        self.y = data['y']
        # Augmentation        
        self.do_augment = do_augment
        self.normalize = normalize()
        self.weak = weak_augment()
        
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.x[index]
        y = self.y[index]
        
        x = self.normalize(self.augment(x))
        return x, y
    
    def augment(self, image: Image) -> np.ndarray:
        if self.dataset_type == 'train' and self.do_augment:
            image = self.weak(image)
        return image
    
    def __len__(self) -> int:
        return len(self.x)
        