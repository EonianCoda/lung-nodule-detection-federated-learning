import os
import random
import numpy as np
from PIL import Image
from numpy.typing import NDArray
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from .augmentation import strong_augment, weak_augment

class Cifar10UnsupervisedDataset(Dataset):
    def __init__(self,
                dataset_type: str,
                x: NDArray[np.uint8],
                targets: Tuple[str, str]):
        """
        Args:
            dataset_type:
                'train' or 'test'
            x:
                A numpy array of shape (N, 32, 32, 3)
            targets:
                A tuple of (augment_type_1, augment_type_2), where augment_type_i is either 'strong', 'weak', 'none'
        """
        super(Cifar10UnsupervisedDataset, self).__init__()
        self.dataset_type = dataset_type
        self.x = x
        self.targets = targets
    
    def __getitem__(self, index: int) -> List[torch.Tensor]:
        x = self.x[index]
        data = []
        for augment_type in self.targets:
            data.append(self.augment(x, augment_type))
        return data
    
    def augment(self, image: np.ndarray, augment_type: str) -> np.ndarray:
        image = image.copy()
        if augment_type == 'none':
            image = transforms.ToTensor()(image)
        elif augment_type == 'strong':
            image = strong_augment(image)
        elif augment_type == 'weak':
            image = weak_augment(image)
        return image
    
    def __len__(self) -> int:
        return len(self.x)

class Cifar10SupervisedDataset(Dataset):
    def __init__(self,
                dataset_type: str,
                x: NDArray[np.uint8],
                y: NDArray[np.int32],
                do_augment: bool = True) -> None:
        super(Cifar10SupervisedDataset, self).__init__()
        self.dataset_type = dataset_type

        self.x = x
        self.y = torch.from_numpy(y).long()
        self.do_augment = do_augment
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.x[index]
        y = self.y[index]
        x = self.augment(x)
        return x, y
    
    def augment(self, image: np.ndarray) -> np.ndarray:
        if self.dataset_type != 'train' or not self.do_augment:
            image = transforms.ToTensor()(image)
        else:
            image = weak_augment(image)
        return image
    
    def __len__(self) -> int:
        return len(self.x)
        