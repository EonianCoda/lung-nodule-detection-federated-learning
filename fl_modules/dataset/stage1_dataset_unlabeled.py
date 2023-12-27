import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple, List
from scipy import ndimage

from .utils import load_series_list, get_start_and_end_slice
from .augmentation import random_color, random_blur

HU_MIN, HU_MAX = -1000, 400

class RandomFlipYXZ:
    def __init__(self, p: float = 0.5):
        self.p = p
    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns: A tuple of (images, flip_axes)
            images: A list of numpy array with shape (D, H, W)
            flip_axes: A one-hot vector with shape (3)
        """
        flip_axes = []
        for i in range(3):
            if random.random() < self.p:
                flip_axes.append(i)
        if len(flip_axes) != 0:
            image = np.flip(image, axis = flip_axes)
                
        # Convert flip axes to one-hot vector
        flip_axes = np.array(flip_axes)
        one_hot_flip_axes = np.zeros((3))
        one_hot_flip_axes[flip_axes] = 1
        
        return image, one_hot_flip_axes

class Stage1UnlabeledDataset(Dataset):
    def __init__(self, 
                series_list_path: str,
                depth: int,
                num_patches_of_each_series: int = 2):
        super(Stage1UnlabeledDataset, self).__init__()
        self.depth = depth
        self.stride = depth // 2
        self.num_patches_of_each_series = num_patches_of_each_series
        # Generate data pair for training or validating
        self.data_list = []
        self.series_data_list = []

        self.series_paths = []
        self.series_first_and_end_valid_slice = []
        
        for folder, file_name in load_series_list(series_list_path):
            series_path = os.path.join(folder, 
                        'npy', 
                        f'{file_name}.npy')
            lobe_info_path = os.path.join(folder, 
                                            'npy', 
                                            'lobe_info.txt')
            self.series_paths.append(series_path)
            
            
            self.series_first_and_end_valid_slice.append(get_start_and_end_slice(lobe_info_path))
              
        self.offsets = list(range(-8, 8 + 1))
        self.shuffle_data()

    def shuffle_data(self):
        self.series_data_list = []
        # Random generate data
        for i in range(len(self.series_paths)):
            series_path = self.series_paths[i]
            nodule_slice_ids = random.sample(range(self.series_first_and_end_valid_slice[i][0],
                                                    self.series_first_and_end_valid_slice[i][1]),
                                                self.num_patches_of_each_series)
            
            first_slice_id, end_slice_id = self.series_first_and_end_valid_slice[i]
            
            start_slice_ids = set()
            last_slice_id = end_slice_id - self.depth
            
            offset = random.choice(self.offsets)
            # Get candiate slice id based on start slice id of nodule
            candidates_slice_ids = []
            for slice_id in nodule_slice_ids:
                start_slice_id = slice_id - (slice_id % self.stride) + offset
                candidates_slice_ids.append(start_slice_id)
                candidates_slice_ids.append(start_slice_id - self.stride)
            
            for slice_id in candidates_slice_ids:
                if slice_id > last_slice_id:
                    slice_id = last_slice_id
                elif slice_id < first_slice_id:
                    slice_id = first_slice_id
                
                if slice_id + self.depth > end_slice_id:
                    slice_id = end_slice_id - self.depth
                start_slice_ids.add(slice_id)
                
            series_data = []
            for start_slice_id in start_slice_ids:
                series_data.append([series_path, start_slice_id])
            self.series_data_list.append(series_data)

        self.data_list = []
        random.shuffle(self.series_data_list)
        for series_data in self.series_data_list:
            random.shuffle(series_data)
            self.data_list.extend(series_data)

    def load_image(self, path: str, start_slice_id: int) -> np.ndarray:
        image = np.load(path, mmap_mode='c') # (h, w, c)
        image = image[..., start_slice_id: start_slice_id + self.depth]
        image = np.clip(image, HU_MIN, HU_MAX)
        image = image - HU_MIN
        image = image.astype(np.float32) / (HU_MAX - HU_MIN)
        return image

    def augmentation(self, image: np.ndarray, strong: bool = False) -> np.ndarray:
        # Random flip
        image, one_hot_flip_axes = RandomFlipYXZ(p = 0.3)([image])[0]

        if strong:
            image = random_color(image, p = 1.0, intensity = 1.0)
            image = random_blur(image, p = 0.8)
     
        return image, one_hot_flip_axes

    def __getitem__(self, index: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns: A tuple of ((image_w, image_s), (flip_axes_w, flip_axes_s))
            image_w: A tensor with shape (1, D, H, W)
                image after weak augmentation
            flip_axes_w: A tensor with shape (3)
                one-hot vector of flip axes after weak augmentation
            image_s: A tensor with shape (1, D, H, W)
                image after strong augmentation
            flip_axes_s: A tensor with shape (3)
                one-hot vector of flip axes after strong augmentation

        """
        image_path, start_slice_id = self.data_list[index]
        image = self.load_image(image_path, start_slice_id)
        
        # Augmentation
        image_w, flip_axes_w = self.augmentation(image, strong = False)
        image_s, flip_axes_s = self.augmentation(image, strong = True)

        # Change dimension order from (H, W, D) to (D, H, W)
        image_w = np.transpose(image_w, (2, 0, 1))
        image_s = np.transpose(image_s, (2, 0, 1))

        # Add channel dimension from (D, H, W) to (1, D, H, W)
        image_w = np.expand_dims(image_w, 0)
        image_s = np.expand_dims(image_s, 0)
        
        # Convert to tensor
        image_w = torch.from_numpy(image_w.copy()).float()
        flip_axes_w = torch.from_numpy(flip_axes_w.copy()).int()
        
        image_s = torch.from_numpy(image_s.copy()).float()
        flip_axes_s = torch.from_numpy(flip_axes_s.copy()).int()
        
        return (image_w, image_s), (flip_axes_w, flip_axes_s)
    
    def __len__(self):
        return len(self.data_list)