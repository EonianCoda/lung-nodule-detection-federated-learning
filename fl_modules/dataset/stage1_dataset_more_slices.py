import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple, List
from scipy import ndimage

from .utils import load_series_list, get_start_and_end_slice
from .augmentation import RandomFlipYXZ

HU_MIN, HU_MAX = -1000, 400

class Stage1Dataset(Dataset):
    def __init__(self, 
                dataset_type: str,
                nodule_size_ranges: Dict[str, Tuple[int, int]],
                num_nodules: Dict[str, int],
                series_list_path: str,
                depth: int):
        
        self.dataset_type = dataset_type
        self.nodule_size_ranges = nodule_size_ranges
        self.depth = depth
        self.stride = depth // 2
        self.num_nodules = num_nodules
        # Generate data pair for training or validating
        self.data_list = []
        self.series_data_list = []

        self.series_paths = []
        self.gt_mask_maps_paths = []
        
        self.series_nodule_slice_ids = []
        self.series_first_and_end_valid_slice = []
        
        for folder, file_name in load_series_list(series_list_path):
            series_path = os.path.join(folder, 
                        'npy', 
                        f'{file_name}.npy')
            lobe_info_path = os.path.join(folder, 
                                            'npy', 
                                            'lobe_info.txt')
            gt_mask_maps_path = os.path.join(folder,
                                    'mask', 
                                    f'{file_name}.npz')
            series_nodules_counts_path = os.path.join(folder,
                                        'mask', 
                                        f'{file_name}_nodule_count.json')
            self.series_paths.append(series_path)
            self.gt_mask_maps_paths.append(gt_mask_maps_path)
            
            # Read nodule information in series.
            with open(series_nodules_counts_path, 'r') as f:
                series_nodules_counts = json.load(f)
            nodule_start_slice_ids = series_nodules_counts['nodule_start_slice_ids']
            self.series_nodule_slice_ids.append(nodule_start_slice_ids)
            
            self.series_first_and_end_valid_slice.append(get_start_and_end_slice(lobe_info_path))
              
        self.offsets = list(range(-8, 8 + 1))
        self.shuffle_data()

    def shuffle_data(self):
        self.series_data_list = []
        # Random generate data
        for i in range(len(self.series_paths)):
            series_path = self.series_paths[i]
            gt_mask_maps_path = self.gt_mask_maps_paths[i]
            nodule_slice_ids = self.series_nodule_slice_ids[i]
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
                candidates_slice_ids.append(start_slice_id + self.stride)
            
            for slice_id in candidates_slice_ids:
                if slice_id > last_slice_id:
                    slice_id = last_slice_id
                elif slice_id < first_slice_id:
                    slice_id = first_slice_id
                start_slice_ids.add(slice_id)
                
            series_data = []
            for start_slice_id in start_slice_ids:
                series_data.append([series_path, gt_mask_maps_path, start_slice_id])
            self.series_data_list.append(series_data)

        self.data_list = []
        random.shuffle(self.series_data_list)
        for series_data in self.series_data_list:
            random.shuffle(series_data)
            self.data_list.extend(series_data)

    def load_gt_mask(self, path: str, start_slice_id: int) -> np.ndarray:
        """Load the mask image of ground truth 
        """
        gt_mask = np.load(path)['image'] # (h, w, c)
        gt_mask = gt_mask[..., start_slice_id: start_slice_id + self.depth]
        binary_masks = np.where(gt_mask > 125, 1., 0.)
        return binary_masks
    
    def load_image(self, path: str, start_slice_id: int) -> np.ndarray:
        image = np.load(path, mmap_mode='c') # (h, w, c)
        image = image[..., start_slice_id: start_slice_id + self.depth]
        image = np.clip(image, HU_MIN, HU_MAX)
        image = image - HU_MIN
        image = image.astype(np.float32) / (HU_MAX - HU_MIN)
        return image

    def augmentation(self, images: List[np.ndarray]) -> List[np.ndarray]:
        images = RandomFlipYXZ(p = 0.3)(images)
        return images

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path, mask_path, start_slice_id = self.data_list[index]
        image = self.load_image(image_path, start_slice_id)
        target512 = self.load_gt_mask(mask_path, start_slice_id)
        
        if self.dataset_type == 'train':
            image, target512 = self.augmentation([image, target512])

        # Change dimension order from (H, W, D) to (D, H, W)
        image = np.transpose(image, (2, 0, 1))  
        target512 = np.transpose(target512, (2, 0, 1))
        
        target256 = ndimage.zoom(target512, zoom=(1 / 2, 1 / 2, 1 / 2), mode="nearest", order=0)

        # Add channel dimension from (D, H, W) to (1, D, H, W)
        image = np.expand_dims(image, 0)
        target512 = np.expand_dims(target512, 0)
        target256 = np.expand_dims(target256, 0)
        
        # Convert to tensor
        image = torch.from_numpy(image.copy()).float()
        target512 = torch.from_numpy(target512.copy()).float()
        target256 = torch.from_numpy(target256.copy()).float()
        
        return image, target512, target256
    
    def __len__(self):
        return len(self.data_list)