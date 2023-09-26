import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple, List
from scipy import ndimage
from sklearn.cluster import KMeans
from collections import defaultdict


from .utils import load_series_list, get_start_and_end_slice, get_nodule_type
from .augmentation import RandomFlipYXZ, RandomRotation, random_color, RandomRotation90

HU_MIN, HU_MAX = -1000, 400

def z_norm(feat: np.ndarray):
    feat = (feat - np.mean(feat)) / np.sqrt(np.var(feat) + 1e-8)
    return feat

def patch_is_background(background_slice_ids: np.ndarray, start_slice_id: int, depth: int) -> bool:
    if start_slice_id + depth > len(background_slice_ids):
        return 0
    elif np.all(background_slice_ids[start_slice_id: start_slice_id + depth]):
        return 1
    return 0

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
        
        self.nodules_keys = []
        series_nodule_sizes = []
        self.series_first_and_end_valid_slice = []
        self.series_nodule_background_slice_ids = []
        
        for series_idx, (folder, file_name) in enumerate(load_series_list(series_list_path)):
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
            nodule_bboxes = np.array(series_nodules_counts['bboxes'], dtype = np.int32)
            min_zs, max_zs = nodule_bboxes[:, 0, 2], nodule_bboxes[:, 1, 2]
            
            nodule_sizes = series_nodules_counts['nodule_size']
            for min_z, max_z in zip(min_zs, max_zs):
                nodules_key = [series_idx, min_z, max_z]
                self.nodules_keys.append(nodules_key)
            series_nodule_sizes.extend(nodule_sizes)
            
            first_slice_id, end_slice_id = get_start_and_end_slice(lobe_info_path)
            self.series_first_and_end_valid_slice.append([first_slice_id, end_slice_id])
            background_slice_ids = np.ones((end_slice_id - first_slice_id + 1), dtype = np.uint8)
            for min_z, max_z in zip(min_zs, max_zs):
                background_slice_ids[min_z - first_slice_id: max_z - first_slice_id + 1] = 0
            
            background_slice_ids[:self.stride] = 0
            background_slice_ids[-self.stride:] = 0
            for i in range(self.stride, len(background_slice_ids) - self.stride):
                background_slice_ids[i] = patch_is_background(background_slice_ids, i, self.depth)
            background_slice_ids = np.where(background_slice_ids == 1)[0] + first_slice_id
            self.series_nodule_background_slice_ids.append(background_slice_ids)
        
        area_range = np.array([0, 52, 176, 418, 1000])
        
        new_range_areas = []
        for a in series_nodule_sizes:
            for i in range(len(area_range) - 1):
                if a >= area_range[i] and a <= area_range[i + 1]:
                    out = i + (a - area_range[i]) / (area_range[i + 1] - area_range[i])
                    break
                elif (a >= area_range[i + 1] and i == len(area_range) - 2):
                    out = i + 1
                    break
            new_range_areas.append(out)
        new_range_areas = np.array(new_range_areas)
        
        new_range_areas = z_norm(new_range_areas)
        nodule_feats = np.stack([new_range_areas], axis = 1)
        
        n_clusters = 10
        kmeans = KMeans(n_clusters = n_clusters, random_state = 1029, n_init = 'auto')
        kmeans.fit(nodule_feats)
        cluster_labels = kmeans.labels_
        
        cluster_samples = np.bincount(cluster_labels, minlength = n_clusters)
        print(cluster_samples)
        
        type_of_nodule = dict()
        for nodule_size, label in zip(series_nodule_sizes, cluster_labels):
            nodule_type = get_nodule_type(nodule_size, self.nodule_size_ranges)
            if type_of_nodule.get(label, None) == None:
                type_of_nodule[label] = dict()
            if type_of_nodule[label].get(nodule_type, None) == None:
                type_of_nodule[label][nodule_type] = 0
            type_of_nodule[label][nodule_type] += 1
        print(type_of_nodule)
        cluster_labels = kmeans.labels_
        self.cluster_nodules = defaultdict(list)
        for nodule_idx, cluster_idx in enumerate(cluster_labels):
            self.cluster_nodules[cluster_idx].append(int(nodule_idx))
        num_nodule = len(nodule_feats)
        self.max_num_of_type = int(num_nodule / n_clusters)
        self.offsets = list(range(-8, 8 + 1))
        
        self.shuffle_data()

    def shuffle_data(self):
        self.data_list = set()
        for nodule_indices in self.cluster_nodules.values():
            num_of_data = len(nodule_indices)
            if num_of_data >= self.max_num_of_type:
                nodule_indices = random.sample(nodule_indices, self.max_num_of_type)
                for nodule_idx in nodule_indices:
                    series_idx, min_z, max_z = self.nodules_keys[nodule_idx]
                    series_path = self.series_paths[series_idx]
                    gt_mask_maps_path = self.gt_mask_maps_paths[series_idx]
                    first_slice_id, last_slice_id = self.series_first_and_end_valid_slice[series_idx]
                    
                    thickness = max_z - min_z
                    offset_of_nodule = max(thickness // 3, 1)
                    slice_id_lower_bound = max(max_z - offset_of_nodule - self.depth, first_slice_id)
                    slice_id_upper_bound = min_z + offset_of_nodule
                    
                    start_slice_id = random.choice(range(slice_id_lower_bound, slice_id_upper_bound))
                    self.data_list.add((series_path, gt_mask_maps_path, start_slice_id, 0))
            else:
                cur_type_data_list = set()
                for aug_intensity in range(4):
                    random.shuffle(nodule_indices)
                    for nodule_idx in nodule_indices:
                        series_idx, min_z, max_z = self.nodules_keys[nodule_idx]
                        series_path = self.series_paths[series_idx]
                        gt_mask_maps_path = self.gt_mask_maps_paths[series_idx]
                        first_slice_id, last_slice_id = self.series_first_and_end_valid_slice[series_idx]
                        
                        thickness = max_z - min_z
                        offset_of_nodule = max(thickness // 3, 1)
                        slice_id_lower_bound = max(max_z - offset_of_nodule - self.depth, first_slice_id)
                        slice_id_upper_bound = min_z + offset_of_nodule
                        
                        start_slice_id = random.choice(range(slice_id_lower_bound, slice_id_upper_bound))
                        cur_type_data_list.add((series_path, gt_mask_maps_path, start_slice_id, aug_intensity))
                        if len(cur_type_data_list) >= self.max_num_of_type:
                            break
                    if len(cur_type_data_list) >= self.max_num_of_type:
                        break
                self.data_list = self.data_list | cur_type_data_list
        # Add background data
        background_data_list = set()
        for series_idx, background_slice_ids in enumerate(self.series_nodule_background_slice_ids):
            series_path = self.series_paths[series_idx]
            gt_mask_maps_path = self.gt_mask_maps_paths[series_idx]
            first_slice_id, last_slice_id = self.series_first_and_end_valid_slice[series_idx]
            if len(background_slice_ids) == 0:
                continue
            start_slice_id = int(np.random.choice(background_slice_ids, 1))
            aug_intensity = int(np.random.choice(range(4), 1, p = [0.5, 0.3, 0.1, 0.1]))
            
            background_data_list.add((series_path, gt_mask_maps_path, start_slice_id, aug_intensity))
        
        self.data_list = list(self.data_list)
        background_data_list = list(background_data_list)
        max_num_of_background = int(len(self.data_list) * 0.2)
        if len(background_data_list) > max_num_of_background:
            random.shuffle(background_data_list)
            background_data_list = background_data_list[:max_num_of_background]
            
        self.data_list = self.data_list + background_data_list
        random.shuffle(self.data_list)

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

    def augmentation(self, images: List[np.ndarray], aug_intensity: int) -> List[np.ndarray]:
        is_masks = [False, True]
        
        if aug_intensity == 0:
            images = RandomFlipYXZ(p = 0.3)(images)

        elif aug_intensity == 1:
            images = RandomFlipYXZ(p = 0.5)(images)
            images = RandomRotation(angle_range = [-5, 5], p = 0.5)(images, is_masks)
            images[0] = random_color(images[0], p = 0.5, intensity = 0.5)
            
        elif aug_intensity == 2:
            images = RandomFlipYXZ(p = 0.7)(images)
            images = RandomRotation(angle_range = [-5, 5], p = 0.7)(images, is_masks)
            images[0] = random_color(images[0], p = 0.7, intensity = 1.0)
            
        elif aug_intensity == 3:
            images = RandomFlipYXZ(p = 0.8)(images)
            images = RandomRotation90(p = 0.5)(images)
            images = RandomRotation(angle_range = [-5, 5], p = 1.0)(images, is_masks)
            images[0] = random_color(images[0], p = 0.8, intensity = 1.0)
            
        return images

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path, mask_path, start_slice_id, aug_intensity = self.data_list[index]
        image = self.load_image(image_path, start_slice_id)
        target512 = self.load_gt_mask(mask_path, start_slice_id)
        
        if self.dataset_type == 'train':
            image, target512 = self.augmentation([image, target512], aug_intensity)

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