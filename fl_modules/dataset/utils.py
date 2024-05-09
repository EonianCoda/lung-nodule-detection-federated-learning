import os
import json
import numpy as np
from typing import Dict, List

DEFAULT_WINDOW_LEVEL = -300
DEFAULT_WINDOW_WIDTH = 1400
# HU MIN is -1000, HU MAX is 400
# Water value is 0.0 in HU, so the normalized value is (0 - 1000) / (400 - (-1000)) = 0.714
DEFAULT_WATER_RATIO = 0.714

BBOXES = 'bboxes'
NODULE_SIZE = 'nodule_size'
ALL_LOC = 'all_loc'
ALL_RAD = 'all_rad'
ALL_CLS = 'all_cls'
ALL_PROB = 'all_prob'
ALL_IOU = 'all_iou'

ALL_HARD_FP_LOC = 'all_hard_fp_loc'
ALL_HARD_FP_RAD = 'all_hard_fp_rad'
ALL_HARD_FP_PROB = 'all_hard_fp_prob'

TRUE_NODULE_TYPE = 'true_nodule_type' # For patch json

def gen_dicom_path(folder: str, series_name: str) -> str:
    return os.path.join(folder, 'npy', f'{series_name}_crop.npy')

def gen_label_path(dir_name: str, name: str) -> str:
    return os.path.join(dir_name, 'mask', f'{name}_nodule_count_crop.json')

def gen_lobe_path(dir_name: str, name: str) -> str:
    return os.path.join(dir_name, 'npy', f'{name}_lobe_crop.npz')

def gen_patch_label_path(dir_name: str, name: str) -> str:
    return os.path.join(dir_name, 'mask', f'{name}_nodule_count_patch_crop.json')

def load_series_list(series_list_path: str) -> List[List[str]]:
    """
    Return:
        series_list: list of tuples (series_folder, file_name)

    """
    with open(series_list_path, 'r') as f:
        lines = f.readlines()
        lines = lines[1:] # Remove the line of description
        
    series_list = []
    for series_info in lines:
        series_info = series_info.strip()
        series_folder, file_name = series_info.split(',')
        series_list.append([series_folder, file_name])
    return series_list

def load_lobe(lobe_path: str) -> np.ndarray:
    lobe = np.load(lobe_path)['image']
    lobe = np.transpose(lobe, (2, 0, 1))
    return lobe

def get_HU_MIN_MAX(window_level: int, window_width: int):
    hu_min = window_level - window_width // 2
    hu_max = window_level + window_width // 2
    return hu_min, hu_max

def normalize_raw_image(image: np.ndarray, window_level: int = DEFAULT_WINDOW_LEVEL, window_width: int = DEFAULT_WINDOW_WIDTH) -> np.ndarray:
    hu_min, hu_max = get_HU_MIN_MAX(window_level, window_width)    
    image = np.clip(image, hu_min, hu_max)
    image = image - hu_min
    image = image.astype(np.float32) / (hu_max - hu_min)
    return image

def normalize_processed_image(image: np.ndarray, method: str = 'scale') -> np.ndarray:
    """
    Args:
        image: 3D numpy array with dimension order [D, H, W] (z, y, x) whose value is ranged from 0 to 1
        method: 'mean_std' or 'scale' or 'none'
    Return:
        Normalized image
        1) 'mean_std': the image is normalized to have mean 0 and std 1
        2)  'scale': the image is normalized to have value ranged from -1 to 1
        3) 'none': the image is not normalized
    """
    if method == 'mean_std':
        image = (image - np.mean(image)) / np.std(image)
        # min max scale to -1 to 1
        min_val, max_val = np.min(image), np.max(image)
        image = (image - min_val) / (max_val - min_val) * 2.0 - 1.0
    elif method == 'scale':
        image = image * 2.0 - 1.0
    elif method == 'none':
        pass
    return image

def get_image_padding_value(method: str = 'scale', use_water: bool = True) -> float:
    if method == 'mean_std':
        return -1.0 + DEFAULT_WATER_RATIO * 2.0 if use_water else -1.0
    elif method == 'scale':
        return -1.0 + DEFAULT_WATER_RATIO * 2.0 if use_water else -1.0
    elif method == 'none':
        return 0.0 + DEFAULT_WATER_RATIO if use_water else 0.0

def load_image(dicom_path: str, window_level: int = DEFAULT_WINDOW_LEVEL, window_width: int = DEFAULT_WINDOW_WIDTH) -> np.ndarray:
    """
    Return:
        A 3D numpy array with dimension order [D, H, W] (z, y, x)
    """
    image = np.load(dicom_path)
    image = np.transpose(image, (2, 0, 1))
    image = normalize_raw_image(image, window_level, window_width)
    return image

def load_label(label_path: str, image_spacing: np.ndarray, min_d = 0, min_size = 0) -> Dict[str, np.ndarray]:
    """
    Return:
        A dictionary with keys 'all_loc', 'all_rad', 'all_cls'
        (1) 'all_loc': 3D numpy array with shape (n, 3) (z, y, x)
        (2) 'all_rad': depth, height, width of nodules with shape (n, 3) (z, y, x)
        (3) 'all_cls': 1D numpy array with shape (n,)
    """
    min_d = int(min_d)
    with open(label_path, 'r') as f:
        info = json.load(f)
    
    bboxes = np.array(info[BBOXES]) # (n, 2, 3)
    nodule_sizes = np.array(info[NODULE_SIZE]) # (n, 1)
    if len(bboxes) == 0:
        label = {ALL_LOC: np.zeros((0, 3), dtype=np.float64),
                ALL_RAD: np.zeros((0, 3), dtype=np.int32),
                ALL_CLS: np.zeros((0,), dtype=np.float64),
                NODULE_SIZE: np.zeros((0,), dtype=np.int32)}
    else:
        bboxes[:, 0, :] = np.maximum(bboxes[:, 0, :], 0) # clip to 0
        if (bboxes < 0).any():
            print(f'Warning: {label_path} has negative values')
        # calculate center of bboxes
        all_loc = ((bboxes[:, 0] + bboxes[:, 1]) / 2).astype(np.float64) # (y, x, z)
        all_rad = (bboxes[:, 1] - bboxes[:, 0]).astype(np.float64) # (y, x, z)
        
        all_loc = all_loc[:, [2, 0, 1]] # (z, y, x)
        all_rad = all_rad[:, [2, 0, 1]] # (z, y, x)
        
        valid_mask = all_rad[:, 0] >= min_d
        if min_size > 0:
            valid_mask = valid_mask & (nodule_sizes >= min_size)
        
        all_rad = all_rad * image_spacing # (z, y, x)
        all_cls = np.zeros((all_loc.shape[0],), dtype=np.int32)

        if np.sum(valid_mask) == 0:
            label = {ALL_LOC: np.zeros((0, 3), dtype=np.float64),
                    ALL_RAD: np.zeros((0, 3), dtype=np.int32),
                    ALL_CLS: np.zeros((0, ), dtype=np.float64),
                    NODULE_SIZE: np.zeros((0,), dtype=np.int32)}
        else:
            all_loc = all_loc[valid_mask]
            all_rad = all_rad[valid_mask]
            all_cls = all_cls[valid_mask]
            nodule_sizes = nodule_sizes[valid_mask]
            label = {ALL_LOC: all_loc, 
                    ALL_RAD: all_rad,
                    ALL_CLS: all_cls,
                    NODULE_SIZE: nodule_sizes}
    return label

def load_patch_label(label_path: str, image_spacing: np.ndarray, min_d = 0, min_size = 0) -> Dict[str, np.ndarray]:
    """ Load patch label from json file
    Patch label is a json file which contains some missing nodule compared to the original label file
    Args:
        label_path: str, path to the patch label file
        image_spacing: np.ndarray, spacing of the image in z, y, x
        min_d: int, minimum depth of nodule
        min_size: int, minimum size of nodule
    Return:
        A dictionary with keys 'all_loc', 'all_rad', 'all_cls'
        (1) 'all_loc': 3D numpy array with shape (n, 3) (z, y, x)
        (2) 'all_rad': depth, height, width of nodules with shape (n, 3) (z, y, x)
        (3) 'all_cls': 1D numpy array with shape (n,)
    """
    min_d = int(min_d)
    with open(label_path, 'r') as f:
        info = json.load(f)
    
    bboxes = np.array(info[BBOXES]) # (n, 2, 3)
    nodule_sizes = np.array(info[NODULE_SIZE]) # (n, 1)
    true_nodule_type = info[TRUE_NODULE_TYPE]
    
    nodule_cls = []
    for nodule_type in true_nodule_type:
        if nodule_type == 'tp':
            nodule_cls.append(0)
        elif nodule_type == 'benign':
            nodule_cls.append(1)
        elif nodule_type == 'confuse':
            nodule_cls.append(2)
            
    if len(bboxes) == 0:
        label = {ALL_LOC: np.zeros((0, 3), dtype=np.float64),
                ALL_CLS: np.zeros((0, 3), dtype=np.float64),
                ALL_RAD: np.zeros((0,), dtype=np.int32),
                NODULE_SIZE: np.zeros((0,), dtype=np.int32)}
    else:
        bboxes[:, 0, :] = np.maximum(bboxes[:, 0, :], 0) # clip to 0
        if (bboxes < 0).any():
            print(f'Warning: {label_path} has negative values')
        # calculate center of bboxes
        all_loc = ((bboxes[:, 0] + bboxes[:, 1]) / 2).astype(np.float64) # (y, x, z)
        all_rad = (bboxes[:, 1] - bboxes[:, 0]).astype(np.float64) # (y, x, z)
        
        all_loc = all_loc[:, [2, 0, 1]] # (z, y, x)
        all_rad = all_rad[:, [2, 0, 1]] # (z, y, x)
        
        valid_mask = all_rad[:, 0] >= min_d
        if min_size > 0:
            valid_mask = valid_mask & (nodule_sizes >= min_size)
        
        all_rad = all_rad * image_spacing # (z, y, x)
        all_cls = np.zeros((all_loc.shape[0],), dtype=np.int32)

        if np.sum(valid_mask) == 0:
            label = {ALL_LOC: np.zeros((0, 3), dtype=np.float64),
                    ALL_CLS: np.zeros((0, 3), dtype=np.float64),
                    ALL_RAD: np.zeros((0,), dtype=np.int32),
                    NODULE_SIZE: np.zeros((0,), dtype=np.int32)}
        else:
            all_loc = all_loc[valid_mask]
            all_rad = all_rad[valid_mask]
            all_cls = all_cls[valid_mask]
            
            nodule_cls = np.array(nodule_cls, dtype=np.int32)[valid_mask]
            
            nodule_sizes = nodule_sizes[valid_mask]
            label = {ALL_LOC: all_loc, 
                    ALL_RAD: all_rad,
                    ALL_CLS: nodule_cls,
                    NODULE_SIZE: nodule_sizes}
    return label

def compute_bbox3d_intersection_volume(box1: np.ndarray, box2: np.ndarray):
    """ 
    Args:
        box1 (shape = [N, 2, 3])
        box2 (shape = [M, 2, 3])
    Return:
        the area of the intersection between box1 and box2, shape = [N, M]
    """
    a1, a2 = box1[:,np.newaxis, 0,:], box1[:,np.newaxis, 1,:] # [N, 1, 3]
    b1, b2 = box2[np.newaxis,:, 0,:], box2[np.newaxis,:, 1,:] # [1, N, 3]
    inter_volume = np.clip((np.minimum(a2, b2) - np.maximum(a1, b1)),0, None).prod(axis=2)

    return inter_volume

def compute_bbox3d_iou(box1: np.ndarray, box2: np.ndarray):
    """ 
    Args:
        box1 (shape = [N, 2, 3])
        box2 (shape = [M, 2, 3])
    Return:
        the iou between box1 and box2, shape = [N, M]
    """
    inter_volume = compute_bbox3d_intersection_volume(box1, box2)
    box1_volume = np.prod(box1[:, 1] - box1[:, 0], axis=1)
    box2_volume = np.prod(box2[:, 1] - box2[:, 0], axis=1)
    union_volume = box1_volume[:, np.newaxis] + box2_volume[np.newaxis, :] - inter_volume
    iou = inter_volume / union_volume
    return iou