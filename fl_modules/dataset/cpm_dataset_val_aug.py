# -*- coding: utf-8 -*-
from __future__ import print_function, division
import logging
import numpy as np
from typing import List, Tuple
from .utils import load_series_list, load_image, load_label, load_lobe, ALL_RAD, ALL_LOC, ALL_CLS, gen_dicom_path, gen_label_path, \
                    gen_lobe_path, normalize_processed_image, normalize_raw_image
from torch.utils.data import Dataset
import torchvision
import copy
import math
logger = logging.getLogger(__name__)

from transform.ctr_transform import OffsetMinusCTR, RotateCTR, TransposeCTR
from transform.feat_transform import FlipFeatTransform, Rot90FeatTransform, TransposeFeatTransform

class FlipTransform():
    def __init__(self, flip_depth=True, flip_height=True, flip_width=True):
        """
            flip_depth (bool) : random flip along depth axis or not, only used for 3D images
            flip_height (bool): random flip along height axis or not
            flip_width (bool) : random flip along width axis or not
        """
        self.flip_depth = flip_depth
        self.flip_height = flip_height
        self.flip_width = flip_width

    def __call__(self, sample):
        image = sample['image']
        input_shape = image.shape
        flip_axes = []
        
        if self.flip_width:
            flip_axes.append(-1)
        if self.flip_height:
            flip_axes.append(-2)
        if self.flip_depth:
            flip_axes.append(-3)

        if len(flip_axes) > 0:
            # use .copy() to avoid negative strides of numpy array
            # current pytorch does not support negative strides
            image_t = np.flip(image, flip_axes).copy()
            sample['image'] = image_t

            offset = np.array([0, 0, 0]) # (z, y, x)
            for axis in flip_axes:
                offset[axis] = input_shape[axis] - 1
            sample['ctr_transform'].append(OffsetMinusCTR(offset))
            sample['feat_transform'].append(FlipFeatTransform(flip_axes))
        return sample

class Rotate90():
    def __init__(self, rot_xy: bool = True, rot_xz: bool = False, rot_yz: bool = False):
        self.rot_xy = rot_xy
        self.rot_xz = rot_xz
        self.rot_yz = rot_yz
    
    def __call__(self, sample):
        image = sample['image']
        image_shape = image.shape[1:] # remove channel dimension
        
        all_rot_axes = []
        rot_angles = []
        if self.rot_xy:
            all_rot_axes.append((-1, -2))
            rot_angles.append(90)
        
        if self.rot_xz:
            all_rot_axes.append((-1, -3))
            rot_angles.append(90)
        
        if self.rot_yz:
            all_rot_axes.append((-2, -3))
            rot_angles.append(90)
        
        if len(all_rot_axes) > 0:
            rot_image = sample['image']
            for rot_axes, rot_angle in zip(all_rot_axes, rot_angles):
                rot_image = self.rotate_3d_image(rot_image, rot_axes, rot_angle)
                sample['ctr_transform'].append(RotateCTR(rot_angle, rot_axes, image_shape))
                sample['feat_transform'].append(Rot90FeatTransform(rot_angle, rot_axes))
            sample['image'] = rot_image
        return sample
    
    @staticmethod
    def rotate_3d_image(data: np.ndarray, rot_axes: Tuple[int], rot_angle: int):
        """
        Args:
            data: 3D image data with shape (D, H, W).
            rot_axes: rotation axes.
            rot_angle: rotation angle. One of 90, 180, or 270.
        """
        rotated_data = data.copy()
        rotated_data = np.rot90(rotated_data, k=rot_angle // 90, axes=rot_axes)
        return rotated_data

    @staticmethod
    def rotate_3d_bbox(ctrs: np.ndarray, bbox_shapes: np.ndarray, image_spacing: np.ndarray, image_shape: np.ndarray, rot_axes: Tuple[int], angle: int):
        """
        Args:
            ctrs: 3D bounding box centers with shape (N, 3).
            bbox_shapes: 3D bounding box shapes with shape (N, 3).
            image_shape: 3D image shape with shape (3,).
            angle: rotation angle. One of 90, 180, or 270.
            plane: rotation plane. One of 'xy', 'xz', or 'yz'.
        """
        new_ctr_zyx = ctrs.copy()
        new_shape_dhw = bbox_shapes.copy()
        new_image_spacing = image_spacing.copy()
        
        if len(ctrs) != 0:
            radian = math.radians(angle)
            cos = np.cos(radian)
            sin = np.sin(radian)
            img_center = np.array(image_shape) / 2
            new_ctr_zyx = ctrs.copy()
            new_ctr_zyx[:, rot_axes[0]] = (ctrs[:, rot_axes[0]] - img_center[rot_axes[0]]) * cos - (ctrs[:, rot_axes[1]] - img_center[rot_axes[1]]) * sin + img_center[rot_axes[0]]
            new_ctr_zyx[:, rot_axes[1]] = (ctrs[:, rot_axes[0]] - img_center[rot_axes[0]]) * sin + (ctrs[:, rot_axes[1]] - img_center[rot_axes[1]]) * cos + img_center[rot_axes[1]]
        
        if angle == 90 or angle == 270:
            if len(bbox_shapes) != 0:
                new_shape_dhw[:, rot_axes[0]] = bbox_shapes[:, rot_axes[1]] 
                new_shape_dhw[:, rot_axes[1]] = bbox_shapes[:, rot_axes[0]]
            new_image_spacing[rot_axes[0]] = image_spacing[rot_axes[1]]
            new_image_spacing[rot_axes[1]] = image_spacing[rot_axes[0]]
        return new_ctr_zyx, new_shape_dhw, new_image_spacing

class TransPose():
    def __init__(self, trans_xy=True, trans_zx=False, trans_zy=False):
        self.trans_xy = trans_xy
        self.trans_zx = trans_zx
        self.trans_zy = trans_zy

    def __call__(self, sample):
        transpose_list = []

        if self.trans_zy:
            transpose_list.append(np.array([-2, -3]))
        if self.trans_xy:
            transpose_list.append(np.array([-1, -2]))
        if self.trans_zx:
            transpose_list.append(np.array([-1, -3]))

        if len(transpose_list) > 0:
            num_dims = len(sample['image'].shape)
            transpose_order = np.arange(num_dims)
            transpose_order[-3:] = np.array([-3, -2, -1])
            transpose_order = transpose_order.astype(np.int32)
            
            for transpose in transpose_list:
                a = transpose_order[transpose[0]]
                b = transpose_order[transpose[1]]
                transpose_order[transpose[0]] = b
                transpose_order[transpose[1]] = a
            
            sample['image'] = np.transpose(sample['image'], transpose_order)
            # Use last 3 dimensions (D, H, W)
            transpose_order = transpose_order[-3:] 
            sample['ctr_transform'].append(TransposeCTR(transpose_order.copy())) # last 3 dimensions (D, H, W)
            sample['feat_transform'].append(TransposeFeatTransform(transpose_order))
            
            if 'ctr' in sample and len(sample['ctr']) > 0:
                sample['ctr'] = sample['ctr'][:, transpose_order]
                sample['rad'] = sample['rad'][:, transpose_order]
                
        return sample
    
class DetDataset(Dataset):
    """Detection dataset for inference
    """
    def __init__(self, series_list_path: str, image_spacing: List[float], SplitComb, norm_method='scale', apply_lobe=False, out_stride = 4):
        self.series_list_path = series_list_path
        self.apply_lobe = apply_lobe
        self.norm_method = norm_method
        self.out_stride = out_stride
        
        self.image_spacing = np.array(image_spacing, dtype=np.float32) # (z, y, x)
        self.series_infos = load_series_list(series_list_path)
        
        self.dicom_paths = []
        self.lobe_paths = []
        for folder, series_name in self.series_infos:
            self.dicom_paths.append(gen_dicom_path(folder, series_name))
            self.lobe_paths.append(gen_lobe_path(folder, series_name))
        self.splitcomb = SplitComb
        transforms = [[FlipTransform(flip_depth=False, flip_height=False, flip_width=True)],
                    [FlipTransform(flip_depth=False, flip_height=True, flip_width=False)],
                    [FlipTransform(flip_depth=True, flip_height=False, flip_width=False)]]
        
        self.transforms_weight = [0.5] + [0.5 / len(transforms)] * len(transforms) # first one is for no augmentation
        self.transforms_weight = np.array([w / sum(self.transforms_weight) for w in self.transforms_weight]) # normalize to 1
        self.transforms = []
        for i in range(len(transforms)):
            self.transforms.append(torchvision.transforms.Compose(transforms[i]))
        
    def __len__(self):
        return len(self.dicom_paths)
    
    def __getitem__(self, idx):
        dicom_path = self.dicom_paths[idx]
        series_folder = self.series_infos[idx][0]
        series_name = self.series_infos[idx][1]
        
        image_spacing = self.image_spacing.copy() # z, y, x
        image = load_image(dicom_path) # z, y, x
        image = normalize_processed_image(image, self.norm_method)

        # split_images [N, 1, crop_z, crop_y, crop_x]
        if self.apply_lobe: # load lobe mask
            lobe_path = self.lobe_paths[idx]
            lobe_mask = load_lobe(lobe_path)
            split_images, split_lobes, nzhw, image_shape = self.splitcomb.split(image, lobe_mask, self.out_stride) # split_images [N, 1, crop_z, crop_y, crop_x]
        else:
            split_images, nzhw, image_shape = self.splitcomb.split(image) # split_images [N, 1, crop_z, crop_y, crop_x]
        
        split_images = np.squeeze(split_images, axis=1) # (N, crop_z, crop_y, crop_x)
        
        sample = {'image': split_images.copy(), 'ctr_transform': [], 'feat_transform': []}
        all_samples = [sample]
        for i, transform_fn in enumerate(self.transforms):
            all_samples.append(transform_fn(copy.deepcopy(sample)))
        
        # Stack the split images
        split_images = np.stack([s['image'] for s in all_samples], axis=1) # (N, num_aug, crop_z, crop_y, crop_x)
        split_images = np.expand_dims(split_images, axis=2) # (N, num_aug, 1, crop_z, crop_y, crop_x)
        split_images = np.ascontiguousarray(split_images)
        
        ctr_transforms = [s['ctr_transform'] for s in all_samples] # (num_aug,)
        feat_transforms = [s['feat_transform'] for s in all_samples] # (num_aug,)
        
        all_samples = {'split_images': split_images, 
                       'ctr_transform': ctr_transforms, 
                       'feat_transform': feat_transforms,
                       'transform_weight': self.transforms_weight.copy()}
        all_samples['nzhw'] = nzhw
        all_samples['spacing'] = image_spacing
        all_samples['series_name'] = series_name
        all_samples['series_folder'] = series_folder
        all_samples['image_shape'] = image_shape
        
        # Add lobe mask
        if self.apply_lobe:
            all_samples['split_lobes'] = np.ascontiguousarray(split_lobes)
        
        return all_samples