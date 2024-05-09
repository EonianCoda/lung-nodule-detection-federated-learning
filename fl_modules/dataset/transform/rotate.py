# -*- coding: utf-8 -*-
from __future__ import print_function, division

import json
from .abstract_transform import AbstractTransform
from .ctr_transform import TransposeCTR, RotateCTR
from .feat_transform import Rot90FeatTransform, TransposeFeatTransform
import random
import math
import numpy as np
from typing import Tuple


class RandomRotate90(AbstractTransform):
    def __init__(self, p=0.5, rot_xy: bool = True, rot_xz: bool = False, rot_yz: bool = False):
        self.p = p
        self.rot_xy = rot_xy
        self.rot_xz = rot_xz
        self.rot_yz = rot_yz
        self.rot_angles = np.array([90, 180, 270], dtype=np.int32)
    
    def __call__(self, sample):
        image = sample['image']
        image_shape = image.shape[1:] # remove channel dimension
        
        all_rot_axes = []
        rot_angles = []
        if random.random() < self.p and self.rot_xy:
            all_rot_axes.append((-1, -2))
            rot_angles.append(np.random.choice(self.rot_angles, 1)[0])
        
        if random.random() < self.p and self.rot_xz:
            all_rot_axes.append((-1, -3))
            rot_angles.append(np.random.choice(self.rot_angles, 1)[0])
        
        if random.random() < self.p and self.rot_yz:
            all_rot_axes.append((-2, -3))
            rot_angles.append(np.random.choice(self.rot_angles, 1)[0])
        
        if len(all_rot_axes) > 0:
            rot_image = sample['image']
            rot_ctr = sample['ctr']
            rot_rad = sample['rad']
            rot_spacing = sample['spacing']
            for rot_axes, rot_angle in zip(all_rot_axes, rot_angles):
                rot_image = self.rotate_3d_image(rot_image, rot_axes, rot_angle)
                rot_ctr, rot_rad, rot_spacing = self.rotate_3d_bbox(rot_ctr, rot_rad, rot_spacing, image_shape, rot_axes, rot_angle)
                sample['ctr_transform'].append(RotateCTR(rot_angle, rot_axes, image_shape))
                sample['feat_transform'].append(Rot90FeatTransform(rot_angle, rot_axes))
            sample['image'] = rot_image
            sample['ctr'] = rot_ctr
            sample['rad'] = rot_rad
            sample['spacing'] = rot_spacing
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
    
class SemiRandomRotate90(AbstractTransform):
    def __init__(self, p=0.5, rot_xy: bool = True, rot_xz: bool = False, rot_yz: bool = False):
        self.p = p
        self.rot_xy = rot_xy
        self.rot_xz = rot_xz
        self.rot_yz = rot_yz
        self.rot_angles = np.array([90, 180, 270], dtype=np.int32)
    
    def __call__(self, sample):
        image = sample['image']
        image_shape = image.shape[1:] # remove channel dimension
        
        all_rot_axes = []
        rot_angles = []
        if random.random() < self.p and self.rot_xy:
            all_rot_axes.append((-1, -2))
            rot_angles.append(np.random.choice(self.rot_angles, 1)[0])
        
        if random.random() < self.p and self.rot_xz:
            all_rot_axes.append((-1, -3))
            rot_angles.append(np.random.choice(self.rot_angles, 1)[0])
        
        if random.random() < self.p and self.rot_yz:
            all_rot_axes.append((-2, -3))
            rot_angles.append(np.random.choice(self.rot_angles, 1)[0])
        
        if len(all_rot_axes) > 0:
            rot_image = sample['image']
            rot_ctr = sample['ctr']
            rot_rad = sample['rad']
            
            rot_gt_ctr = sample['gt_ctr']
            rot_gt_rad = sample['gt_rad']
            
            rot_spacing = sample['spacing']
            for rot_axes, rot_angle in zip(all_rot_axes, rot_angles):
                rot_image = self.rotate_3d_image(rot_image, rot_axes, rot_angle)
                rot_ctr, rot_rad, rot_gt_ctr, rot_gt_rad, rot_spacing = self.rotate_3d_bbox(rot_ctr, rot_rad, rot_gt_ctr, rot_gt_rad, rot_spacing, image_shape, rot_axes, rot_angle)
                sample['ctr_transform'].append(RotateCTR(rot_angle, rot_axes, image_shape))
                sample['feat_transform'].append(Rot90FeatTransform(rot_angle, rot_axes))
            sample['image'] = rot_image
            sample['ctr'] = rot_ctr
            sample['rad'] = rot_rad
            
            sample['gt_ctr'] = rot_gt_ctr
            sample['gt_rad'] = rot_gt_rad
            
            sample['spacing'] = rot_spacing
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
    def rotate_3d_bbox(ctrs: np.ndarray, bbox_shapes: np.ndarray, 
                       gt_ctrs: np.ndarray, gt_bbox_shapes: np.ndarray,
                       image_spacing: np.ndarray, image_shape: np.ndarray, rot_axes: Tuple[int], angle: int):
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
        
        new_gt_ctr_zyx = gt_ctrs.copy()
        new_gt_shape_dhw = gt_bbox_shapes.copy()
        
        radian = math.radians(angle)
        cos = np.cos(radian)
        sin = np.sin(radian)
        img_center = np.array(image_shape) / 2
        if len(ctrs) != 0:
            new_ctr_zyx[:, rot_axes[0]] = (ctrs[:, rot_axes[0]] - img_center[rot_axes[0]]) * cos - (ctrs[:, rot_axes[1]] - img_center[rot_axes[1]]) * sin + img_center[rot_axes[0]]
            new_ctr_zyx[:, rot_axes[1]] = (ctrs[:, rot_axes[0]] - img_center[rot_axes[0]]) * sin + (ctrs[:, rot_axes[1]] - img_center[rot_axes[1]]) * cos + img_center[rot_axes[1]]

        if len(gt_ctrs) != 0:
            new_gt_ctr_zyx[:, rot_axes[0]] = (gt_ctrs[:, rot_axes[0]] - img_center[rot_axes[0]]) * cos - (gt_ctrs[:, rot_axes[1]] - img_center[rot_axes[1]]) * sin + img_center[rot_axes[0]]
            new_gt_ctr_zyx[:, rot_axes[1]] = (gt_ctrs[:, rot_axes[0]] - img_center[rot_axes[0]]) * sin + (gt_ctrs[:, rot_axes[1]] - img_center[rot_axes[1]]) * cos + img_center[rot_axes[1]]
            
        if angle == 90 or angle == 270:
            if len(bbox_shapes) != 0:
                new_shape_dhw[:, rot_axes[0]] = bbox_shapes[:, rot_axes[1]] 
                new_shape_dhw[:, rot_axes[1]] = bbox_shapes[:, rot_axes[0]]
                
            if len(gt_bbox_shapes) != 0:
                new_gt_shape_dhw[:, rot_axes[0]] = gt_bbox_shapes[:, rot_axes[1]] 
                new_gt_shape_dhw[:, rot_axes[1]] = gt_bbox_shapes[:, rot_axes[0]]
                
            # Swap the spacing
            new_image_spacing[rot_axes[0]] = image_spacing[rot_axes[1]]
            new_image_spacing[rot_axes[1]] = image_spacing[rot_axes[0]]
        return new_ctr_zyx, new_shape_dhw, new_gt_ctr_zyx, new_gt_shape_dhw, new_image_spacing