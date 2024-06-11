# -*- coding: utf-8 -*-
from __future__ import print_function, division
import random
import numpy as np
from .abstract_transform import AbstractTransform
from .ctr_transform import OffsetMinusCTR
from .feat_transform import FlipFeatTransform

class RandomFlip(AbstractTransform):
    """ random flip the image (shape [C, D, H, W] or [C, H, W]) """

    def __init__(self, flip_depth=True, flip_height=True, flip_width=True,p=0.5):
        """
            flip_depth (bool) : random flip along depth axis or not, only used for 3D images
            flip_height (bool): random flip along height axis or not
            flip_width (bool) : random flip along width axis or not
        """
        self.flip_depth = flip_depth
        self.flip_height = flip_height
        self.flip_width = flip_width
        self.p = p

    def __call__(self, sample):
        image = sample['image']
        input_shape = image.shape
        input_dim = len(input_shape) - 1
        flip_axes = []
        if self.flip_width:
            if random.random() < self.p:
                flip_axes.append(-1)
        if self.flip_height:
            if random.random() < self.p:
                flip_axes.append(-2)
        if input_dim == 3 and self.flip_depth:
            if random.random() < self.p:
                flip_axes.append(-3)

        if len(flip_axes) > 0:
            # use .copy() to avoid negative strides of numpy array
            # current pytorch does not support negative strides
            image_t = np.flip(image, flip_axes).copy()
            sample['image'] = image_t

            ctr = sample['ctr'].copy()
            offset = np.array([0, 0, 0]) # (z, y, x)
            for axis in flip_axes:
                ctr[:, axis] = input_shape[axis] - 1 - ctr[:, axis]
                offset[axis] = input_shape[axis] - 1
            sample['ctr'] = ctr
            sample['ctr_transform'].append(OffsetMinusCTR(offset))
            sample['feat_transform'].append(FlipFeatTransform(flip_axes))
        return sample
    
class SemiRandomFlip(AbstractTransform):
    """ random flip the image (shape [C, D, H, W] or [C, H, W]) """

    def __init__(self, flip_depth=True, flip_height=True, flip_width=True,p=0.5):
        """
            flip_depth (bool) : random flip along depth axis or not, only used for 3D images
            flip_height (bool): random flip along height axis or not
            flip_width (bool) : random flip along width axis or not
        """
        self.flip_depth = flip_depth
        self.flip_height = flip_height
        self.flip_width = flip_width
        self.p = p

    def __call__(self, sample):
        image = sample['image']
        input_shape = image.shape
        input_dim = len(input_shape) - 1
        flip_axes = []
        if self.flip_width:
            if random.random() < self.p:
                flip_axes.append(-1)
        if self.flip_height:
            if random.random() < self.p:
                flip_axes.append(-2)
        if input_dim == 3 and self.flip_depth:
            if random.random() < self.p:
                flip_axes.append(-3)

        if len(flip_axes) > 0:
            # use .copy() to avoid negative strides of numpy array
            # current pytorch does not support negative strides
            image_t = np.flip(image, flip_axes).copy()
            sample['image'] = image_t

            ctr = sample['ctr'].copy()
            if 'gt_ctr' in sample:
                gt_ctr = sample['gt_ctr'].copy()
            offset = np.array([0, 0, 0]) # (z, y, x)
            for axis in flip_axes:
                ctr[:, axis] = input_shape[axis] - 1 - ctr[:, axis]
                if 'gt_ctr' in sample:
                    gt_ctr[:, axis] = input_shape[axis] - 1 - gt_ctr[:, axis]
                offset[axis] = input_shape[axis] - 1
                
            sample['ctr'] = ctr
            if 'gt_ctr' in sample:
                sample['gt_ctr'] = gt_ctr
            sample['ctr_transform'].append(OffsetMinusCTR(offset))
            sample['feat_transform'].append(FlipFeatTransform(flip_axes))
        return sample
    

class RandomMaskFlip(AbstractTransform):
    """ random flip the image (shape [C, D, H, W] or [C, H, W]) """

    def __init__(self, flip_depth=True, flip_height=True, flip_width=True,p=0.5):
        """
            flip_depth (bool) : random flip along depth axis or not, only used for 3D images
            flip_height (bool): random flip along height axis or not
            flip_width (bool) : random flip along width axis or not
        """
        self.flip_depth = flip_depth
        self.flip_height = flip_height
        self.flip_width = flip_width
        self.p = p

    def __call__(self, sample):
        image = sample['image']
        input_shape = image.shape
        input_dim = len(input_shape) - 1
        flip_axis = []
        if self.flip_width:
            if random.random() < self.p:
                flip_axis.append(-1)
        if self.flip_height:
            if random.random() < self.p:
                flip_axis.append(-2)
        if input_dim == 3 and self.flip_depth:
            if random.random() < self.p:
                flip_axis.append(-3)

        if len(flip_axis) > 0:
            # use .copy() to avoid negative strides of numpy array
            # current pytorch does not support negative strides
            image_t = np.flip(image, flip_axis).copy()
            mask = sample['mask']
            mask_t = np.flip(mask, flip_axis).copy()
            sample['image'] = image_t
            sample['mask'] = mask_t
        
        return sample
