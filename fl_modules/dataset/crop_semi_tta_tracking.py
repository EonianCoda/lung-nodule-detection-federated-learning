# -*- coding: utf-8 -*-
from __future__ import print_function, division
import numpy as np
import random
import scipy
from itertools import product
from .utils import compute_bbox3d_intersection_volume

class InstanceCrop(object):
    """Randomly crop the input image (shape [C, D, H, W])

    Args:
        crop_size (list[int]): The size of the patch to be cropped.
        rand_trans (list[int], optional): The range of random translation. Defaults to None.
        rand_rot (list[int], optional): The range of random rotation. Defaults to None.
        instance_crop (bool, optional): Whether to perform additional sampling with instance around the center. Defaults to True.
        overlap_size (list[int], optional): The size of overlap of sliding window. Defaults to [16, 32, 32].
        tp_ratio (float, optional): The sampling rate for a patch containing at least one lesion. Defaults to 0.7.
        sample_num (int, optional): The number of patches per CT. Defaults to 2.
        blank_side (int, optional): The number of pixels near the patch border where labels are set to ignored. Defaults to 0.
        sample_cls (list[int], optional): The list of classes to sample patches from. Defaults to [0].
    """

    def __init__(self, crop_size, overlap_ratio: float = 0.25, rand_trans=None, rand_rot=None, sample_num=2, blank_side=0, sample_cls=[0], out_stride = 4, tp_ratio=0.75):
        """This is crop function with spatial augmentation for training Lesion Detection.

        Arguments:
            crop_size: patch size
            rand_trans: random translation
            rand_rot: random rotation
            instance_crop: additional sampling with instance around center
            spacing: output patch spacing, [z,y,x]
            base_spacing: spacing of the numpy image.
            overlap_size: the size of overlap  of sliding window
            sample_num: patch number per CT
            blank_side:  labels within blank_side pixels near patch border is set to ignored.
        """
        self.sample_cls = sample_cls
        self.crop_size = np.array(crop_size, dtype=np.int32)
        self.overlap_ratio = overlap_ratio
        self.overlap_size = (self.crop_size * self.overlap_ratio).astype(np.int32)
        self.stride_size = self.crop_size - self.overlap_size
        
        self.sample_num = sample_num
        self.blank_side = blank_side

        self.out_stride = out_stride
        self.lobe_zoom_ratio = [1 / out_stride, 1 / out_stride, 1 / out_stride]
        if rand_trans == None:
            self.rand_trans = None
        else:
            self.rand_trans = np.array(rand_trans)

        if rand_rot == None:
            self.rand_rot = None
        else:
            self.rand_rot = np.array(rand_rot)

        self.tp_ratio = tp_ratio
        
    def get_crop_centers(self, shape, dim: int):
        crop = self.crop_size[dim]
        overlap = self.overlap_size[dim]
        stride = self.stride_size[dim]
        shape = shape[dim]
        
        crop_centers = np.arange(0, shape - overlap, stride) + crop / 2
        crop_centers = np.clip(crop_centers, a_max=shape - crop / 2, a_min=None)
        
        # Add final center
        crop_centers = np.append(crop_centers, shape - crop / 2)
        crop_centers = np.unique(crop_centers)
        
        return crop_centers
    
    def __call__(self, sample, lobe, image_spacing: np.ndarray, use_gt_crop=False):
        image = sample['image'].astype('float32')
        all_loc = sample['all_loc']
        all_rad = sample['all_rad']
        all_cls = sample['all_cls']
        all_prob = sample['all_prob']
        
        all_rad_pixel = all_rad / image_spacing
        all_nodule_bb_min = all_loc - all_rad_pixel / 2
        all_nodule_bb_max = all_loc + all_rad_pixel / 2
        nodule_bboxes = np.stack([all_nodule_bb_min, all_nodule_bb_max], axis=1) # [N, 2, 3]
        nodule_volumes = np.prod(all_rad_pixel, axis=1) # [N]
        
        gt_all_loc = sample['gt_all_loc']
        gt_all_rad = sample['gt_all_rad']
        gt_all_cls = sample['gt_all_cls']
        
        gt_all_rad_pixel = gt_all_rad / image_spacing
        gt_all_nodule_bb_min = gt_all_loc - gt_all_rad_pixel / 2
        gt_all_nodule_bb_max = gt_all_loc + gt_all_rad_pixel / 2
        gt_nodule_bboxes = np.stack([gt_all_nodule_bb_min, gt_all_nodule_bb_max], axis=1) # [N, 2, 3]
        gt_nodule_volumes = np.prod(gt_all_rad_pixel, axis=1) # [N]
        
        if len(gt_all_loc) != 0:
            gt_instance_loc = gt_all_loc[np.sum([gt_all_cls == cls for cls in self.sample_cls], axis=0, dtype='bool')]
        else:
            gt_instance_loc = []
        
        if len(all_loc) != 0:
            instance_loc = all_loc[np.sum([all_cls == cls for cls in self.sample_cls], axis=0, dtype='bool')]
        else:
            instance_loc = []
            
        shape = image.shape
        crop_size = np.array(self.crop_size)
        # Generate crop centers based on fixed stride
        z_crop_centers = self.get_crop_centers(shape, 0)
        y_crop_centers = self.get_crop_centers(shape, 1)
        x_crop_centers = self.get_crop_centers(shape, 2)
        
        # Generate crop centers
        crop_centers = np.array([*product(z_crop_centers, y_crop_centers, x_crop_centers)])
        if self.rand_trans is not None:
            crop_centers = crop_centers + np.random.randint(low=-self.rand_trans, high=self.rand_trans, size=(len(crop_centers), 3))
            
        # Get crop centers for instance
        if len(instance_loc) > 0 and not use_gt_crop:
            if self.rand_trans is not None:
                instance_crop = instance_loc + np.random.randint(low=-self.rand_trans * 2, high=self.rand_trans * 2, size=(len(instance_loc), 3))
            else:
                instance_crop = instance_loc
            crop_centers = np.append(crop_centers, instance_crop, axis=0)
        elif len(gt_instance_loc) > 0 and use_gt_crop:
            if self.rand_trans is not None:
                gt_instance_crop = gt_instance_loc + np.random.randint(low=-self.rand_trans * 2, high=self.rand_trans * 2, size=(len(gt_instance_loc), 3))
            else:
                gt_instance_crop = gt_instance_loc
            crop_centers = np.append(crop_centers, gt_instance_crop, axis=0)
                
        all_crop_bb_min = crop_centers - crop_size / 2
        all_crop_bb_min = np.clip(all_crop_bb_min, a_min=0, a_max=shape - crop_size)
        all_crop_bb_min = np.unique(all_crop_bb_min, axis=0)
        
        all_crop_bb_max = all_crop_bb_min + crop_size
        all_crop_bboxes = np.stack([all_crop_bb_min, all_crop_bb_max], axis=1) # [M, 2, 3]
        
        # Compute IoU to determine the label of the patches
        if len(nodule_bboxes) != 0:
            inter_volumes = compute_bbox3d_intersection_volume(all_crop_bboxes, nodule_bboxes) # [M, N]
            all_ious = inter_volumes / nodule_volumes[np.newaxis, :] # [M, N]
            max_ious = np.max(all_ious, axis=1) # [M]
        else:
            all_ious = np.zeros((len(all_crop_bboxes), 1))
            max_ious = np.zeros(len(all_crop_bboxes))
            
        if len(gt_instance_loc) > 0:
            gt_inter_volumes = compute_bbox3d_intersection_volume(all_crop_bboxes, gt_nodule_bboxes) # [M, N]
            gt_all_ious = gt_inter_volumes / gt_nodule_volumes[np.newaxis, :] # [M, N]
            gt_max_ious = np.max(gt_all_ious, axis=1) # [M]
        else:
            gt_all_ious = np.zeros((len(all_crop_bboxes), 1))
            gt_max_ious = np.zeros(len(all_crop_bboxes))    
        
        if not use_gt_crop:
            tp_indices = max_ious > 0
        else:
            tp_indices = gt_max_ious > 0
        neg_indices = ~tp_indices

        # Sample patches
        tp_prob = self.tp_ratio / tp_indices.sum() if tp_indices.sum() > 0 else 0
        probs = np.zeros(shape=len(max_ious))
        probs[tp_indices] = tp_prob
        probs[neg_indices] = (1. - probs.sum()) / neg_indices.sum() if neg_indices.sum() > 0 else 0
        probs = probs / probs.sum() # normalize
        sample_indices = np.random.choice(np.arange(len(all_crop_bboxes)), size=self.sample_num, p=probs, replace=False)
        
        # Crop patches
        samples = []
        for sample_i in sample_indices:
            crop_bb_min = all_crop_bb_min[sample_i].astype(np.int32)
            crop_bb_max = crop_bb_min + crop_size
            image_crop = image[crop_bb_min[0]: crop_bb_max[0], 
                               crop_bb_min[1]: crop_bb_max[1], 
                               crop_bb_min[2]: crop_bb_max[2]]
            lobe_crop = lobe[crop_bb_min[0]: crop_bb_max[0],
                            crop_bb_min[1]: crop_bb_max[1],
                            crop_bb_min[2]: crop_bb_max[2]]
            
            resized_lobe_crop = scipy.ndimage.zoom(lobe_crop, self.lobe_zoom_ratio, order=0, mode='nearest')
            
            ious = all_ious[sample_i] # [N]
            in_idx = np.where(ious > 0)[0]
            if in_idx.size > 0:
                # Compute new ctr and rad because of the crop
                all_nodule_bb_min_crop = all_nodule_bb_min - crop_bb_min
                all_nodule_bb_max_crop = all_nodule_bb_max - crop_bb_min
                
                nodule_bb_min_crop = all_nodule_bb_min_crop[in_idx]
                nodule_bb_max_crop = all_nodule_bb_max_crop[in_idx]
                
                ctr = (nodule_bb_min_crop + nodule_bb_max_crop) / 2
                rad = nodule_bb_max_crop - nodule_bb_min_crop
                cls = all_cls[in_idx]
                prob = all_prob[in_idx]
            else:
                ctr = np.array([]).reshape(-1, 3)
                rad = np.array([])
                cls = np.array([])
                prob = np.array([])
            
            if len(gt_instance_loc) > 0:
                gt_ious = gt_all_ious[sample_i] # [N]
                in_idx_gt = np.where(gt_ious > 0)[0]
            else:
                in_idx_gt = np.array([])
                
            if in_idx_gt.size > 0:
                # Compute new ctr and rad because of the crop
                gt_all_nodule_bb_min_crop = gt_all_nodule_bb_min - crop_bb_min
                gt_all_nodule_bb_max_crop = gt_all_nodule_bb_max - crop_bb_min
                
                gt_nodule_bb_min_crop = gt_all_nodule_bb_min_crop[in_idx_gt]
                gt_nodule_bb_max_crop = gt_all_nodule_bb_max_crop[in_idx_gt]
                
                gt_ctr = (gt_nodule_bb_min_crop + gt_nodule_bb_max_crop) / 2
                gt_rad = gt_nodule_bb_max_crop - gt_nodule_bb_min_crop
                gt_cls = gt_all_cls[in_idx_gt]
            else:
                gt_ctr = np.array([]).reshape(-1, 3)
                gt_rad = np.array([])
                gt_cls = np.array([])
                
            CT_crop = np.expand_dims(image_crop, axis=0)
            resized_lobe_crop = np.expand_dims(resized_lobe_crop, axis=0)
            shape = np.array(CT_crop.shape[1:])
                
            sample = dict()
            sample['image'] = CT_crop
            sample['lobe'] = resized_lobe_crop
            sample['ctr'] = ctr
            sample['rad'] = rad
            sample['cls'] = cls
            sample['prob'] = prob
            
            sample['gt_ctr'] = gt_ctr
            sample['gt_rad'] = gt_rad
            sample['gt_cls'] = gt_cls
            
            sample['crop_bb_min'] = crop_bb_min
            
            sample['spacing'] = image_spacing
            samples.append(sample)
        return samples