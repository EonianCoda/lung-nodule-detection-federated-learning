# -*- coding: utf-8 -*-
import logging
import pickle
import copy
import torchvision
import numpy as np
from typing import List, Tuple

from torch.utils.data import Dataset
import torch

from .utils import load_series_list, load_image, load_label, load_lobe, ALL_RAD, ALL_LOC, ALL_CLS, ALL_PROB, \
                    gen_dicom_path, gen_label_path, gen_lobe_path, normalize_processed_image, normalize_raw_image, \
                    compute_bbox3d_iou
from transform.ctr_transform import OffsetMinusCTR, RotateCTR
from transform.feat_transform import FlipFeatTransform, Rot90FeatTransform
from fl_modules.utilities.box_utils import nms_3D
import math

logger = logging.getLogger(__name__)

def get_prob_threshold(rads, base_threshold: float = 0.6, small_size_threshold: int = 200) -> np.ndarray:
    sizes = np.prod(rads, axis=1)
    thresholds = np.array([base_threshold] * len(sizes))
    
    small_threshold = base_threshold - 0.1
    small_size_mask = sizes < small_size_threshold
    small_sizes = sizes[small_size_mask]
    
    thresholds[small_size_mask] = base_threshold - (base_threshold - small_threshold) * (small_size_threshold - small_sizes) / small_size_threshold
    
    return thresholds

class TrainDataset(Dataset):
    """Dataset for loading numpy images with dimension order [D, H, W]

    Arguments:
        series_list_path (str): Path to the series list file.
        image_spacing (List[float]): Spacing of the image in the order [z, y, x].
        transform_post (optional): Transform object to be applied after cropping.
        crop_fn (optional): Cropping function.
        use_bg (bool, optional): Flag indicating whether to use background or not.

    Attributes:
        labels (List): List of labels.
        dicom_paths (List): List of DICOM file paths.
        series_list_path (str): Path to the series list file.
        series_names (List): List of series names.
        image_spacing (ndarray): Spacing of the image in the order [z, y, x].
        transform_post (optional): Transform object to be applied after cropping.
        crop_fn (optional): Cropping function.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the item at the given index.

    """
    def __init__(self, series_list_path: str, image_spacing: List[float], transform_post=None, crop_fn=None, 
                 use_bg=False, min_d=0, min_size: int = 0, norm_method='scale', mmap_mode=None):
        self.labels = []
        self.dicom_paths = []
        self.series_list_path = series_list_path
        self.norm_method = norm_method
        self.image_spacing = np.array(image_spacing, dtype=np.float32) # (z, y, x)
        self.min_d = int(min_d)
        self.min_size = int(min_size)
        
        if self.min_d > 0:
            logger.info('When training, ignore nodules with depth less than {}'.format(min_d))
        if self.min_size != 0:
            logger.info('When training, ignore nodules with size less than {}'.format(min_size))
        
        if self.norm_method == 'mean_std':
            logger.info('Normalize image to have mean 0 and std 1, and then scale to -1 to 1')
        elif self.norm_method == 'scale':
            logger.info('Normalize image to have value ranged from -1 to 1')
        elif self.norm_method == 'none':
            logger.info('Normalize image to have value ranged from 0 to 1')
        
        if use_bg:
            logger.info('Using background(healthy lung) as training data')
        
        self.series_infos = load_series_list(series_list_path)
        for folder, series_name in self.series_infos:
            label_path = gen_label_path(folder, series_name)
            dicom_path = gen_dicom_path(folder, series_name)
           
            label = load_label(label_path, self.image_spacing, min_d, min_size)
            if label[ALL_LOC].shape[0] == 0 and not use_bg:
                continue
            
            self.dicom_paths.append(dicom_path)
            self.labels.append(label)

        self.transform_post = transform_post
        self.crop_fn = crop_fn
        self.mmap_mode = mmap_mode

    def __len__(self):
        return len(self.labels)
    
    def load_image(self, dicom_path: str) -> np.ndarray:
        """
        Return:
            A 3D numpy array with dimension order [D, H, W] (z, y, x)
        """
        image = np.load(dicom_path, mmap_mode=self.mmap_mode)
        image = np.transpose(image, (2, 0, 1))
        return image
    
    def __getitem__(self, idx):
        dicom_path = self.dicom_paths[idx]
        series_folder = self.series_infos[idx][0]
        series_name = self.series_infos[idx][1]
        label = self.labels[idx]

        image_spacing = self.image_spacing.copy() # z, y, x
        image = self.load_image(dicom_path) # z, y, x
        
        samples = {}
        samples['image'] = image
        samples['all_loc'] = label['all_loc'] # z, y, x
        samples['all_rad'] = label['all_rad'] # d, h, w
        samples['all_cls'] = label['all_cls']
        samples['file_name'] = series_name
        samples = self.crop_fn(samples, image_spacing)
        random_samples = []

        for i in range(len(samples)):
            sample = samples[i]
            sample['image'] = normalize_raw_image(sample['image'])
            sample['image'] = normalize_processed_image(sample['image'], self.norm_method)
            if self.transform_post:
                sample['ctr_transform'] = []
                sample['feat_transform'] = []
                sample = self.transform_post(sample)
            random_samples.append(sample)

        return random_samples

class DetDataset(Dataset):
    """Detection dataset for inference
    """
    def __init__(self, series_list_path: str, image_spacing: List[float], SplitComb, norm_method='scale'):
        self.series_list_path = series_list_path
        
        self.labels = []
        self.dicom_paths = []
        self.norm_method = norm_method
        self.image_spacing = np.array(image_spacing, dtype=np.float32) # (z, y, x)
        self.series_infos = load_series_list(series_list_path)
        
        for folder, series_name in self.series_infos:
            dicom_path = gen_dicom_path(folder, series_name)
            self.dicom_paths.append(dicom_path)
        self.splitcomb = SplitComb
        
    def __len__(self):
        return len(self.dicom_paths)
    
    def __getitem__(self, idx):
        dicom_path = self.dicom_paths[idx]
        series_folder = self.series_infos[idx][0]
        series_name = self.series_infos[idx][1]
        
        image_spacing = self.image_spacing.copy() # z, y, x
        image = load_image(dicom_path) # z, y, x
        image = normalize_processed_image(image, self.norm_method)

        data = {}
        # split_images [N, 1, crop_z, crop_y, crop_x]
        split_images, nzhw, image_shape = self.splitcomb.split(image)
        data['split_images'] = np.ascontiguousarray(split_images)
        data['nzhw'] = nzhw
        data['spacing'] = image_spacing
        data['series_name'] = series_name
        data['series_folder'] = series_folder
        data['image_shape'] = image_shape
        return data

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
                offset[axis] = input_shape[axis]
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

class UnLabeledDataset(Dataset):
    def __init__(self, series_list_path: str, image_spacing: List[float], strong_aug = None, crop_fn=None, use_bg=False, 
                 min_d=0, min_size: int = 0, norm_method='scale', mmap_mode=None, use_gt_crop=True, pseudo_remove_threshold=0.4,
                  use_rotate90=False, small_size_threshold=200,
                  pseudo_crop_threshold = 0.5, pseudo_update_ema_alpha = 0.9, pseudo_label_pkl_path=None, **kwargs):
        self.series_list_path = series_list_path
        self.norm_method = norm_method
        self.image_spacing = np.array(image_spacing, dtype=np.float32) # (z, y, x)
        self.min_d = int(min_d)
        self.min_size = int(min_size)
        self.psuedo_remove_threshold = pseudo_remove_threshold
        self.pseudo_update_ema_alpha = pseudo_update_ema_alpha
        self.pseudo_crop_threshold = pseudo_crop_threshold
        
        self.pseudo_label_threshold = kwargs['pseudo_label_threshold']
        
        
        if self.min_d > 0:
            logger.info('When training, ignore nodules with depth less than {}'.format(min_d))
        
        if self.norm_method == 'mean_std':
            logger.info('Normalize image to have mean 0 and std 1, and then scale to -1 to 1')
        elif self.norm_method == 'scale':
            logger.info('Normalize image to have value ranged from -1 to 1')
        elif self.norm_method == 'none':
            logger.info('Normalize image to have value ranged from 0 to 1')
        
        self.all_dicom_paths = []
        self.all_series_names = []
        self.all_lobe_paths = []
        self.dicom_paths = []
        self.series_names = []
        self.lobe_paths = []
        
        self.labels = dict()
        self.gt_labels = dict()
        self.series_infos = load_series_list(series_list_path)
        
        for folder, series_name in self.series_infos:
            gt_label_path = gen_label_path(folder, series_name)
            gt_label = load_label(gt_label_path, self.image_spacing, min_d, min_size)
            dicom_path = gen_dicom_path(folder, series_name)
            self.all_dicom_paths.append(dicom_path)
            self.all_series_names.append(series_name)
            self.all_lobe_paths.append(gen_lobe_path(folder, series_name))
            self.gt_labels[series_name] = gt_label
        
        self.dicom_paths = copy.deepcopy(self.all_dicom_paths)
        self.series_names = copy.deepcopy(self.all_series_names)
        
        if not use_rotate90:
            tta_transforms = [[FlipTransform(flip_depth=False, flip_height=False, flip_width=True)],
                                [FlipTransform(flip_depth=False, flip_height=True, flip_width=False)],
                                [FlipTransform(flip_depth=True, flip_height=False, flip_width=False)]]
            raw_weight = 0.5
        else:
            tta_transforms = [[FlipTransform(flip_depth=False, flip_height=False, flip_width=True)],
                            [FlipTransform(flip_depth=False, flip_height=True, flip_width=False)],
                            [FlipTransform(flip_depth=True, flip_height=False, flip_width=False)],
                            [Rotate90(rot_xy = True, rot_xz = False, rot_yz = False)],
                            [Rotate90(rot_xy = False, rot_xz = True, rot_yz = False)],
                            [Rotate90(rot_xy = False, rot_xz = False, rot_yz = True)]]
            raw_weight = 0.3
            
        self.tta_transforms = []
        for i in range(len(tta_transforms)):
            self.tta_transforms.append(torchvision.transforms.Compose(tta_transforms[i]))
        self.tta_trans_weight = [raw_weight] + [(1 - raw_weight) / len(tta_transforms)] * len(tta_transforms) # first one is for no augmentation
        self.tta_trans_weight = np.array([w / sum(self.tta_trans_weight) for w in self.tta_trans_weight]) # normalize to 1
        
        self.strong_aug = strong_aug
        self.crop_fn = crop_fn
        self.mmap_mode = mmap_mode
        self.use_gt_crop = use_gt_crop
        self.small_size_threshold = small_size_threshold
        
        if pseudo_label_pkl_path is not None:
            with open(pseudo_label_pkl_path, 'rb') as f:
                pseu_labels = pickle.load(f)
            for series_name, label in pseu_labels.items():
                prob = label[ALL_PROB]
                if len(prob) != 0:
                    valid_mask = (prob > get_prob_threshold(label[ALL_RAD], self.pseudo_label_threshold))
                    if len(valid_mask) == 0:
                        label = {ALL_LOC: np.zeros((0, 3)),
                                ALL_RAD: np.zeros((0,)),
                                ALL_CLS: np.zeros((0, 3), dtype=np.int32),
                                ALL_PROB: np.zeros((0,))}
                    else:
                        label = {key: value[valid_mask] for key, value in label.items()}
                    pseu_labels[series_name] = label
                    
            self.set_pseu_labels(pseu_labels)
        
    def set_pseu_labels(self, labels):
        """
        the shape in labels is pixel spacing, and the order is [z, y, x]
        """
        new_labels = {}
        keep_indices = []
        for i, series_name in enumerate(self.all_series_names):
            if series_name in labels and len(labels[series_name]['all_loc']) > 0:
                keep_indices.append(i)
                new_labels[series_name] = labels[series_name]
                
        self.labels = new_labels
        self.dicom_paths = [self.all_dicom_paths[i] for i in keep_indices]
        self.series_names = [self.all_series_names[i] for i in keep_indices]
        self.lobe_paths = [self.all_lobe_paths[i] for i in keep_indices]
        
        self.ema_updated_labels = dict()
        
    def __len__(self):
        return len(self.dicom_paths)

    def load_image(self, dicom_path: str) -> np.ndarray:
        """
        Return:
            A 3D numpy array with dimension order [D, H, W] (z, y, x)
        """
        image = np.load(dicom_path, mmap_mode=self.mmap_mode)
        image = np.transpose(image, (2, 0, 1))
        return image
    
    def update_pseudo_label(self, series_name, ctrs, rads, probs):
        label = dict()
        label[ALL_LOC] = np.array(ctrs, dtype=np.float32)
        label[ALL_RAD] = np.array(rads, dtype=np.float32)
        label[ALL_PROB] = np.array(probs, dtype=np.float32)
        
        if series_name in self.ema_updated_labels:
            self.ema_updated_labels[series_name][ALL_LOC] = np.concatenate([self.ema_updated_labels[series_name][ALL_LOC], label[ALL_LOC]], axis=0)
            self.ema_updated_labels[series_name][ALL_RAD] = np.concatenate([self.ema_updated_labels[series_name][ALL_RAD], label[ALL_RAD]], axis=0)
            self.ema_updated_labels[series_name][ALL_PROB] = np.concatenate([self.ema_updated_labels[series_name][ALL_PROB], label[ALL_PROB]], axis=0)
        else:
            self.ema_updated_labels[series_name] = label
            
    def confirm_pseudo_labels(self):
        if len(self.ema_updated_labels) == 0:
            logger.warning('No pseudo label is updated')
            return
        
        labels = dict()
        for series_name in self.labels.keys():
            if series_name in self.ema_updated_labels:
                new_label = self.ema_updated_labels[series_name]
                new_ctrs = new_label[ALL_LOC]
                new_rads = new_label[ALL_RAD]
                new_probs = new_label[ALL_PROB]
            else:
                new_ctrs = np.zeros((0, 3), dtype=np.float32)
                new_rads = np.zeros((0, 3), dtype=np.float32)
                new_probs = np.zeros((0,), dtype=np.float32)
            
            history_label = self.labels[series_name]
            history_ctrs = history_label[ALL_LOC]
            history_rads = history_label[ALL_RAD]
            history_probs = history_label[ALL_PROB]
            if len(history_ctrs) > 0:
                history_bbxes = np.stack([history_ctrs - history_rads / 2, history_ctrs + history_rads / 2], axis=1)
            else:
                history_bbxes = np.zeros((0, 2, 3), dtype=np.float32)
            
            if len(new_ctrs) == 0 and len(history_ctrs) == 0:
                continue
            # Construct the pseudo labels
            # (N, 7) [prob, ctr_z, ctr_y, ctr_x, d, h, w]
            if len(new_ctrs) > 0:
                if len(history_bbxes) != 0:
                    new_bboxes = np.stack([new_ctrs - new_rads / 2, new_ctrs + new_rads / 2], axis=1)
                    matched_ious = compute_bbox3d_iou(history_bbxes, new_bboxes)
                    history_matched_ious = np.max(matched_ious, axis=1)
                    valid_mask = history_matched_ious >= 0.05
                    if np.count_nonzero(valid_mask) > 0:
                        history_probs[valid_mask] *= self.pseudo_update_ema_alpha # Penalize the history labels
                    
                new_dets = np.concatenate([new_probs.reshape(-1, 1), new_ctrs.reshape(-1, 3), new_rads.reshape(-1, 3)], axis=1).astype('float32')
            else:
                new_dets = np.zeros((0, 7), dtype=np.float32)
                
            if len(history_ctrs) > 0:
                history_dets = np.concatenate([history_probs.reshape(-1, 1), history_ctrs.reshape(-1, 3), history_rads.reshape(-1, 3)], axis=1).astype('float32')
            else:
                history_dets = np.zeros((0, 7), dtype=np.float32)
        
            dets = np.concatenate([new_dets, history_dets], axis=0)          
            dets = torch.from_numpy(dets)
            
            # NMS
            keep = nms_3D(dets, overlap=0.05, top_k=20)
            dets = dets[keep.long()]
            dets = dets.numpy()
            probs = dets[:, 0]
            ctrs = dets[:, 1:4]
            rads = dets[:, 4:7]
            
            valid_mask = (probs >= get_prob_threshold(rads, self.psuedo_remove_threshold, self.small_size_threshold))
            if np.count_nonzero(valid_mask) == 0:
                continue
            
            probs = probs[valid_mask]
            ctrs = ctrs[valid_mask]
            rads = rads[valid_mask]
            
            labels[series_name] = {ALL_LOC: ctrs, 
                                   ALL_RAD: rads, 
                                   ALL_PROB: probs,
                                   ALL_CLS: np.zeros((len(ctrs),), dtype=np.float64)}
        
        self.set_pseu_labels(labels)
        # Reset the updated labels
        self.ema_updated_labels = dict()
     
    def get_pseudo_recall_precision(self, iou_threshold = 0.1):
        tp, fp, fn = 0, 0, 0
        for series_name in self.labels.keys():
            label = self.labels[series_name]
            gt_label = self.gt_labels[series_name]
            
            prob = label[ALL_PROB]
            valid_mask = prob >= get_prob_threshold(label[ALL_RAD], self.pseudo_label_threshold)
            all_loc = label[ALL_LOC][valid_mask]
            all_rad = label[ALL_RAD][valid_mask]
            if len(all_loc) == 0 and len(gt_label[ALL_LOC]) == 0:
                continue
            elif len(all_loc) == 0:
                fn += len(gt_label[ALL_LOC])
                continue
            elif len(gt_label[ALL_LOC]) == 0:
                fp += len(all_loc)
                continue
            
            # Compute iou
            label_bboxes = np.stack([all_loc - all_rad / 2, all_loc + all_rad / 2], axis=1)
            gt_bboxes = np.stack([gt_label[ALL_LOC] - gt_label[ALL_RAD] / 2, gt_label[ALL_LOC] + gt_label[ALL_RAD] / 2], axis=1)
            ious = compute_bbox3d_iou(label_bboxes, gt_bboxes)
            
            matched_gt_ious = np.max(ious, axis=0)
            matched_ious = np.max(ious, axis=1)
            
            tp += np.count_nonzero(matched_ious >= iou_threshold)
            fn += np.count_nonzero(matched_gt_ious < iou_threshold)
            fp += np.count_nonzero(matched_ious < iou_threshold)
        recall = tp / (tp + fn + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        return recall, precision, tp, fp, fn
        
    def __getitem__(self, idx):
        dicom_path = self.dicom_paths[idx]
        lobe_path = self.lobe_paths[idx]
        series_name = self.series_names[idx]
        
        label = self.labels[series_name].copy()
        gt_label = self.gt_labels[series_name].copy()

        image_spacing = self.image_spacing.copy() # z, y, x
        image = self.load_image(dicom_path) # z, y, x
        lobe = load_lobe(lobe_path) # z, y, x
        
        samples = {}
        samples['image'] = image
        # We need to convert pixel spacing to world spacing
        samples[ALL_LOC] = label[ALL_LOC] # z, y, x
        samples[ALL_RAD] = label[ALL_RAD] # d, h, w
        samples[ALL_CLS] = label[ALL_CLS]
        samples[ALL_PROB] = label[ALL_PROB]
        
        samples['gt_all_loc'] = gt_label[ALL_LOC]
        samples['gt_all_rad'] = gt_label[ALL_RAD]
        samples['gt_all_cls'] = gt_label[ALL_CLS]
        
        # Before cropping, we need to convert the pixel spacing to world spacing
        # If we use ground truth crop, we do not need to convert the pixel spacing to world spacing because this step is done in the `load_label` function
        if len(samples[ALL_RAD]) > 0:
            samples[ALL_RAD] = samples[ALL_RAD] * image_spacing # d, h, w
        samples['file_name'] = series_name
        samples = self.crop_fn(samples, lobe, image_spacing, self.use_gt_crop)
        random_samples = []

        raw_images = []
        raw_lobes = []
        raw_annots = []
        strong_samples = []
        for i in range(len(samples)):
            sample = samples[i]
            sample['image'] = normalize_raw_image(sample['image'])
            sample['image'] = normalize_processed_image(sample['image'], self.norm_method)
            sample['ctr_transform'] = []
            sample['feat_transform'] = []

            raw_images.append(sample['image'].copy())
            raw_lobes.append(sample['lobe']) # not use copy because we do not modify the lobe in strong augmentation
            annots = {'ctr': sample['ctr'].copy(), 
                      'rad': sample['rad'].copy(), 
                      'cls': sample['cls'].copy(),
                      'prob': sample['prob'].copy(),
                      'gt_ctr': sample['gt_ctr'].copy(),
                      'gt_rad': sample['gt_rad'].copy(),
                      'gt_cls': sample['gt_cls'].copy(),
                      'spacing': sample['spacing'].copy(),
                      'crop_bb_min': sample['crop_bb_min'].copy()}
            raw_annots.append(annots)
            strong_samples.append(self.strong_aug(sample))
        
        # Process the weak samples
        raw_images = np.stack(raw_images, axis=0) # (n, z, y, x)
        raw_lobes = np.stack(raw_lobes, axis=0) # (n, 1, z // out_stride, y // out_stride, x // out_stride)
        sample = {'image': raw_images, 'ctr_transform': [], 'feat_transform': []}
        weak_samples = [sample]
        for i, transform_fn in enumerate(self.tta_transforms):
            weak_samples.append(transform_fn(copy.deepcopy(sample)))
        weak_ctr_transforms = [s['ctr_transform'] for s in weak_samples]
        weak_feat_transforms = [s['feat_transform'] for s in weak_samples]
        weak_images = np.stack([s['image'] for s in weak_samples], axis=1) # (n, tta, 1, z, y, x)
        weak_samples = {'image': weak_images,
                        'lobe': raw_lobes, # (n, z, y, x)
                        'ctr_transform': weak_ctr_transforms, # (tta,)
                        'feat_transform': weak_feat_transforms, # (tta,)
                        'transform_weight': self.tta_trans_weight.copy(),
                        'series_name': [series_name] * raw_images.shape[0],
                        'annots': raw_annots}
        
        # Process the final samples
        random_samples = dict()
        random_samples['weak'] = weak_samples
        random_samples['strong'] = strong_samples
        
        return random_samples
    