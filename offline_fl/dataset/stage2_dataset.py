import os
from os.path import join
import shutil
import psutil
import logging
import random
import json
import copy
import numpy as np
import numpy.typing as npt

import torch
from torch.utils.data import Dataset

from multiprocessing.pool import Pool
from scipy import ndimage as nd
from typing import Tuple, Dict, List, Any, Union
from .utils import get_nodule_type, load_series_list
from .augmentation import RandomFlipYXZ, RandomRotation90, RandomRotation, RandomCutout, random_color, random_blur, random_gauss_noise

logger = logging.getLogger(__name__)
MIN_MEM_GB = 100

def reset_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def auto_thickness(threshold_upper_limit: float):
    def cal_prob_threshold(thickness: int) -> float:
        if thickness <= 3:
            return threshold_upper_limit
        elif thickness >= 9:
            return max(threshold_upper_limit - 0.2, 0)
        else:
            # line (T, 3) to (T-0.2, 9)
            # y = ax + b
            a = -30
            b = 3 - a*threshold_upper_limit

            # we're calculating x, hence x = y/a - b/a
            return thickness/a - b/a
    return cal_prob_threshold

def get_threshold_based_nodule_3d_thickness(thickness: int) -> float:
    return 0.5
    # if isinstance(thickness, np.ndarray) and len(thickness.shape) == 1:
    #     thickness = thickness[0]
    
    # thickness = int(thickness)
    
    # threshold_upper_limit = 0.4
    # if thickness <= 3:
    #     return threshold_upper_limit
    # elif thickness >= 9:
    #     return max(threshold_upper_limit - 0.2, 0)
    # else:
    #     # line (T, 3) to (T-0.2, 9)
    #     # y = ax + b
    #     a = -30
    #     b = 3 - a*threshold_upper_limit

    #     # we're calculating x, hence x = y/a - b/a
    #     return thickness/a - b/a

    # # adaptive threshold for 3D model
    # if thickness <= 3:
    #     return 0.5
    # elif thickness >= 9:
    #     return 0.3
    # else:
    #     return (15 - thickness) / 30

class Stage2Dataset(Dataset):
    """
        Train/validation data generator
    """
    def __init__(self,
                dataset_type: str,
                nodule_size_ranges: Dict[str, Tuple[int, int]],
                num_nodules: Dict[str, int],
                series_list_path: str,
                crop_setting: dict,
                cache_folder: str,
                reset_data_in_disk: bool = True,
                prepare_data_in_disk: bool = True,
                cache_in_memory: bool = True):
        """
            In the initialization function, we load all the annotations and 
            record the path to the corresponding dicom series.
        """
        super(Stage2Dataset, self).__init__()
        self.dataset_type = dataset_type
        self.nodule_size_ranges = nodule_size_ranges
        # Crop Setting
        self.crop_setting = crop_setting
        self.crop_shape = crop_setting['crop_shape']
        self.final_shape = crop_setting['final_shape']
        
        if self.dataset_type == 'train':
            self.crop_offset = np.array([6, 6, 4], dtype = np.int32)
        else:
            self.crop_offset = np.array([0, 0, 0], dtype = np.int32)
            
        self.series_list_path = series_list_path
        series_infos = load_series_list(self.series_list_path)
        self.num_patient = len(series_infos)
        
        self.num_nodule_in_dataset = num_nodules
        # collect paths to resampled CT scan .npy file and corresponding annotation .txt file
        self.annotations = []
        self.series_paths = []
        self.nodule_3d_indices = []
        self.labels = []
        self.nodule_info_mapping = dict()
        self.series_nodule_bboxes = []
        num_tp_nodule = 0
        self.num_nodule_3d = 0

        self.metric_born = {key: np.array([0, 0, 0, 0], dtype=np.int32) for key in self.nodule_size_ranges} # [tp, fp, fn, tn]
        self.nodule_hitting_born = set()
        
        for series_folder, file_name in series_infos:
            npy_image_path = os.path.join(series_folder, 
                                        'npy', 
                                        f'{file_name}.npy')
            annotation_path = os.path.join(series_folder, 
                                        'stage1_post_process', 
                                        f'{file_name}.txt')
            
            series_nodules_counts_path = os.path.join(series_folder,
                                        'mask', 
                                        f'{file_name}_nodule_count.json')
            
            # Read nodule information in series.
            with open(series_nodules_counts_path, 'r') as f:
                series_nodules_counts = json.load(f)
            nodule_bboxes = np.array(series_nodules_counts['bboxes'], dtype = np.int32) # shape = (N, 2, 3)
            with open(annotation_path, 'r') as f:
                nodule_3d_list = f.readlines()[1:]

            for nodule_3d in nodule_3d_list:
                nodule_bboxs_2d = nodule_3d.split(' ')
                gt_nodule_index, gt_nodule_sizes, pred_nodule_sizes = [int(c) for c in nodule_bboxs_2d[0].split(',')]
                
                is_true_nodule = (gt_nodule_sizes > 0)
            
                nodule_bboxs_2d = nodule_bboxs_2d[1:]
                nodule_thickness = len(nodule_bboxs_2d)
                if nodule_thickness == 0:
                    continue
                self.nodule_info_mapping[self.num_nodule_3d] = [int(is_true_nodule), series_folder, npy_image_path, gt_nodule_index, gt_nodule_sizes, pred_nodule_sizes]
                
                # Get bbox of center
                bbox_2ds = [[float(c) for c in bbox_2d.split(',')] for bbox_2d in nodule_bboxs_2d] # (slice_id, y_min, x_min, y_max, x_max, iou)
                bbox_2ds = np.array(bbox_2ds)
                y_min, x_min = np.min(bbox_2ds[:,[1, 2]], axis=0)
                y_max, x_max = np.max(bbox_2ds[:,[3, 4]], axis=0)
                bbox = [bbox_2ds[nodule_thickness // 2][0], y_min, x_min, y_max, x_max, nodule_thickness] # (slice_id, y_min, x_min, y_max, x_max, nodule_thickness)
                # Count number of true nodule
                if is_true_nodule:
                    num_tp_nodule += 1
                    self.labels.append([1])
                else:
                    self.labels.append([0])
                
                self.annotations.append(bbox)
                
                self.series_paths.append(npy_image_path)
                # self.series_nodule_bboxes.append(nodule_bboxes)
                self.nodule_3d_indices.append(self.num_nodule_3d)
                self.num_nodule_3d += 1

        self.nodule_3d_indices = np.array(self.nodule_3d_indices)
        self.alpha = 1 - (num_tp_nodule / self.num_nodule_3d) # alpha for focal loss
        
        # The setting of cache
        self.cache = dict()
        self.prepare_data_in_disk = prepare_data_in_disk
        self.reset_datas = reset_data_in_disk
        self.already_prepare_data = False
        mem_gib = psutil.virtual_memory().total // 1024 ** 3
        self.cache_in_memory = cache_in_memory if mem_gib >= MIN_MEM_GB else False

        self.cache_path = os.path.join(cache_folder, 'stage2_cache')
        os.makedirs(self.cache_path, exist_ok = True)
        self.prepare_datas()

    def compute_nodule_3d_metrics(self, 
                                indices: npt.NDArray[np.int32], 
                                preds: npt.NDArray[np.float32],
                                gt_length_threshold = 1,
                                fp_per_patient = None) -> Dict[str, Dict[str, float]]:
        if not isinstance(indices, np.ndarray):
            indices = np.array(indices, dtype = np.int32)
        if not isinstance(preds, np.ndarray):
            preds = np.array(preds, dtype = np.float32)

        def nodule_3d_matching(preds, indices, cal_prob_threshold):
            thickness = np.expand_dims(np.array([annot[-1] for annot in self.annotations]), axis=1)
            threshold = np.apply_along_axis(cal_prob_threshold, axis = 1, arr = thickness)
            preds = preds[np.argsort(indices)]
            pred_2d_positives = (preds >= threshold).astype(np.int32)
            metric_result = {k: np.zeros(4, dtype=np.int32) for k in self.nodule_size_ranges.keys()}

            nodule_hitting_map = copy.deepcopy(self.nodule_hitting_born)
            
            for nodule_3d_id in self.nodule_info_mapping.keys():
                is_true_nodule, series_folder, npy_image_path, gt_nodule_index, gt_nodule_sizes, pred_nodule_sizes = self.nodule_info_mapping[nodule_3d_id]
                nodule_sizes = gt_nodule_sizes if is_true_nodule == 1 else pred_nodule_sizes
                target = (self.nodule_3d_indices == nodule_3d_id)
                pred_3d = (np.sum(pred_2d_positives[target]) >= gt_length_threshold)
                
                tp, fp, fn, tn = 0, 0, 0, 0
                nodule_hitting_key = (npy_image_path, gt_nodule_index)
                
                if (pred_3d == 1 and is_true_nodule == 1) and (nodule_hitting_key not in nodule_hitting_map):
                    tp += 1
                    nodule_hitting_map.add(nodule_hitting_key)
                elif pred_3d == 0 and is_true_nodule == 1:
                    fn += 1
                elif pred_3d == 1 and is_true_nodule == 0:
                    fp += 1
                else:
                    tn += 1

                metric_result[get_nodule_type(nodule_sizes, self.nodule_size_ranges)] += np.array([tp, fp, tn, fn], dtype=np.int32)

            # Add metric born
            for nodule_type, value in metric_result.items():
                metric_result[nodule_type] += self.metric_born[nodule_type]
            
            # Replace false negative(fn) by real number of nodule in dataset
            for nodule_type, value in metric_result.items():
                tp, fp, tn, fn = value
                real_num_nodule = self.num_nodule_in_dataset[nodule_type]
                fn = real_num_nodule - tp
                metric_result[nodule_type] = np.array([tp, fp, tn, fn], dtype=np.int32)
            
            metric_result['all'] = np.zeros(4, dtype=np.int32)
            for k, v in metric_result.items():
                if k == 'all':
                    continue
                metric_result['all'] += v
            return metric_result
        
        if fp_per_patient == None:
            result = nodule_3d_matching(preds, indices, get_threshold_based_nodule_3d_thickness)
        else:
            step = -0.025
            for threshold_upper_limit in np.arange(0.5, 0.25 + step, step):
                threshold_upper_limit = round(threshold_upper_limit, 3)
                cal_prob_threshold = auto_thickness(threshold_upper_limit)
                result = nodule_3d_matching(preds, indices, cal_prob_threshold)
                tp, fp, tn, fn = result['all']
                if (fp / self.num_patient) >= fp_per_patient:
                    break
        
        for nodule_type, value in result.items():
            tp, fp, tn, fn = value
            recall = tp / max(tp + fn, 1)
            precision = tp / max(tp + fp, 1)
            if recall + precision <= 0:
                f1_score = 0
            else:
                f1_score = 2 * ((recall * precision) / (recall + precision))

            if recall + precision <= 0:
                f2_score = 0
            else:
                f2_score = 5 * ((recall * precision) / (4 * recall + precision))

            if tp + fn + fp + tn <= 0:
                accuracy = 0
            else:
                accuracy = (tp + tn) / max(tp + fn + fp + tn, 1)

            metrics = {'recall': recall, 
                       'precision': precision, 
                       'f1_score': f1_score, 
                       'f2_score': f2_score,
                       'accuracy': accuracy,
                       'tp': tp,
                       'fp': fp,
                       'fn': fn,
                       'tn': tn}
            
            result[nodule_type] = metrics
    
        return result

    def compute_nodule_3d_metrics_of_each_series(self, 
                                                indices: npt.NDArray[np.int32], 
                                                preds: npt.NDArray[np.float32],
                                                nodule_count_of_series: Dict[str, int],
                                                gt_length_threshold = 2,
                                                threshold_upper_limit = 0.5) -> Dict[str, Dict[str, Union[int, float]]]:
        """
        Args:
            indices (npt.NDArray[np.int32]):
            preds (npt.NDArray[np.float32]):
        Returns: Dict[str, Dict[str, Union[int, float]]]
            A dict of (series_folder, metrics), metrics is a dict of (metric_name, value)
        """
        if not isinstance(indices, np.ndarray):
            indices = np.array(indices, dtype = np.int32)
        if not isinstance(preds, np.ndarray):
            preds = np.array(preds, dtype = np.float32)

        thickness = np.expand_dims(np.array([annot[-1] for annot in self.annotations]), axis=1)
        threshold = np.apply_along_axis(auto_thickness(threshold_upper_limit), axis = 1, arr = thickness)
        preds = preds[np.argsort(indices)]
        pred_2d_positives = (preds >= threshold).astype(np.int32)

        nodule_hitting_map = set()
        metric_result = dict()
        for nodule_3d_id in self.nodule_info_mapping.keys():
            is_true_nodule, series_folder, npy_image_path, gt_nodule_index, gt_nodule_sizes, pred_nodule_sizes = self.nodule_info_mapping[nodule_3d_id]
            nodule_sizes = gt_nodule_sizes if is_true_nodule == 1 else pred_nodule_sizes
            
            target = (self.nodule_3d_indices == nodule_3d_id)
            pred_3d = (np.sum(pred_2d_positives[target]) >= gt_length_threshold)
            
            tp, fp, fn, tn = 0, 0, 0, 0
            nodule_hitting_key = (npy_image_path, gt_nodule_index)
            
            if (pred_3d == 1 and is_true_nodule == 1) and (nodule_hitting_key not in nodule_hitting_map):
                tp += 1
                nodule_hitting_map.add(nodule_hitting_key)
            elif pred_3d == 0 and is_true_nodule == 1:
                fn += 1
            elif pred_3d == 1 and is_true_nodule == 0:
                fp += 1
            else:
                tn += 1

            if len(metric_result.get(series_folder, [])) == 0:
                metric_result[series_folder] = np.zeros((4,), dtype=np.int32)
            metric_result[series_folder] += np.array([tp, fp, tn, fn], dtype=np.int32)

        # Replace false negative(fn) by real number of nodule in dataset
        for series_folder, real_num_nodule in nodule_count_of_series.items():
            tp, fp, tn, fn = metric_result[series_folder]
            fn = real_num_nodule - tp
            recall = tp / max(tp + fn, 1)
            precision = tp / max(tp + fp, 1)
            if recall + precision <= 0:
                f1_score = 0
            else:
                f1_score = 2 * ((recall * precision) / (recall + precision))

            if recall + precision <= 0:
                f2_score = 0
            else:
                f2_score = 5 * ((recall * precision) / (4 * recall + precision))

            if tp + fn + fp + tn <= 0:
                accuracy = 0
            else:
                accuracy = (tp + tn) / max(tp + fn + fp + tn, 1)
            metric_result[series_folder] = {'recall': recall, 
                                            'precision': precision, 
                                            'f1_score': f1_score, 
                                            'f2_score': f2_score,
                                            'accuracy': accuracy,
                                            'tp': tp,
                                            'fp': fp,
                                            'fn': fn,
                                            'tn': tn}
    
        return metric_result

    def prepare_datas(self):
        logger.info("Start to prepare data!")

        save_folder = join(self.cache_path, self.dataset_type)
        if self.reset_datas:
            reset_folder(save_folder)
            
        self.cache_paths = []
        try:
            tasks = []
            pool = Pool(os.cpu_count() // 2)
            for i in range(len(self)):
                series_path = self.series_paths[i]
                annotation = self.annotations[i]
                label = self.labels[i]
                cache_path = os.path.join(save_folder, '{}.npz'.format(i))
                self.cache_paths.append(cache_path)
                tasks.append((series_path, annotation, label, self.crop_setting, cache_path, False, self.crop_offset))
                
            pool.starmap(self._prepare_data, tasks)
            pool.close()
            pool.join()
        finally:
            pool.terminate()
    
    @staticmethod
    def _prepare_data(series_path: str, 
                      annotation: tuple, 
                      label: List[int], 
                      crop_setting: Dict[str, List[int]], 
                      save_path: str, 
                      return_data: bool, 
                      crop_offset: npt.NDArray[np.int32]) -> Union[Dict[str, np.ndarray], None]:
        crop_shape = copy.deepcopy(crop_setting['crop_shape'])
        final_shape = copy.deepcopy(crop_setting['final_shape'])
        
        # If the crop offset is not zero, we need to adjust the patch shape and final shape
        if crop_offset[0] != 0:
            crop_shape = np.array(crop_shape, dtype = np.int32)
            final_shape = np.array(final_shape, dtype = np.int32)
            crop_shape = (crop_shape + (crop_offset * (crop_shape / final_shape))).astype(np.int32)
            final_shape = (final_shape + crop_offset).astype(np.int32)
            
        image = np.load(series_path, mmap_mode = 'c')
        
        patch_h, patch_w, patch_d  = crop_shape
        final_h, final_w, final_d  = final_shape
        image_h, image_w, image_d = image.shape
    
        # Read annotation
        slice_number, y_min, x_min, y_max, x_max, thickness = annotation
        slice_number = int(slice_number)
        thickness = int(thickness)
        y_center = round(y_min + (y_max - y_min) / 2)
        x_center = round(x_min + (x_max - x_min) / 2)
        
        # Calculate the patch range
        patch_start_x = int(np.clip(x_center - (patch_w // 2), 0, image_w - patch_w))
        patch_start_y = int(np.clip(y_center - (patch_h // 2), 0, image_h - patch_h))
        patch_start_z = int(np.clip(slice_number - (patch_d // 2), 0, image_d - patch_d))
        
        patch = image[patch_start_y : patch_start_y + patch_h, 
                      patch_start_x : patch_start_x + patch_w, 
                      patch_start_z : patch_start_z + patch_d]
        
        # Convert HU values to (0, 1400)
        patch = np.clip(patch, -1000, 400) + 1000
           
        # Resize patch to final shape
        patch_resized = nd.zoom(patch, zoom=(final_h / patch_h,
                                            final_w / patch_w,
                                            final_d / patch_d), 
                                            mode = 'nearest',
                                            order = 3)
        
        label = np.array(label, dtype = np.int16)
        thickness = np.array(thickness, dtype = np.int16)
        
        datas = {'patch': patch_resized,
                'label': label,
                'thickness': thickness}
        
        if return_data:
            return datas
        np.savez(save_path, **datas)
        
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: A tuple of 4 tensors (patch, label, threshold, index)
            patch (torch.Tensor): (1, D, H, W)
            label (torch.Tensor): (1, )
            threshold (torch.Tensor): (1, )
            index (torch.Tensor): (1, )
        """
        datas = self.cache.get(index, None)
        if datas == None:
            # Read data from disk
            if self.prepare_data_in_disk:
                cache = self.cache_paths[index]
                datas = np.load(cache)
            else: # Read raw data and process
                series_path = self.series_paths[index]
                annotation = self.annotations[index]
                label = self.labels[index]
                datas = self._prepare_data(series_path = series_path,
                                            annotation = annotation,
                                            label = label,
                                            crop_setting = self.crop_setting,
                                            save_path = '',
                                            return_data = True,
                                            crop_offset = self.crop_offset)
            patch = datas['patch']
            label = datas['label']
            # Calculate threshold
            thickness = datas['thickness']
            threshold = get_threshold_based_nodule_3d_thickness(thickness)
            threshold = np.array([threshold], np.float32)
            # Cache data into memory
            if self.cache_in_memory:
                self.cache[index] = [patch, label, threshold]
        else:
            patch, label, threshold = datas

        # Convert HU values to (0, 1400)
        patch = patch.astype(np.float32) / 1400
        
        if self.dataset_type == 'train':
            patch = self.augmentation(patch)
        
        patch = np.expand_dims(np.transpose(patch, (2, 0, 1)), axis = 0) # (H, W, D) -> (1, D, H, W)
        index = np.array([index], dtype = np.float32)
        
        # Convert to tensor
        patch = torch.from_numpy(patch.copy()).float()
        label = torch.from_numpy(label.copy()).float()
        threshold = torch.from_numpy(threshold.copy()).float()
        index = torch.from_numpy(index.copy()).float()
        
        return patch, label, threshold, index
    
    def augmentation(self, patch: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        patch = self._random_crop(patch)
        images = [patch]
        images = RandomFlipYXZ(0.3)(images)
        images = RandomRotation90(0.2)(images)
        images = RandomRotation(p = 0.2)(images, [False])
        
        patch = images[0]
        patch = random_color(patch, p = 0.2)
        patch = random_blur(patch, p = 0.1)
        patch = random_gauss_noise(patch, p = 0.1)
        patch = RandomCutout(0.2, img_size = self.final_shape)([patch])[0]
        
        return patch
    
    def _random_crop(self, patch: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Args:
            patch: npt.NDArray[np.float32]
                shpae = (H, W, D)
        Returns:
            patch: (npt.NDArray[np.float32])
                shpae = (H, W, D)
        """
        if self.crop_offset[0] != 0:
            start_p = [random.randrange(s) for s in self.crop_offset]
            final_y, final_x, final_z = self.final_shape
            start_y, start_x, start_z = start_p
            patch = patch[start_y : start_y + final_y,
                          start_x : start_x + final_x,
                          start_z : start_z + final_z
                          ].copy()
        return patch

    def __len__(self) -> int:
        return len(self.annotations)
    
    def get_alpha(self):
        return self.alpha