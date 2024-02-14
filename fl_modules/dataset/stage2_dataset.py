import os
import shutil
import psutil
import random
import logging
from collections import defaultdict
from multiprocessing.pool import Pool
from scipy import ndimage as nd
from typing import Tuple, Dict, List, Any, Union

import numpy as np
import numpy.typing as npt

import torch
from torch.utils.data import Dataset

from .utils import get_nodule_type, load_series_list

logger = logging.getLogger(__name__)
MIN_MEM_GB = 100

class RandomFlipYXZ:
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, images: List[torch.Tensor]) -> List[torch.Tensor]:
        flip_axes = []
        
        if len(images[0].shape) == 4: # (C, D, H, W)
            start_dim = 1
        else:
            start_dim = 0
        
        for i in range(start_dim, len(images[0].shape)):
            if random.random() < self.p:
                flip_axes.append(i)
        
        if len(flip_axes) != 0:
            for i in range(len(images)):
                if isinstance(images[i], torch.Tensor):
                    images[i] = torch.flip(images[i], flip_axes)
                else:
                    images[i] = np.flip(images[i], flip_axes)
       
        return images

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
    if isinstance(thickness, np.ndarray) and len(thickness.shape) == 1:
        thickness = thickness[0]
    thickness = int(thickness)
    # adaptive threshold for 3D model
    if thickness <= 3:
        return 0.5
    elif thickness >= 9:
        return 0.3
    else:
        return 18 / 30 - thickness / 30

class Stage2Dataset(Dataset):
    """
        Train/validation data generator
    """
    def __init__(self,
               dataset_type: str,
                nodule_size_ranges: Dict[str, Tuple[int, int]],
                num_nodules: Dict[str, int],
                series_list_path: str,
                crop_settings: dict,
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
        self.large_shape = crop_settings['large_shape']
        self.medium_shape = crop_settings['medium_shape']
        self.small_shape = crop_settings['small_shape']
        self.final_shape= crop_settings['final_shape']
        self.crop_settings = [self.large_shape, self.medium_shape, self.small_shape, self.final_shape]
        
        self.series_list_path = series_list_path
        self.num_nodule_in_dataset = num_nodules
        
        series_infos = load_series_list(self.series_list_path)
        self.num_patient = len(series_infos)
        
        # collect paths to resampled CT scan .npy file and corresponding annotation .txt file
        self.annotations = []
        self.series_paths = []
        self.nodule_3d_indices = []
        self.slice_is_nodule = []

        self.series_hardness = []
        self.series_cluster_label = []
        self.series_index_to_annotation_indices = defaultdict(list) # series_id -> [annotation_indices]
        self.series_index_to_nodule_3d_indices = defaultdict(list) # series_id -> [nodule_3d_indices]
            
        self.nodule_info_mapping = dict()
        num_tp_slices = 0
        total_num_slices = 0
        self.num_nodule_3d = 0
        
        for series_index, series_info in enumerate(series_infos):
            series_folder = series_info[0]
            file_name = series_info[1]
            npy_image_path = os.path.join(series_folder, 
                                        'npy', 
                                        f'{file_name}.npy')
            annotation_path = os.path.join(series_folder, 
                                        'stage1_post_process', 
                                        f'{file_name}.txt')
            
            with open(annotation_path, 'r') as f:
                nodule_3d_list = f.readlines()[1:]     

            for nodule_3d in nodule_3d_list:
                nodule_bboxs_2d = nodule_3d.split(' ')
                gt_nodule_index, gt_nodule_size, pred_nodule_size = [int(c) for c in nodule_bboxs_2d[0].split(',')]
                
                is_true_nodule = (gt_nodule_size > 0)
                self.nodule_info_mapping[self.num_nodule_3d] = [int(is_true_nodule), series_folder, npy_image_path, gt_nodule_index, gt_nodule_size, pred_nodule_size]
                self.series_index_to_nodule_3d_indices[series_index].append(self.num_nodule_3d)
                
                nodule_bboxs_2d = nodule_bboxs_2d[1:]
                nodule_thickness = len(nodule_bboxs_2d)
                for i, bbox_2d in enumerate(nodule_bboxs_2d):
                    bbox = [float(c) for c in bbox_2d.split(',')] # (slice_id, y_min, x_min, y_max, x_max, iou)
                    # Remove Iou
                    iou = bbox.pop(-1)
                    # Count number of true nodule
                    if (iou > 0.0) or (is_true_nodule and (not (i == 0 or i == len(nodule_bboxs_2d) - 1))):
                        num_tp_slices += 1
                        self.slice_is_nodule.append([1])
                    elif is_true_nodule: # If the slice is true nodule but in the first or last slice, we don't count it
                        continue
                    else:
                        self.slice_is_nodule.append([0])
                    
                    bbox.append(nodule_thickness) # (slice_id, y_min, x_min, y_max, x_max, nodule_thickness)
                    total_num_slices += 1
                    
                    self.annotations.append(bbox)
                    self.series_index_to_annotation_indices[series_index].append(len(self.annotations) - 1)
                    self.series_paths.append(npy_image_path)
                    self.nodule_3d_indices.append(self.num_nodule_3d)
                self.num_nodule_3d += 1
                
        self.nodule_3d_indices = np.array(self.nodule_3d_indices)
        self.alpha = 1 - (num_tp_slices / total_num_slices) # alpha for focal loss

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
                                gt_length_threshold = 2,
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

            nodule_hitting_map = set()
            for nodule_3d_id in self.nodule_info_mapping.keys():
                is_true_nodule, series_folder, npy_image_path, gt_nodule_index, gt_nodule_size, pred_nodule_size = self.nodule_info_mapping[nodule_3d_id]
                nodule_size = gt_nodule_size if is_true_nodule == 1 else pred_nodule_size
                
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

                metric_result[get_nodule_type(nodule_size, self.nodule_size_ranges)] += np.array([tp, fp, tn, fn], dtype=np.int32)

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

            if tp + fn + fp + tn <= 0:
                accuracy = 0
            else:
                accuracy = (tp + tn) / max(tp + fn + fp + tn, 1)

            metrics = {'recall': recall, 
                       'precision': precision, 
                       'f1_score': f1_score, 
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
            num_nodules_of_series: Dict[str, int]
                A dict of pair[series_folder, num_nodule]
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
            is_true_nodule, series_folder, npy_image_path, gt_nodule_index, gt_nodule_size, pred_nodule_size = self.nodule_info_mapping[nodule_3d_id]
            nodule_size = gt_nodule_size if is_true_nodule == 1 else pred_nodule_size
            
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
            tp, fp, tn, fn = metric_result.get(series_folder, [0, 0, 0, 0])
            fn = real_num_nodule - tp
            recall = tp / max(tp + fn, 1)
            precision = tp / max(tp + fp, 1)
            if recall + precision <= 0:
                f1_score = 0
            else:
                f1_score = 2 * ((recall * precision) / (recall + precision))

            if tp + fn + fp + tn <= 0:
                accuracy = 0
            else:
                accuracy = (tp + tn) / max(tp + fn + fp + tn, 1)
            metric_result[series_folder] = {'recall': recall, 
                                            'precision': precision, 
                                            'f1_score': f1_score, 
                                            'accuracy': accuracy,
                                            'tp': tp,
                                            'fp': fp,
                                            'fn': fn,
                                            'tn': tn}
    
        return metric_result

    def prepare_datas(self):
        logger.info("Start to prepare data!")
        pool = Pool(os.cpu_count() // 2)

        saving_folder = os.path.join(self.cache_path, self.dataset_type)
        if self.reset_datas:
            reset_folder(saving_folder)
        self.prepare_data_paths = []
        
        def error_when_running(error):
            print(error)
            
        try:
            for index in range(len(self.annotations)):
                series_path = self.series_paths[index]
                annotation = self.annotations[index]
                saving_path = os.path.join(saving_folder, '{}.npz'.format(index))

                self.prepare_data_paths.append(saving_path)
                if not self.reset_datas:
                    continue
                pool.apply_async(self._prepare_data,
                                 args = (series_path, annotation, self.crop_settings, saving_path, self.slice_is_nodule[index], False),
                                error_callback = error_when_running)
            pool.close()
            pool.join()
        finally:
            pool.terminate()
    
    @staticmethod
    def _prepare_data(series_path: str, annotation: tuple, crop_settings:tuple, saving_path: str, slice_is_nodule: list, return_data: bool):
        large_shape, medium_shape, small_shape, final_shape = crop_settings
        # get the series
        series = np.load(series_path, mmap_mode='c')
        # read the bbox information
        slice_number, y_min, x_min, y_max, x_max, thickness = annotation
        slice_number = int(slice_number)

        x_center = round(x_min + (x_max - x_min) / 2)
        y_center = round(y_min + (y_max - y_min) / 2)
        thickness = int(thickness)
        # Updated on 2023/03/24
        lower_index = slice_number - (large_shape[2] // 2)

        if lower_index < 0:
            lower_index = 0
        if lower_index + large_shape[2] > series.shape[2]:
            lower_index = series.shape[2] - large_shape[2]

        # get the patch of the nodule by cropping the image(y, x, channel)
        large_patch = series[int(max(y_center - (large_shape[0]/2), 0)) : int(min(y_center + (large_shape[0]/2), 512)), 
                            int(max(x_center - (large_shape[1]/2), 0)) : int(min(x_center + (large_shape[1]/2), 512)),
                            lower_index : lower_index+large_shape[2]]
        
        # if the patch is too small, add padding to it
        if np.any(large_patch.shape[0:2]!=(large_shape[0], large_shape[1])):
            large_patch = np.pad(large_patch, ((max(int(large_shape[0]/2)-y_center, 0), int(large_shape[0]/2) - min(series.shape[0]-y_center, int(large_shape[0]/2))),
                                            (max(int(large_shape[1]/2)-x_center, 0), int(large_shape[1]/2) - min(series.shape[1]-x_center, int(large_shape[1]/2))),
                                            (0, 0)), 
                                            mode='constant', 
                                            constant_values=-1024)
        
        # convert HU values (-1000, 400) to (0, 1)
        large_patch[large_patch<-1000] = -1000
        large_patch[large_patch>400] = 400
        large_patch = large_patch + 1000
            
        # crop patches of 3 different scales
        medium_patch = large_patch[round((large_shape[0]-medium_shape[0])/2) : -round((large_shape[0]-medium_shape[0])/2),
                                round((large_shape[1]-medium_shape[1])/2) : -round((large_shape[1]-medium_shape[1])/2),
                                round((large_shape[2]-medium_shape[2])/2) : -round((large_shape[2]-medium_shape[2])/2)]

        small_patch = large_patch[round((large_shape[0]-small_shape[0])/2) : -round((large_shape[0]-small_shape[0])/2),
                                round((large_shape[1]-small_shape[1])/2) : -round((large_shape[1]-small_shape[1])/2),
                                round((large_shape[2]-small_shape[2])/2) : -round((large_shape[2]-small_shape[2])/2)]
        
        # Updated on 2023/03/24
        # resize patches
        large_patch_resized = large_patch

        medium_patch_resized = nd.zoom(medium_patch, zoom=(final_shape[0]/medium_shape[0],
                                                        final_shape[1]/medium_shape[1],
                                                        final_shape[2]/medium_shape[2]), 
                                                    mode='nearest')

        small_patch_resized = nd.zoom(small_patch, zoom=(final_shape[0]/small_shape[0],
                                                        final_shape[1]/small_shape[1],
                                                        final_shape[2]/small_shape[2]), 
                                                mode='nearest')
        
        # Change dimension order from (H, W, D) to (1, D, H, W)
        large_patch_resized = np.transpose(large_patch_resized, (2, 0, 1))
        medium_patch_resized = np.transpose(medium_patch_resized, (2, 0, 1))
        small_patch_resized = np.transpose(small_patch_resized, (2, 0, 1))
        
        # add 1 dimension for 3D convolutions
        large_patch_resized = np.expand_dims(large_patch_resized, axis=0)
        medium_patch_resized = np.expand_dims(medium_patch_resized, axis=0)
        small_patch_resized = np.expand_dims(small_patch_resized, axis=0)

        # get the label of the nodule based on IoU calculated in Stage 1
        label = slice_is_nodule

        label = np.array(label, dtype=np.int16)
        thickness = np.array(thickness, dtype=np.int16)

        datas = {'large_patch_resized': large_patch_resized,
                'medium_patch_resized': medium_patch_resized,
                'small_patch_resized': small_patch_resized, 
                'label': label,
                'thickness': thickness}
        if return_data:
            return datas
        
        np.savez(saving_path, **datas)
        
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get train/validatation data.
        This function crops the dicom series to get 3 patches of different receptive field.
        
        Returns: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            large_patch_resized (torch.Tensor): (1, D, H, W)
            medium_patch_resized (torch.Tensor): (1, D, H, W)
            small_patch_resized (torch.Tensor): (1, D, H, W)
            label (torch.Tensor): (1, )
            threshold (torch.Tensor): (1, )
            index (torch.Tensor): (1, )
        """
        datas = self.cache.get(index, None)
        if datas == None:
            if self.prepare_data_in_disk: # Read data prepared in disk
                cache = self.prepare_data_paths[index]
                datas = np.load(cache)
            else: # Read raw data and process
                series_path = self.series_paths[index]
                annotation = self.annotations[index]
                datas = self._prepare_data(series_path = series_path,
                                            annotation = annotation,
                                            crop_settings = self.crop_settings,
                                            slice_is_nodule = self.slice_is_nodule[index],
                                            saving_path = '',
                                            return_data = True)
            large_patch_resized = datas['large_patch_resized']
            medium_patch_resized = datas['medium_patch_resized']
            small_patch_resized = datas['small_patch_resized']
            label = datas['label']
            # Calculate threshold
            thickness = datas['thickness']
            threshold = get_threshold_based_nodule_3d_thickness(thickness)
            threshold = np.array([threshold], np.float32)
            # Cache data into memory
            if self.cache_in_memory:
                self.cache[index] = [large_patch_resized, medium_patch_resized, small_patch_resized, label, threshold]
        else:
            large_patch_resized, medium_patch_resized, small_patch_resized, label, threshold = datas

        if self.dataset_type == 'train':
            large_patch_resized, medium_patch_resized, small_patch_resized = self.augmentation(large_patch_resized, medium_patch_resized, small_patch_resized)
        
        large_patch_resized = large_patch_resized.astype(np.float32) / 1400
        medium_patch_resized = medium_patch_resized.astype(np.float32) / 1400
        small_patch_resized = small_patch_resized.astype(np.float32) / 1400
        index = np.array([index], dtype=np.float32)
        
        # Convert to tensor
        large_patch_resized = torch.from_numpy(large_patch_resized).float()
        medium_patch_resized = torch.from_numpy(medium_patch_resized).float()
        small_patch_resized = torch.from_numpy(small_patch_resized).float()
        label = torch.from_numpy(label).float()
        threshold = torch.from_numpy(threshold).float()        
        index = torch.from_numpy(index).float()
        
        return [large_patch_resized, medium_patch_resized, small_patch_resized], label, threshold, index

    def augmentation(self, large_patch_resized, medium_patch_resized, small_patch_resized):
        images = [large_patch_resized, medium_patch_resized, small_patch_resized]
        images = RandomFlipYXZ(0.3)(images)
        return images
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def get_alpha(self):
        return self.alpha