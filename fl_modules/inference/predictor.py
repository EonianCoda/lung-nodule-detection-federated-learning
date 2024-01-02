from typing import Tuple, Dict, Union, List, Any
import numpy as np
import numpy.typing as npt
from multiprocessing.pool import Pool
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# openfl system package
from .utils import normalize_paths, compute_recall, compute_precision, compute_f1_score
from .utils import load_model, load_gt_mask_maps, load_series_images
from .utils import get_3d_connected_componment, generate_bboxes_of_one_nodule, compute_bbox3d_intersection_volume, get_3d_connected_componment_after_lobe, expand_neighboring_bbox2d
from .parallel_utils import PrefetchSeries
from .parallel_utils import AvgMethodProcessPrediction, MaxMethodProcessPrediction
from .lobe3d_segmentation import prepare_lobe_of_series, get_first_and_last_slice_of_lobe
import logging

logger = logging.getLogger(__name__)
PRINT_INTERVAL_STEPS = 100

def generate_metric_of_nodule(tp: int, fp: int, fn: int) -> Dict[str, Union[int, float]]:
    recall = compute_recall(tp, fn)
    precision = compute_precision(tp, fp)
    f1_score =  compute_f1_score(recall, precision)
    
    return {'recall': recall,
            'precision': precision,
            'f1_score': f1_score,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': 0}

class MetricStage1(object):
    def __init__(self, nodule_size_ranges: dict):
        self.nodule_size_ranges = list(nodule_size_ranges.keys()) + ['all']
        self.reset()
    
    def reset(self) -> None: 
        self.tp_fp_fn_of_series = dict()
            
    def get_whole_result(self) -> Dict[str, Dict[str, Union[int, float]]]:
        """
        Return: Dict[str, Union[int, float]]]
            A dict of [nodule_type, metrics], metrics is a dict of (<metric>, value), <metric> can be (recall, precision, f1_score, TP, FP, FN, TN)
        """
        if len(self.tp_fp_fn_of_series) == 0:
            return None
        rs = dict()
        tp_fn_fn = np.sum(list(self.tp_fp_fn_of_series.values()), axis=0)
        for i, nodule_type in enumerate(self.nodule_size_ranges):
            tp, fp, fn = tp_fn_fn[i]
            rs[nodule_type] = generate_metric_of_nodule(tp, fp, fn)
        return rs

    def get_each_task_result(self) -> Dict[int, Dict[str, Dict[str, Union[int, float]]]]:
        """
        Return: Dict[int, Dict[str, Dict[str, Union[int, float]]]]
            A dict of dict[task_i, dict[nodule_type, metrics]], metrics is a dict of (<metric>, value), <metric> can be (recall, precision, f1_score, TP, FP, FN, TN)
        """
        if len(self.tp_fp_fn_of_series) == 0:
            return None
        total_rs = dict()
        for task_i in self.tp_fp_fn_of_series.keys():
            rs = dict()
            for i, nodule_type in enumerate(self.nodule_size_ranges):
                tp, fp, fn = self.tp_fp_fn_of_series[task_i][i]
                rs[nodule_type] = generate_metric_of_nodule(tp, fp, fn)
            total_rs[task_i] = rs
        return total_rs

    def update(self, task_i: int, rs: dict) -> None:
        tp_fp_fn = np.zeros((len(self.nodule_size_ranges), 3), dtype = np.int32)
        for nodule_type, v in rs.items():
            idx = self.nodule_size_ranges.index(nodule_type)
            tp, fp, fn = v['tp'], v['fp'], v['fn']
            tp_fp_fn[idx] = [tp, fp, fn]
        
        self.tp_fp_fn_of_series[task_i] = tp_fp_fn

class SeriesSlicesGenerator:
    def __init__(self,
                series_image: Union[torch.Tensor, npt.NDArray[np.float32]], 
                device: torch.device = torch.device('cpu'),
                depth: int = 32,
                first_slice_of_lobe: int = 0,
                last_slice_of_lobe: int = -1,
                reverse: bool = False):
        """
        Args:
            series_image: torch.Tensor or np.ndarray
                A image series data with shape (D, H, W)
            last_slice_of_lobe: int
                the last slice number of lobe segmentation, default = -1
        """
        # Ensure the data type of series_image is torch.Tensor
        if not isinstance(series_image, torch.Tensor):
            series_image = torch.from_numpy(series_image).float()
            
        self.device = device
        series_image = series_image.to(self.device, non_blocking=True)
            
        # Generate all pairs of series(start_sliceID, end_sliceID)
        self.slice_id_pairs = []
        stride = depth // 2
        num_slice = series_image.shape[-1]
        
        # If not giving last_slice_of_lobe, then set it same as num_slice
        if last_slice_of_lobe == -1:
            last_slice_of_lobe = num_slice
        
        self.flip_depth = reverse
        # Flip depth dimension
        if self.flip_depth == True:
            series_image = series_image[::-1, :, :]
            start_slice_id = num_slice - last_slice_of_lobe
            last_slice_of_lobe = num_slice - first_slice_of_lobe
        else:
            start_slice_id = first_slice_of_lobe
        
        self.series_data = torch.unsqueeze(torch.unsqueeze(series_image, 0), 0) # (D, H, W) -> (1, 1, D, H, W)
        end_slice_id = 0
        while True:
            end_slice_id = start_slice_id + depth
            if end_slice_id >= last_slice_of_lobe:
                self.slice_id_pairs.append((last_slice_of_lobe - depth, last_slice_of_lobe))
                break
            self.slice_id_pairs.append((start_slice_id, end_slice_id))
            start_slice_id += stride

    def __len__(self):
        return len(self.slice_id_pairs)

    def __iter__(self) -> Tuple[int, torch.Tensor]:
        def swap_last_middle(m):
            tmp = m[-1]
            middle_idx = len(m) // 2
            m[-1] = m[middle_idx]
            m[middle_idx] = tmp

        # Re-order(important!!)
        # We use the parallel technique to speed up processing the result of prediction(e.g, average method or maximum method).
        # To prevent revising the same part of result simultaneously, we use thread lock tricks.If we yield slices image in order,
        # it will spend extra time to wait until another parallel process finish their task because one slices images has some duplicate 
        # part comparing with its previous slices image or its next slices image (e.g, when the stride is 10, slices images(20 ~ 40) 
        # and its next slices images(30 ~ 50) has duplicate part in slice(30~40)). To avoid it, we reorder to ensure that one slices images 
        # does not have duplicate part comparing with its previous slices image or its next slices image.
        odds = list(range(0, len(self), 2))
        evens = list(range(1, len(self), 2))
        swap_last_middle(odds)
        if len(evens) > 0:
            swap_last_middle(evens)
            reordered_indices = odds + evens
        else:
            reordered_indices = odds
            
        for index in reordered_indices:
            start_slice_id, end_slice_id = self.slice_id_pairs[index]
            yield start_slice_id, self.series_data[:, :, start_slice_id : end_slice_id, ...] # (1, 1, self.depth, H, W)

class PredictorStage1(object):
    def __init__(self, 
                model: Union[nn.Module, str], 
                model_input_shape: tuple = (512, 512, 32, 1),
                device: torch.device = torch.device('cpu'),
                inference_mode: str = 'avg',
                foreground_threshold: float = 0.3,
                iou_threshold: float = 0.01,
                nodule_3d_minimum_size: int = 5,
                nodule_3d_minimum_thickness: int = 3,
                combined_offset: Tuple[npt.NDArray[np.int32], List[int]] = [],
                nodule_size_ranges: Dict[str, Tuple[int, int]] = dict(),
                use_lobe: bool = True,
                iou_mode: str = 'pixel',
                log_metrics: bool = True,
                augmented_inference: bool = False):
        """Predictor of Stage1

        Load the trained model and setup some inference configurations for stage1
        Args:
            model: str or nn.Module
                The path to model or a model entity.If it is the path of model, then read it.
            model_input_shape: tuple
                The shape of input image for model
            inference_mode: str
                There are two method 'avg' or 'max'.
            foreground_threshold: float
                The threshold to decide foreground in the result of prediction, probability > threshold => foreground
            iou_threshold: float, default=0.01
                A threshold of deciding a nodule of prediction is true positive or false positive
        """
        assert(foreground_threshold <= 1 and foreground_threshold >= 0)
        assert(iou_threshold <= 1 and iou_threshold > 0)

        if len(nodule_size_ranges) == 0:
            self.nodule_size_ranges = {'benign': [0, 52],
                                        'probably_benign': [52, 176], 
                                        'probably_suspicious': [176, 418], 
                                        'suspicious': [418, -1]}
        else:
            self.nodule_size_ranges = nodule_size_ranges
        
        if len(combined_offset) == 0:
            self.combined_offset = np.array([[-5, -5, -3], [5, 5, 3]])
        else:
            self.combined_offset = combined_offset
        
        self.device = device    
        # initialize model
        self.model = load_model(1, model, device)
        self.model.eval()
        self.model_input_shape = model_input_shape[:-1] # ignore the dimension only for conv3d
        self.depth = self.model_input_shape[-1]
        self.stride = self.depth // 2

        self.foreground_threshold = foreground_threshold
        self.iou_threshold = iou_threshold

        self.inference_mode = inference_mode.lower()
        if self.inference_mode != 'avg' and self.inference_mode != 'max':
            raise ValueError(f"Inference model {self.inference_mode} does not exist!!")

        self.nodule_3d_minimum_size = nodule_3d_minimum_size
        self.nodule_3d_minimum_thickness = nodule_3d_minimum_thickness
        self.use_lobe = use_lobe
        self.augmented_inference = augmented_inference
        self.iou_mode = iou_mode
        self.log_metrics = log_metrics
        
        self.num_workers = max(os.cpu_count() // 4, 2)
        
    def get_bboxes_of_nodules_in_series(self, series_path: str) -> List[List[Tuple[int, int, int, int, int]]]:
        """
        Return: list
            A list of 3D nodules, each 3D nodules is a list of 2D bbox formatted by a tuple(z, y_min, x_min, y_max, x_max).
        """
        if self.use_lobe:
            lobe_paths, first_and_last_slice_of_lobe_paths = prepare_lobe_of_series(series_path)
            lobe_path = lobe_paths[0]
            first_slice_of_lobe, last_slice_of_lobe = get_first_and_last_slice_of_lobe(first_and_last_slice_of_lobe_paths[0])
        else:
            lobe_path = ''
            first_slice_of_lobe, last_slice_of_lobe = 0, -1

        # Predict
        series_image = load_series_images(series_path) 
        pred_binary_mask_maps = self.predict_nodule_in_series(series_image, first_slice_of_lobe, last_slice_of_lobe)
        labels, valid_component_indices, valid_nodule_sizes, bboxes = get_3d_connected_componment_after_lobe(pred_binary_mask_maps = pred_binary_mask_maps, 
                                                                                                             lobe_path = lobe_path, 
                                                                                                             nodule_3d_minimum_size = self.nodule_3d_minimum_size, 
                                                                                                             nodule_3d_minimum_thickness = self.nodule_3d_minimum_thickness, 
                                                                                                             combined_offset = self.combined_offset)
        
        result_of_3d_nodules = [] 
        for component_id, bbox in zip(valid_component_indices, bboxes):
            # Formatted by [[z, y_min, x_min, y_max, x_max], [z, y_min, x_min, y_max, x_max] ...]
            result_of_one_3d_nodule = generate_bboxes_of_one_nodule(bbox, labels, component_id) 
            if len(result_of_one_3d_nodule) == 0:
                continue
            result_of_3d_nodules.append(result_of_one_3d_nodule)

        return result_of_3d_nodules

    def get_bboxes_ious_of_nodules_in_series(self, 
                                            series_paths: Union[str, List[str]],
                                            gt_mask_maps_paths: Union[str, List[str]],
                                            pred_mask_save_paths: Union[str, List[str]] = None,
                                            ) -> List[List[Tuple[int, int, int, int, int, float]]]:
        """
        Return: list
            A list of 3D nodules, each 3D nodules is a list of 2D bbox formatted by a tuple(z, y_min, x_min, y_max, x_max, iou).
        """
        series_paths = normalize_paths(series_paths)
        gt_mask_maps_paths = normalize_paths(gt_mask_maps_paths)
        if pred_mask_save_paths is not None or len(pred_mask_save_paths) != 0:
            pred_mask_save_paths = normalize_paths(pred_mask_save_paths)
            assert(len(series_paths) == len(pred_mask_save_paths))
        if len(series_paths) != len(gt_mask_maps_paths):
            raise ValueError("Number of element in series_paths and gt_mask_maps_paths not equal!")
        
        # Prepare lobe_paths and first_and_last_slice_of_lobe_paths
        if self.use_lobe:
            lobe_paths, first_and_last_slice_of_lobe_paths = prepare_lobe_of_series(series_paths)

        bboxes_ious_dict = dict()
        def collect_result(result):
            task_i, bboxes_ious = result
            bboxes_ious_dict[task_i] = bboxes_ious

        def error_when_running(error):
            print(error)

        prefetch_dataset = PrefetchSeries(series_paths)
        pool = Pool(self.num_workers)
        try:
            for task_i, series_image in enumerate(prefetch_dataset):
                gt_mask_maps_path = gt_mask_maps_paths[task_i]
                pred_mask_save_path = pred_mask_save_paths[task_i] if pred_mask_save_paths is not None else None
                if self.use_lobe:
                    lobe_path = lobe_paths[task_i]
                    first_slice_of_lobe, last_slice_of_lobe = get_first_and_last_slice_of_lobe(first_and_last_slice_of_lobe_paths[task_i])
                else:
                    lobe_path = ''
                    first_slice_of_lobe, last_slice_of_lobe = 0, -1
                    
                pred_mask_maps = self.predict_nodule_in_series(series_image, first_slice_of_lobe, last_slice_of_lobe)
                pool.apply_async(self._cal_bboxes_ious_of_nodules_in_series, 
                                args = (pred_mask_maps, gt_mask_maps_path, pred_mask_save_path, self.nodule_3d_minimum_size, self.nodule_3d_minimum_thickness, lobe_path, self.combined_offset, task_i), 
                                callback = collect_result,
                                error_callback = error_when_running)
                if self.log_metrics and ((task_i + 1) % PRINT_INTERVAL_STEPS == 0 or task_i == len(prefetch_dataset) - 1):
                    logger.info("{}/{}: Calculate the bbox and ious for '{}'".format(task_i + 1, 
                                                                                    len(prefetch_dataset), 
                                                                                    os.path.splitext(os.path.basename(gt_mask_maps_path))[0]))
            pool.close()
            pool.join()
        finally:
            pool.terminate()
        
        results = []
        for key in sorted(bboxes_ious_dict.keys()):
            results.append(bboxes_ious_dict[key])

        return results
    
    @staticmethod
    def _cal_bboxes_ious_of_nodules_in_series(pred_binary_mask_maps: npt.NDArray[np.uint8],
                                                gt_mask_maps_path: str,
                                                pred_mask_save_path: str,
                                                nodule_3d_minimum_size: int,
                                                nodule_3d_minimum_thickness: int,
                                                lobe_path: str,
                                                combined_offset: Tuple[npt.NDArray[np.int32], List[int]] = None,
                                                task_i: int = None,
                                                ) -> Union[List[List[Any]], Tuple[int, List[List[Any]]]]:
        """Calculate bbox and iou of nodules in series between prediction and groud truth
        
        Return: Tuple[int, List[List[Any]] or List[List[Any]
            Option1: Tuple[int, List[List[Any]]
                First int mean 'task_i', if argument 'task_i' is passed to this function, then it will return same 'task_i'. Second list contains bbox and ious of multiple 
                nodules, formatted by [[gt_nodule_index, gt_nodule_size, pred_nodule_size], [z, y_min, x_min, y_max, x_max, iou], [z, y_min, x_min, y_max, x_max, iou] ...].
            Option2: List[List[Any]]:
                same as option1's second returned value
        """
        result_of_3d_nodules = [] 
        
        # Apply lobe on prediction and find nodule in prediction
        pred_labels, pred_valid_component_indices, pred_valid_nodule_sizes, pred_bboxes = get_3d_connected_componment_after_lobe(pred_binary_mask_maps, lobe_path, nodule_3d_minimum_size, nodule_3d_minimum_thickness, combined_offset)
        if pred_mask_save_path is not None:
            binary_pred_mask = np.zeros_like(pred_labels, dtype = np.uint8)
            for valid_idx in pred_valid_component_indices:
                binary_pred_mask[pred_labels == valid_idx] = 1
            np.savez_compressed(pred_mask_save_path, image = binary_pred_mask)
        # Not found any nodule in prediction
        if len(pred_valid_component_indices) == 0:
            if task_i != None:
                return task_i, result_of_3d_nodules
            else:
                return result_of_3d_nodules

        # Find nodule in groud truth
        gt_mask_maps = load_gt_mask_maps(gt_mask_maps_path)
        gt_labels, gt_valid_component_indices, gt_valid_nodule_sizes, gt_bboxes = get_3d_connected_componment(gt_mask_maps, nodule_3d_minimum_size, nodule_3d_minimum_thickness)
        
        # No nodule in groud truth
        if len(gt_valid_component_indices) == 0:
            for pred_component_id, pred_nodule_size, pred_bbox in zip(pred_valid_component_indices, pred_valid_nodule_sizes, pred_bboxes):
                result_of_one_3d_nodule = generate_bboxes_of_one_nodule(pred_bbox, pred_labels, pred_component_id)
                for i, result_2d in enumerate(result_of_one_3d_nodule):
                    result_2d.append(0.0)
                    result_of_one_3d_nodule[i] = result_2d
                # Add nodule size at beginning
                # [[z, y_min, x_min, y_max, x_max, iou], [z, y_min, x_min, y_max, x_max, iou] ...] => [[gt_nodule_index, gt_nodule_size, pred_nodule_size], [z, y_min, x_min, y_max, x_max, iou], [z, y_min, x_min, y_max, x_max, iou] ...]
                result_of_one_3d_nodule = [[-1, 0, pred_nodule_size]] + result_of_one_3d_nodule # -1 mean that there are not any matching groud truth nodule
                result_of_3d_nodules.append(result_of_one_3d_nodule)
            if task_i != None:
                return task_i, result_of_3d_nodules
            else:
                return result_of_3d_nodules
            
        bbox3d_intersection_volumes = compute_bbox3d_intersection_volume(pred_bboxes, gt_bboxes) # shape = (num of pred, num of gt)
        for pred_i, (pred_component_id, pred_nodule_size, pred_bbox) in enumerate(zip(pred_valid_component_indices, pred_valid_nodule_sizes, pred_bboxes)):
            # Get the indices of ground truth, which intersect with current prediction
            pred_z_min, pred_z_max = pred_bbox[0][2], pred_bbox[1][2]
            result_of_one_3d_nodule = generate_bboxes_of_one_nodule(pred_bbox, pred_labels, pred_component_id)
            ious = []
            ious_along_z = []
            # Compute IOUs
            for gt_component_id, inter_area, gt_bbox in zip(gt_valid_component_indices, bbox3d_intersection_volumes[pred_i], gt_bboxes):
                if inter_area <= 0.0:
                    ious_along_z.append([])
                    ious.append(0.0)
                    continue

                gt_z_min, gt_z_max = gt_bbox[0][2], gt_bbox[1][2]
                z_min = min(pred_z_min, gt_z_min)
                z_max = max(pred_z_max, gt_z_max)
                # Generate binary mask for prediction and groud truth label 
                pred_label = (pred_labels[..., z_min: z_max] == pred_component_id)
                gt_label = (gt_labels[..., z_min: z_max] == gt_component_id)

                # Compute intersection area(pixel-level)
                matching_label = pred_label & gt_label
                voxel_intersection_along_z = np.count_nonzero(matching_label, axis=(0,1))
                gt_voxels_along_z = np.count_nonzero(gt_label, axis=(0,1))
                pred_voxels_along_z = np.count_nonzero(pred_label, axis=(0,1))
                voxel_intersection = np.sum(voxel_intersection_along_z)
                gt_voxels = np.sum(gt_voxels_along_z)
                pred_voxels = np.sum(pred_voxels_along_z)
                
                # Compute iou along z-axis and iou on all mask map
                union_volumes_along_z = (gt_voxels_along_z + pred_voxels_along_z - voxel_intersection_along_z)
                iou_along_z = np.divide(voxel_intersection_along_z, union_volumes_along_z, out = np.zeros_like(voxel_intersection_along_z, dtype=np.float32), where=union_volumes_along_z != 0)
                ious_along_z.append(iou_along_z)
                iou = voxel_intersection / (gt_voxels + pred_voxels - voxel_intersection)
                ious.append(iou)
            
            # False Positive
            if np.max(ious) == 0.0:
                ious = [0.0 for _ in range(len(result_of_one_3d_nodule))]
                gt_nodule_size = 0
                gt_i = -1
            # True Positive
            else:
                # Get the index of groud truth, which matchs current index of prediction.
                gt_i = np.argmax(ious)
                gt_component_id = gt_valid_component_indices[gt_i]
                gt_bbox = gt_bboxes[gt_i]
                gt_z_min = gt_bbox[0][2]
                z_min = min(pred_z_min, gt_z_min)

                iou_z_start = pred_z_min - z_min
                iou_z_end = iou_z_start + (pred_z_max - pred_z_min + 1)
                ious = ious_along_z[gt_i][iou_z_start: iou_z_end]

                gt_nodule_size = gt_valid_nodule_sizes[gt_i]
            # Append iou in each bbox, 
            # [[z, y_min, x_min, y_max, x_max], [z, y_min, x_min, y_max, x_max] ...] => [[z, y_min, x_min, y_max, x_max, iou], [z, y_min, x_min, y_max, x_max, iou] ...]
            for i, (result_2d, iou) in enumerate(zip(result_of_one_3d_nodule, ious)):
                result_2d.append(iou)
                result_of_one_3d_nodule[i] = result_2d
            # Add nodule size at the beginning
            # [[z, y_min, x_min, y_max, x_max, iou], [z, y_min, x_min, y_max, x_max, iou] ...] => [[gt_nodule_index, gt_nodule_size, pred_nodule_size], [z, y_min, x_min, y_max, x_max, iou], [z, y_min, x_min, y_max, x_max, iou] ...]
            result_of_one_3d_nodule = [[gt_i, gt_nodule_size, pred_nodule_size]] + result_of_one_3d_nodule
            result_of_3d_nodules.append(result_of_one_3d_nodule)

        if task_i != None:
            return task_i, result_of_3d_nodules
        else:
            return result_of_3d_nodules

    def get_recall_precision_of_nodules_in_series(self, 
                                                series_paths: Union[str, List[str]],
                                                gt_mask_maps_paths: Union[str, List[str]],
                                                return_each_series: bool = False) -> Union[Dict[str, Dict[str, Union[int, float]]], Dict[int, Dict[str, Dict[str, Union[int, float]]]]]:
        """Calculate recall and precision of nodules in multiple series between prediction and ground truth.

        Args:
            series_paths: list of str
                A list contains multiple paths to image of series. 
            gt_mask_maps_paths: list of str
                A list contains multiple paths to mask maps image of groud truth
            return_each_series: bool
                A bool flag for whether returning metric for each given series or not
        Return: Dict[str, Dict[str, Union[int, float]]]
            A dict of [nodule_type, metrics], metrics is a dict of (<metric>, value), <metric> can be (recall, precision, f1_score, TP, FP, FN, TN)
        """
        series_paths = normalize_paths(series_paths)
        gt_mask_maps_paths = normalize_paths(gt_mask_maps_paths)
        if len(series_paths) != len(gt_mask_maps_paths):
            raise ValueError("Number of element in series_paths and gt_mask_maps_paths not equal!")
        
        # Prepare lobe_paths and first_and_last_slice_of_lobe_paths
        if self.use_lobe:
            lobe_paths, first_and_last_slice_of_lobe_paths = prepare_lobe_of_series(series_paths)

        metric_accumulator = MetricStage1(self.nodule_size_ranges)
        def collect_result(rs: tuple):
            metric_accumulator.update(*rs)

        def error_when_running(error):
            print(error)

        sorted_key_of_different_type = list(sorted(self.nodule_size_ranges, key=lambda k: self.nodule_size_ranges[k]))
        sorted_key_of_different_type.append('all')
        pool = Pool(self.num_workers)
        prefetch_dataset = PrefetchSeries(series_paths)
        prefetch_dataloader = DataLoader(prefetch_dataset, batch_size=1, num_workers=1, prefetch_factor=1, pin_memory=True)
        try:
            for task_i, series_image in enumerate(prefetch_dataloader):
                series_image = series_image[0]
                gt_mask_maps_path = gt_mask_maps_paths[task_i]
                if self.use_lobe:
                    lobe_path = lobe_paths[task_i]
                    first_slice_of_lobe, last_slice_of_lobe = get_first_and_last_slice_of_lobe(first_and_last_slice_of_lobe_paths[task_i])
                else:
                    lobe_path = ''
                    first_slice_of_lobe, last_slice_of_lobe = 0, -1

                pred_mask_maps = self.predict_nodule_in_series(series_image, first_slice_of_lobe, last_slice_of_lobe)
                pool.apply_async(self._cal_recall_precision_of_nodules_in_series, 
                                args = (pred_mask_maps, gt_mask_maps_path, self.iou_threshold, self.nodule_3d_minimum_size, self.nodule_3d_minimum_thickness, self.nodule_size_ranges, lobe_path, self.combined_offset, self.iou_mode, task_i),
                                callback = collect_result,
                                error_callback = error_when_running)
                
                if self.log_metrics and ((task_i + 1) % PRINT_INTERVAL_STEPS == 0 or task_i == len(prefetch_dataset) - 1):
                    nodule_result = metric_accumulator.get_whole_result()
                    if nodule_result == None:
                        continue
                    template = '{:20s}: Recall={:.3f}, Precision={:.3f}, F1={:.3f}, TP={:4d}, FP={:4d}, FN={:4d}'
                    logger.info("{}/{}".format(task_i + 1, len(prefetch_dataset)))
                    for nodule_type in sorted_key_of_different_type:
                        rs = nodule_result[nodule_type]
                        logger.info(template.format(nodule_type, rs['recall'], rs['precision'], rs['f1_score'], rs['tp'], rs['fp'], rs['fn']))
            pool.close()
            pool.join()
        finally:
            pool.terminate()
            
        if return_each_series == True:
            return metric_accumulator.get_each_task_result()
        else:
            return metric_accumulator.get_whole_result()

    @staticmethod
    def _cal_recall_precision_of_nodules_in_series(pred_binary_mask_maps: npt.NDArray[np.uint8],
                                                    gt_mask_maps_path: str,
                                                    iou_threshold: float,
                                                    nodule_3d_minimum_size: int,
                                                    nodule_3d_minimum_thickness: int,
                                                    nodule_size_ranges: dict,
                                                    lobe_path: str,
                                                    combined_offset: Tuple[npt.NDArray[np.int32], List[int]] = None,
                                                    iou_mode: str = 'bbox',
                                                    task_i: int = None
                                                    ) -> Union[Dict[str, Dict[str, Union[int, float]]], Tuple[int, Dict[str, Dict[str, Union[int, float]]]]]:
        """Calculate recall and precision of nodules in one series between prediction and ground truth.

        Return: Dict[str, Dict[str, Union[int, float]]]
            A dict of [nodule_type, metrics], metrics is a dict of (<metric>, value), <metric> can be ('recall', 'precision', 'f1_score', 'TP', 'FP', 'FN', 'TN')
        """
        
        # Apply lobe on prediction and find nodule in prediction
        pred_labels, pred_valid_component_indices, pred_valid_nodule_sizes, pred_bboxes = get_3d_connected_componment_after_lobe(pred_binary_mask_maps, lobe_path, nodule_3d_minimum_size, nodule_3d_minimum_thickness, combined_offset)
        num_pred = len(pred_valid_component_indices)
        
        # Find nodule in groud truth
        gt_mask_maps = load_gt_mask_maps(gt_mask_maps_path)
        gt_labels, gt_valid_component_indices, gt_valid_nodule_sizes, gt_bboxes = get_3d_connected_componment(gt_mask_maps, nodule_3d_minimum_size, nodule_3d_minimum_thickness)
        num_gt = len(gt_valid_component_indices)

        # No nodule in prediction or in groud truth
        if num_pred == 0 or num_gt == 0:
            metric_result = dict()
            for nodule_type, size_range in nodule_size_ranges.items():
                lower_bound, upper_bound = size_range
                if upper_bound == -1:
                    upper_bound = max(gt_valid_nodule_sizes) if num_gt != 0 else 0
                    if num_pred != 0:
                        upper_bound = max(upper_bound, max(pred_valid_nodule_sizes))
                    upper_bound = upper_bound + 1

                tp, fp, fn  = 0, 0, 0
                if num_pred != 0:
                    fp = np.count_nonzero((pred_valid_nodule_sizes > lower_bound) & (pred_valid_nodule_sizes <= upper_bound))
                if num_gt != 0:
                    fn = np.count_nonzero((gt_valid_nodule_sizes > lower_bound) & (gt_valid_nodule_sizes <= upper_bound))

                metric_result[nodule_type] = generate_metric_of_nodule(tp, fp, fn)
            # Compute metric include all nodule types
            tp, fp, fn = 0, num_pred, num_gt
            metric_result['all'] = generate_metric_of_nodule(tp, fp, fn)
            # For multiprocessing, collect results
            if task_i != None:
                return task_i, metric_result
            else:
                return metric_result
    
        bbox3d_intersection_volumes = compute_bbox3d_intersection_volume(pred_bboxes, gt_bboxes) # shape = (num_pred, num_gt)
        # Compute IOUs
        ious = np.zeros((num_pred, num_gt), dtype=np.float32)
        for pred_i, pred_component_id in enumerate(pred_valid_component_indices):
            pred_bbox = pred_bboxes[pred_i]
            pred_z_min, pred_z_max = pred_bbox[0][2], pred_bbox[1][2]
            for gt_i, intersection_volume in enumerate(bbox3d_intersection_volumes[pred_i]):
                if intersection_volume <= 0.0:
                    continue
                if iou_mode == 'bbox':
                    gt_bbox = gt_bboxes[gt_i]
                    pred_bbox3d_volume = np.prod(pred_bbox[1] - pred_bbox[0])
                    gt_bbox3d_volume = np.prod(gt_bbox[1] - gt_bbox[0])
                    
                    iou = intersection_volume / (gt_bbox3d_volume + pred_bbox3d_volume - intersection_volume)
                else: # pixel mode
                    gt_component_id = gt_valid_component_indices[gt_i]
                    gt_bbox = gt_bboxes[gt_i]

                    gt_z_min, gt_z_max = gt_bbox[0][2], gt_bbox[1][2]
                    z_min = min(pred_z_min, gt_z_min)
                    z_max = max(pred_z_max, gt_z_max)

                    # Generate binary mask for prediction and groud truth label 
                    pred_label = (pred_labels[..., z_min: z_max] == pred_component_id)
                    gt_label = (gt_labels[..., z_min: z_max] == gt_component_id)

                    # Intersection area
                    matching_label = pred_label & gt_label
                    voxel_intersection = np.count_nonzero(matching_label)
                    gt_voxels = np.count_nonzero(gt_label)
                    pred_voxels = np.count_nonzero(pred_label)

                    iou = voxel_intersection / (gt_voxels + pred_voxels - voxel_intersection)
                
                ious[pred_i][gt_i] = iou
                    
        gt_ious = np.max(ious, axis=0)
        pred_ious = np.max(ious, axis=1)

        # Calculate recall and precision for different type of nodule
        result = dict()
        for nodule_type, size_range in nodule_size_ranges.items():
            lower_bound, upper_bound = size_range
            if upper_bound == -1:
                upper_bound = max(max(gt_valid_nodule_sizes), max(pred_valid_nodule_sizes)) + 1
            gt_ious_of_nodule_type = gt_ious[(gt_valid_nodule_sizes > lower_bound) & (gt_valid_nodule_sizes <= upper_bound)]
            pred_ious_of_nodule_type = pred_ious[(pred_valid_nodule_sizes > lower_bound) & (pred_valid_nodule_sizes <= upper_bound)]

            tp = np.count_nonzero(gt_ious_of_nodule_type > iou_threshold)
            fn = len(gt_ious_of_nodule_type) - tp
            fp = np.count_nonzero(pred_ious_of_nodule_type <= iou_threshold)
            result[nodule_type] = generate_metric_of_nodule(tp, fp, fn)

        # Compute metric include all nodule types
        tp = np.count_nonzero(gt_ious > iou_threshold)
        fn = num_gt - tp
        fp = np.count_nonzero(pred_ious <= iou_threshold)
        result['all'] = generate_metric_of_nodule(tp, fp, fn)
        
        if task_i != None:
            return task_i, result
        else:
            return result
                     
    def predict_nodule_in_series(self, series_image: Union[torch.Tensor, npt.NDArray[np.float32]], first_slice_of_lobe = 0, last_slice_of_lobe = -1) -> npt.NDArray[np.uint8]:
        """Inference on series to get the mask maps of series

        Args:
            series_image: torch.Tensor or npt.NDArray[np.float32]
                the series image
            last_slice_of_lobe: int
                for speeding up the process of inference
        Return: npt.NDArray[np.uint8]
            A binary mask maps of nodule, shape is same as given argument 'series_image'
        """
        # initialize the processor of prediction
        method = AvgMethodProcessPrediction if self.inference_mode == 'avg' else MaxMethodProcessPrediction
        prediction_processor = method(self.model_input_shape, series_image.shape, self.foreground_threshold)

        # do model predict
        for start_slice_id, series_part_data in SeriesSlicesGenerator(series_image, self.device, self.depth, first_slice_of_lobe, last_slice_of_lobe):
            pred_mask_maps = self.predict_nodule_in_slices(series_part_data)
            prediction_processor.process_mask_maps(pred_mask_maps, start_slice_id)
        
        if not self.augmented_inference:
            return prediction_processor.get_result()
    
        reversed_prediction_processor = method(self.model_input_shape, series_image.shape, self.foreground_threshold)
        # Reverse Predict
        for start_slice_id, series_part_data in SeriesSlicesGenerator(series_image, self.device, self.depth, first_slice_of_lobe, last_slice_of_lobe, reverse = True):
            pred_mask_maps = self.predict_nodule_in_slices(series_part_data)
            reversed_prediction_processor.process_mask_maps(pred_mask_maps, start_slice_id)
        final_rs = prediction_processor.get_result() | np.flip(reversed_prediction_processor.get_result(), axis = -1)    
        
        return final_rs 

    @torch.no_grad()
    def predict_nodule_in_slices(self, slices_image: Union[torch.Tensor, npt.NDArray[np.float32]]) -> torch.Tensor:
        """
        Args:
            slices_images: shape is same as 'self.model_input_shape'
        Return: torch.Tensor
            A prediction of mask maps, shape is (B, B, H, W, D)
        """
        output = self.model(slices_image)
        # Only get output512
        if len(output) >= 2:
            output = output[0]
        output = torch.squeeze(output, dim = 1) # (B, 1, D, H, W) -> (B, D, H, W)
        output = torch.permute(output, (0, 2, 3, 1)) # (B, D, H, W) -> (B, H, W, D)
        return output