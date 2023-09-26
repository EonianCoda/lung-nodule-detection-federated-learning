import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import numpy as np
import numpy.typing as npt
from typing import List

import torch
from .utils import load_series_images

logger = logging.getLogger(__name__)

class PrefetchSeries:
    def __init__(self, series_path_list: List[str]):
        self.executor = ThreadPoolExecutor()
        self.series_path_list = series_path_list

    def __iter__(self):
        if len(self.series_path_list) == 0:
            return
        # initialize first data
        future = self.executor.submit(self._prefetch_next_data, 0)

        for cur_idx in range(1, len(self)):
            cur_data = future.result()
            # Prefetch next data before yield current data
            future = self.executor.submit(self._prefetch_next_data, cur_idx)
            yield cur_data

        last_data = future.result()
        yield last_data

    def _prefetch_next_data(self, idx: int) -> torch.Tensor:
        """
        Return: torch.Tensor
            A image series data with shape (D, H, W)
        """
        series_path = self.series_path_list[idx]
        series_data = load_series_images(series_path)
        series_data = torch.tensor(series_data, dtype = torch.float32)
        series_data = series_data.permute(2, 0, 1) # (H, W, D) -> (D, H, W)
        return series_data

    def __len__(self):
        return len(self.series_path_list)

class ParallelProcessPrediction(object):
    def __init__(self, model_input_shape: tuple, image_shape: tuple, fg_threshold: float):
        self.executor = ThreadPoolExecutor()
        self.futures = []

        self.depth = model_input_shape[-1]
        self.stride = self.depth // 2
        
        # image_shape is (D, H, W)
        self.num_slice = image_shape[0]
        d, h, w = image_shape
        self.result = np.zeros((h, w, d), dtype=np.float32)
        self.fg_threshold = fg_threshold

        self.locks = [Lock() for _ in range(self.num_slice // self.stride + 1)]
    
    def wait_until_finish(self):
        if len(self.futures) == 0:
            return
        for f in self.futures:
            while not f.done():
                continue

    def process_mask_maps(self, pred_mask_maps: torch.Tensor, start_slice_id:int):
        self.futures.append(self.executor.submit(self._parallel_process, pred_mask_maps, start_slice_id))

    def _parallel_process(self, pred_mask_maps: torch.Tensor, start_slice_id:int):
        pred_mask_maps = pred_mask_maps.cpu().numpy()
        for pred_mask_map in pred_mask_maps:
            end_slice_id = start_slice_id + self.depth
            if end_slice_id > self.num_slice:
                start_slice_id = self.num_slice - self.depth
                end_slice_id = self.num_slice

            lock_idx = start_slice_id // self.stride
            
            if not self.locks[lock_idx + 1].acquire(timeout = 60):
                release_flag1 = False
                logger.error(f'Cannot acquire lock {lock_idx + 1}')
            else:
                release_flag1 = True
                
            if not self.locks[lock_idx + 2].acquire(timeout = 60):
                release_flag2 = False
                logger.error(f'Cannot acquire lock {lock_idx + 2}')
            else:
                release_flag2 = True
                
            try:
                self._process(pred_mask_map, start_slice_id, end_slice_id)
            except Exception as e:
                logger.error(e)
            finally:
                if release_flag1:
                    self.locks[lock_idx + 1].release()
                if release_flag2:
                    self.locks[lock_idx + 2].release()
            
            start_slice_id += self.stride

    def get_result(self):
        raise NotImplementedError

    def _process(self, pred_mask_map: torch.Tensor, start_slice_id:int, end_slice_id: int):
        raise NotImplementedError

class AvgMethodProcessPrediction(ParallelProcessPrediction):
    def __init__(self, model_input_shape: tuple, image_shape: tuple, fg_threshold: float):
        super().__init__(model_input_shape, image_shape, fg_threshold)
        self.fg_bound = np.zeros((self.num_slice, ), dtype=np.float32)

    def _process(self, pred_mask_map: torch.Tensor, start_slice_id:int, end_slice_id: int):
        self.result[..., start_slice_id: end_slice_id] += pred_mask_map
        self.fg_bound[start_slice_id: end_slice_id] += self.fg_threshold

    def get_result(self) -> npt.NDArray[np.uint8]:
        self.wait_until_finish()
        mask_maps = np.where(self.result > np.expand_dims(self.fg_bound, axis=(0, 1)), 1, 0)  
        return mask_maps.astype(np.uint8, copy=False)

class MaxMethodProcessPrediction(ParallelProcessPrediction):
    def __init__(self, model_input_shape: tuple, image_shape: tuple, fg_threshold: float):
        self.executor = ThreadPoolExecutor()
        self.futures = []

        self.depth = model_input_shape[-1]
        self.stride = self.depth // 2
        self.num_slice = image_shape[-1]
        self.result = np.zeros(image_shape, dtype=np.uint8)
        self.fg_threshold = fg_threshold

        self.locks = [Lock() for _ in range(self.num_slice // self.stride + 1)]
        
    def _process(self, pred_mask_map: torch.Tensor, start_slice_id:int, end_slice_id: int):
        self.result[..., start_slice_id: end_slice_id] |= (pred_mask_map > self.fg_threshold)

    def get_result(self) -> npt.NDArray[np.uint8]:
        self.wait_until_finish()
        return self.result.copy()