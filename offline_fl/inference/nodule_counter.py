import os
import json
import logging
from multiprocessing import Pool
from typing import Dict, Union, List

from offline_fl.dataset.utils import get_nodule_type, load_series_list
from offline_fl.inference.utils import load_gt_mask_maps, get_3d_connected_componment

LAST_MODIFIED_TIME = 'last_modified_time'
NODULE_SIZE = 'nodule_size'
BBOXES = 'bboxes'
NODULE_START_SLICE_IDS = 'nodule_start_slice_ids'

logger = logging.getLogger(__name__)

class NoduleCounter:
    @staticmethod
    def count_and_analyze_nodules_of_multi_series(series_list_path: str, 
                                                  nodule_size_ranges: dict,
                                                  mode: str = 'sum') -> Union[Dict[str, int], List[Dict[str, int]]]:
        """
        Returns: dict[str, int]
            Option1: mode = 'sum'
                a dict of (nodule_type, number of nodule)
            Option2: mode = 'single'
                List of dict of (nodule_type, number of nodule)
        """

        gt_mask_maps_paths = []
        for folder, file_name in load_series_list(series_list_path):
            gt_mask_maps_path = os.path.join(folder,
                                    'mask', 
                                    f'{file_name}.npz')
            gt_mask_maps_paths.append(gt_mask_maps_path)

        pool = Pool(os.cpu_count() // 2)
        target_args = [(gt_maks_maps_path, nodule_size_ranges) for gt_maks_maps_path in gt_mask_maps_paths]
        try:
            nodule_counts_of_series = pool.starmap(NoduleCounter.count_and_analyze_nodules, target_args)
        finally:
            pool.close()
            pool.join()
            pool.terminate()
        
        if mode == 'sum':
            nodule_counts = {nodule_type: 0 for nodule_type in nodule_size_ranges.keys()}
            for counts in nodule_counts_of_series:
                for nodule_type in nodule_size_ranges.keys():
                    nodule_counts[nodule_type] = nodule_counts[nodule_type] + counts[nodule_type]
            nodule_counts['all'] = sum(nodule_counts.values())
            return nodule_counts
        elif mode == 'single':
            return nodule_counts_of_series
        else:
            raise NotImplementedError

    @staticmethod
    def count_and_analyze_nodules(gt_mask_map_path: str, nodule_size_ranges: dict) -> Dict[str, int]:
        """Count and analyze nodules in given groud truth mask map
        Args:
            gt_mask_map_path: str
                A path to groud truth mask map.
            nodule_size_ranges: dict
                A dict of different nodule types and their pixel size.
        Returns:
            a dict of (<nodule_type>, number of nodule for this nodule type)
        """
        cache_path = NoduleCounter.get_cache_path(gt_mask_map_path)
        
        if not os.path.exists(cache_path):
            gt_valid_nodule_sizes = NoduleCounter.get_and_write_num_of_nodules(gt_mask_map_path, cache_path)
        else: # Read cache
            with open(cache_path, 'r') as f:
                nodule_count = json.load(f)
            last_modified_time = os.path.getmtime(gt_mask_map_path)
            # Ground truth does not update, just read it.
            if last_modified_time == nodule_count[LAST_MODIFIED_TIME]:
                gt_valid_nodule_sizes = nodule_count[NODULE_SIZE]
            else:
                gt_valid_nodule_sizes = NoduleCounter.get_and_write_num_of_nodules(gt_mask_map_path, cache_path)
        
        # Count number of nodule of differenct nodule type
        rs = {nodule_type: 0 for nodule_type in nodule_size_ranges.keys()}
        for nodule_size in gt_valid_nodule_sizes:
            nodule_type = get_nodule_type(nodule_size, nodule_size_ranges)
            rs[nodule_type] += 1
        return rs
    
    @staticmethod
    def get_and_write_num_of_nodules(gt_mask_map_path: str, save_path: str):
        last_modified_time = os.path.getmtime(gt_mask_map_path)
    
        gt_mask_maps = load_gt_mask_maps(gt_mask_map_path)
        _, _, gt_valid_nodule_sizes, bboxes = get_3d_connected_componment(gt_mask_maps, nodule_3d_minimum_size = 0, nodule_3d_minimum_thickness = 0)
        gt_valid_nodule_sizes = gt_valid_nodule_sizes.tolist()
        
        if len(bboxes) != 0:
            nodule_start_slice_ids = bboxes[:, 0, 2] 
            nodule_start_slice_ids = nodule_start_slice_ids.tolist() # (N,)
            bboxes = bboxes.tolist() # shape = (N, 2, 3)
        else:
            bboxes = []
            nodule_start_slice_ids = []
            
        nodule_count = {LAST_MODIFIED_TIME: last_modified_time, 
                        NODULE_SIZE: gt_valid_nodule_sizes,
                        BBOXES: bboxes,
                        NODULE_START_SLICE_IDS: nodule_start_slice_ids}
        
        with open(save_path, 'w') as f:
            json.dump(nodule_count, f)
            
        return gt_valid_nodule_sizes
    
    @staticmethod
    def get_cache_path(gt_mask_map_path: str) -> str:
        extension = gt_mask_map_path.split('.')[-1]
        cache_path = gt_mask_map_path.replace(f'.{extension}', '_nodule_count.json')
        return cache_path