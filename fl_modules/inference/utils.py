import os
import cc3d
import numpy as np
import numpy.typing as npt
from collections import defaultdict
from typing import Tuple, Union, List, Optional

HU_MIN, HU_MAX = -1000, 400

def compute_recall(tp, fn) -> float:
    return tp / max(tp + fn, 1)

def compute_precision(tp, fp) -> float:
    return tp / max(tp + fp, 1)

def compute_f1_score(recall: float, precision: float) -> float:
    if recall + precision == 0:
        return 0.0
    else:
        return (2 * precision * recall) / (precision + recall)

def normalize_paths(paths: Tuple[str, List[str], None]):
    if paths == None:
        paths = []
    elif isinstance(paths, str):
        paths = [paths]
    return paths

def load_model(stage: int, model , device):
    """
    Args:
        stage: int
            The stage of model, deciding which model used
        model: str or nn.Module
            The path to model or a model entity. If it is the path of model, then read it.
        device: torch.device
            The device of model, e.g. cpu or cuda.
    Return: nn.Module
    """
    import torch
    import torch.nn as nn
    from fl_modules.model.stage1.stage1_model import Stage1Model
    if isinstance(model, str):
        load_dict = torch.load(model)
        if model.endswith('.pth'):
            model = load_dict['model_structure']
            model.load_state_dict(load_dict['model_state_dict'])
        elif model.endswith('.pt'):
            if stage == 1:
                model_class = Stage1Model
            # elif stage == 2:
            #     model_class = Stage2Model
                
            model = model_class()
            # Load model weights
            model.load_state_dict(load_dict['model_state_dict'])
    model = model.to(device)
    return model

def load_gt_mask_maps(mask_maps_path: str) -> npt.NDArray[np.uint8]:
    gt_mask_maps = np.load(mask_maps_path)
    # npz
    if mask_maps_path.endswith('.npz'):
        gt_mask_maps = gt_mask_maps['image'] 

    # binarize
    bg_mask = (gt_mask_maps <= 125)
    gt_mask_maps[bg_mask] = 0
    gt_mask_maps[~bg_mask] = 1

    return gt_mask_maps.astype(np.uint8, copy=False)

def load_lobe(lobe_path: str) -> npt.NDArray[np.uint8]:
    lobe = np.load(lobe_path)['image']
    return lobe

def load_series_images(series_path: str) -> npt.NDArray[np.float32]:
    if not series_path.endswith('.npy'):
        raise ValueError("The file of series should be npy file!")
    series = np.load(series_path)
    # HU value conversion, which ranges from -1000 ~ 400
    series = np.clip(series, HU_MIN, HU_MAX)
    # transform HU values (-1000, 400) to (0, 1)
    series = (series - HU_MIN).astype(np.float32, copy=False) / (HU_MAX - HU_MIN)
    return series

def expand_neighboring_bbox2d(bbox1: List[int], bbox2: List[int], p=False) -> Tuple[List[int], List[int]]:
    """
    Args:
        bbox1: np.ndarray or List
            An array of (y_min, x_min, y_max, x_max)
        bbox2: np.ndarray
            An array of (y_min, x_min, y_max, x_max)
    """
    # Compute intersection area of two bounding box.
    a1, a2 = bbox1[0:2], bbox1[2:4] # a1 is (y_min, x_min) of bbox1 and a2 is (y_max, x_max) of bbox1.
    b1, b2 = bbox2[0:2], bbox2[2:4] # b1 is (y_min, x_min) of bbox2 and b2 is (y_max, x_max) of bbox2.
    inter_area = np.clip((np.minimum(a2, b2) - np.maximum(a1, b1)),0, None).prod()
    # If there are intersection with neighboring bboxe, then ignore it.
    if inter_area > 0:
        return bbox1, bbox2
    ### Revise the bbox, so that neighboring bbox2d has intersection.
    y_min1, x_min1, y_max1, x_max1 = bbox1
    y_min2, x_min2, y_max2, x_max2 = bbox2
    y_offset = min(y_max1, y_max2) - max(y_min1, y_min2)
    x_offset = min(x_max1, x_max2) - max(x_min1, x_min2)
    
    # y-axis does not has intersection
    if y_offset <= 0:
        y_offset = abs(y_offset) + 1
        y_offset1 = y_offset // 2
        y_offset2 = y_offset - y_offset1
        # Bbox1 is on the bottom of the Bbox2 along y-axis
        if y_max1 < y_min2:
            y_max1 = y_max1 + y_offset1
            y_min2 = y_min2 - y_offset2
        # Bbox1 is on the top of the Bbox2 along y-axis
        else:
            y_min1 = y_min1 - y_offset1
            y_max2 = y_max2 + y_offset2
    # x-axis has intersection
    if x_offset <= 0:
        x_offset = abs(x_offset) + 1
        x_offset1 = x_offset // 2
        x_offset2 = x_offset - x_offset1
        # Bbox1 is on the left of the Bbox2 along x-axis
        if x_max1 < x_min2:
            x_max1 = x_max1 + x_offset1
            x_min2 = x_min2 - x_offset2
        # Bbox1 is on the right of the Bbox2 along x-axis
        else:
            x_min1 = x_min1 - x_offset1
            x_max2 = x_max2 + x_offset2
        
    bbox1 = [y_min1, x_min1, y_max1, x_max1]
    bbox2 = [y_min2, x_min2, y_max2, x_max2]
    return bbox1, bbox2

def generate_bboxes_of_one_nodule(bbox: list, labels: np.ndarray, component_id: int) -> List[Tuple[int, int, int, int, int]]: 
    """
    Returns: List
        A List of bbox2d of one nodule, each element is a tuple of (z, y_min, x_min, y_max, x_max)
    """
    real_z_min, real_z_max = bbox[0][2], bbox[1][2]
    component_label = (labels[..., real_z_min: real_z_max] == component_id)
    
    y_indices, x_indices, z_indices = np.nonzero(component_label)
    valid_z_indices = np.sort(np.unique(z_indices))
    
    valid_bboxes = []
    for valid_z in valid_z_indices:
        cond = (z_indices == valid_z)
        ys, xs = y_indices[cond], x_indices[cond]
        y_min, y_max = min(ys), max(ys)
        x_min, x_max = min(xs), max(xs)
        valid_bboxes.append([y_min, x_min, y_max + 1, x_max + 1])
    
    # Expand bboxes for connecting neighbor bboxes
    for i in range(len(valid_z_indices) - 1):
        z1, z2 = valid_z_indices[i], valid_z_indices[i + 1]
        # There are not intersection along z-axis. We will generate missing bboxes in next step.
        if z2 - z1 != 1:
            continue
        valid_bboxes[i], valid_bboxes[i + 1] = expand_neighboring_bbox2d(valid_bboxes[i], valid_bboxes[i + 1])
    
    # Generate missing bboxes along z-axis
    valid_i = 0
    result_of_one_3d_nodule = []
    for z in range(0, max(valid_z_indices) + 1):
        real_z = z + real_z_min
        if z in valid_z_indices:
            y_min, x_min, y_max, x_max = valid_bboxes[valid_i] 
            result_of_one_3d_nodule.append([real_z, y_min, x_min, y_max, x_max])
            valid_i = valid_i + 1
        else:
            # Generate missing bboxes
            last_valid_z, last_valid_bbox = valid_z_indices[valid_i], valid_bboxes[valid_i]
            next_valid_z, next_valid_bbox = valid_z_indices[valid_i - 1], valid_bboxes[valid_i - 1]
            
            last_valid_bbox = np.array(last_valid_bbox, dtype = np.float64)
            next_valid_bbox = np.array(next_valid_bbox, dtype = np.float64)
            ratio = (next_valid_z - z) / (next_valid_z - last_valid_z)
            new_bbox = np.round(ratio * last_valid_bbox + (1 - ratio) * next_valid_bbox).astype(np.int32)
            y_min, x_min, y_max, x_max = new_bbox.tolist()
            result_of_one_3d_nodule.append([real_z, y_min, x_min, y_max, x_max])
    return result_of_one_3d_nodule

def compute_bbox3d_intersection_volume(box1: npt.NDArray[np.int32], box2: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
    """ 
    Args:
        box1 (shape = [N, 2, 3])
        box2 (shape = [M, 2, 3])
    Return:
        the area of the intersection between box1 and box2, shape = [N, M]
    """
    a1, a2 = box1[:,np.newaxis, 0,:], box1[:,np.newaxis, 1,:] # [N, 1, 3]
    b1, b2 = box2[np.newaxis,:, 0,:], box2[np.newaxis,:, 1,:] # [1, N, 3]
    inter_area = np.clip((np.minimum(a2, b2) - np.maximum(a1, b1)),0, None).prod(axis=2)

    return inter_area

def get_3d_connected_componment(binary_mask_maps: npt.NDArray[np.uint8], 
                                nodule_3d_minimum_size: int,
                                nodule_3d_minimum_thickness: int,
                                combined_offset: Optional[Tuple[npt.NDArray[np.int32], List[int]]] = [],
                                ) -> Tuple[npt.NDArray[np.uint32], npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """Calculate the 3d connected componet of 26-connected for given binary mask maps

    Args:
        binary_mask_maps: np.ndarray
            An array of the binary maps which be conveted.
        nodule_3d_minimum_size: int
            The 3d nodule whose size is less than this argument will be removed. 
            
        combined_offset: combined_offset for combine bbox
    Return: tuple 
        A tuple of (labels, valid_component_indices, valid_nodule_sizes, bboxes)
    """
    labels = cc3d.connected_components(binary_mask_maps, out_dtype=np.uint32)
    stats = cc3d.statistics(labels)
    valid_component_indices = [] # The ID of component whose number of element is larger than nodule_3d_minimum_size.
    valid_nodule_sizes = []
    bboxes = []

    # Get the two diagonal vertex's coordinate(xMin, yMin, zMin) and (xMax, yMax, zMax) of a cube 
    for component_id, (counts, box) in enumerate(zip(stats['voxel_counts'], stats['bounding_boxes'])):
        # If componentID == 0, this compoent is background.
        # If this component is too small, then ignore it.
        if component_id == 0 or counts < nodule_3d_minimum_size:
            continue

        valid_component_indices.append(component_id)
        valid_nodule_sizes.append(counts)
        # y is top-to-down
        # x is left-to-right
        y_range, x_range, z_range = box 
        
        y_min, y_max = y_range.start, y_range.stop
        x_min, x_max = x_range.start, x_range.stop
        z_min, z_max = z_range.start, z_range.stop
        coord = [[y_min, x_min, z_min], [y_max, x_max, z_max]]
        bboxes.append(coord)

    valid_component_indices = np.array(valid_component_indices, dtype = np.int32)
    bboxes = np.array(bboxes, dtype = np.int32) # (N, 2, 3), the order is (y, x, z)
    valid_nodule_sizes = np.array(valid_nodule_sizes, dtype = np.int32)
        
    # If combined_offset is given, it combines the neightbor bbox.
    if len(combined_offset) != 0 and len(valid_component_indices) != 0 and len(bboxes) >= 2:
        labels, valid_component_indices, valid_nodule_sizes, bboxes = combine_neighbor_bbox(labels, valid_component_indices, valid_nodule_sizes, bboxes, combined_offset)
    
    # Remove 3d nodule whose size is less than nodule_3d_minimum_size or thickness is less than nodule_3d_minimum_thickness
    if len(bboxes) > 0:
        nodule_3d_thickness = bboxes[:, 1, 2] - bboxes[:, 0, 2]
        valid_cond = (valid_nodule_sizes >= nodule_3d_minimum_size) & (nodule_3d_thickness >= nodule_3d_minimum_thickness)
        valid_component_indices = valid_component_indices[valid_cond]
        valid_nodule_sizes = valid_nodule_sizes[valid_cond]
        bboxes = bboxes[valid_cond]
    
    return labels, valid_component_indices, valid_nodule_sizes, bboxes

def combine_neighbor_bbox(labels: npt.NDArray[np.uint32], 
                          valid_component_indices: npt.NDArray[np.int32], 
                          valid_nodule_sizes: npt.NDArray[np.int32], 
                          bboxes: npt.NDArray[np.int32],
                          combined_offset: npt.NDArray[np.int32]
                          ) -> Tuple[npt.NDArray[np.uint32], npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    if not isinstance(combined_offset, np.ndarray):
        combined_offset = np.array(combined_offset)
    
    offset_pred_bboxes = bboxes + combined_offset
    rows, cols = np.nonzero(np.triu(compute_bbox3d_intersection_volume(offset_pred_bboxes, offset_pred_bboxes)))
    not_equal_indices = (rows != cols)
    rows = rows[not_equal_indices]
    cols = cols[not_equal_indices]

    # Combine neighbor bbox
    search_sets = list(range(len(bboxes)))
    def get_head(idx: int):
        if search_sets[idx] == idx:
            return idx
        else:
            search_sets[idx] = get_head(search_sets[idx])
            return search_sets[idx]

    for idx1, idx2 in zip(rows, cols):
        head1, head2 = get_head(idx1), get_head(idx2)
        if head1 != head2:
            search_sets[head2] = head1

    disjoint_sets = defaultdict(set)
    for i, head in enumerate(search_sets):
        head = get_head(head)
        disjoint_sets[head].add(i)

    combined_indices_sets = []
    no_combined_indices = []
    for head, indices in disjoint_sets.items():
        if len(indices) == 1:
            no_combined_indices.append(head)
        else:
            combined_indices_sets.append(list(indices))
    
    # Add indices which not do combine
    new_valid_component_indices = valid_component_indices[no_combined_indices].tolist()
    new_valid_nodule_sizes = valid_nodule_sizes[no_combined_indices].tolist()
    new_bboxes = bboxes[no_combined_indices].tolist()

    for indices in combined_indices_sets:
        new_nodule_size = int(np.sum(valid_nodule_sizes[indices]))
        bbox = bboxes[indices].copy()
        new_bbox = np.array([np.min(bbox[:, 0, :], axis=0), np.max(bbox[:, 1, :], axis=0)]) # shape is (2, 3)
        target_component_index = valid_component_indices[indices[0]]
    
        new_bboxes.append(new_bbox)
        new_valid_nodule_sizes.append(new_nodule_size)
        new_valid_component_indices.append(target_component_index)

    if len(combined_indices_sets) != 0:
        # indices_mapping = defaultdict(int)
        mapping = np.zeros((np.max(labels) + 1), dtype=labels.dtype)
        for indices in combined_indices_sets:
            target_component_index = valid_component_indices[indices[0]]
            for i in indices:
                mapping[valid_component_indices[i]] = target_component_index
        for i in no_combined_indices:
            mapping[valid_component_indices[i]] = valid_component_indices[i]
        labels = np.take(mapping, labels)

    new_bboxes = np.array(new_bboxes, dtype = bboxes.dtype)
    new_valid_nodule_sizes = np.array(new_valid_nodule_sizes, dtype = valid_nodule_sizes.dtype)
    new_valid_component_indices = np.array(new_valid_component_indices, dtype = valid_component_indices.dtype)

    return labels, new_valid_component_indices, new_valid_nodule_sizes, new_bboxes

def get_3d_connected_componment_after_lobe(pred_binary_mask_maps: npt.NDArray[np.uint8], 
                                            lobe_path: str, 
                                            nodule_3d_minimum_size: int,
                                            nodule_3d_minimum_thickness: int,
                                            combined_offset: Optional[Tuple[npt.NDArray[np.int32], List[int]]] = [],
                                            ) -> Tuple[npt.NDArray[np.uint32], npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """
    Return: tuple 
        A tuple of (labels, valid_component_indices, valid_nodule_sizes, bboxes)
    """
    # Use lobe segmentation to remove FP nodule in prediction
    if os.path.exists(lobe_path):
        lobe = load_lobe(lobe_path)
        pred_binary_mask_maps = pred_binary_mask_maps & lobe
    pred_labels, pred_valid_component_indices, pred_valid_nodule_sizes, pred_bboxes = get_3d_connected_componment(pred_binary_mask_maps, nodule_3d_minimum_size, nodule_3d_minimum_thickness, combined_offset)
    return pred_labels, pred_valid_component_indices, pred_valid_nodule_sizes, pred_bboxes