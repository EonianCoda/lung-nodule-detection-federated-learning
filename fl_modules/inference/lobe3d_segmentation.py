import cv2
import cc3d
import os
import logging
import numpy as np
from os.path import join, dirname
from scipy import ndimage as nd
from typing import Union, Tuple, List
from numpy.typing import NDArray
from multiprocessing.pool import Pool
from .utils import normalize_paths, HU_MIN, HU_MAX

logger = logging.getLogger(__name__)

def get_first_and_last_slice_of_lobe(path: str):
    if not os.path.exists(path):
        logger.warning("First and last slice of lobe {} does not exist!".format(path))
        return 0, -1
    
    with open(path, 'r') as f:
        lines = f.readlines()[3:]
        lines = [line.strip() for line in lines]
    first_slice_id, end_slice_id = [int(v) for v in lines[0].split(',')]
    return first_slice_id, end_slice_id

def prepare_lobe_of_series(series_paths: Union[List[str], str], overwrite: bool = False) -> Tuple[List[str], List[str]]:
    series_paths = normalize_paths(series_paths)
    # Generate lobe path, e.g one series_path is '/Ser-001/Ser-001.npy', then the lobe path is '/Ser-001/Ser-001_lobe.npz'
    lobe_save_paths = []
    lobe_info_save_paths = []
    for series_path in series_paths:
        lobe_save_path = series_path_to_lobe_path(series_path)
        lobe_info_save_path = join(dirname(lobe_save_path), 'lobe_info.txt')
        
        lobe_save_paths.append(lobe_save_path)
        lobe_info_save_paths.append(lobe_info_save_path)
    
    # Compute the lobe segmentation for given series.
    LobeSegmentation.compute_and_save_lobe_mask_of_multiple_series(image_paths = series_paths,
                                                                lobe_save_paths = lobe_save_paths, 
                                                                lobe_info_save_paths = lobe_info_save_paths,
                                                                overwrite = overwrite)
    return lobe_save_paths, lobe_info_save_paths

def series_path_to_lobe_path(series_path: str) -> str:
    """Convert path of series to path of lobe
        
        For example, from series path '/Ser-001/Ser-001.npy' to '/Ser-001/Ser-001_lobe.npz' 
    """
    series_folder = os.path.dirname(series_path)
    series_name = os.path.basename(series_path).split('.')[0]
    lobe_file_name = '{}_lobe.npz'.format(series_name)
    lobe_path = os.path.join(series_folder, lobe_file_name)
    return lobe_path

class LobeSegmentation:
    @staticmethod
    def compute_and_save_lobe_mask_of_multiple_series(image_paths: List[str], 
                                                      lobe_save_paths: List[str],
                                                      lobe_info_save_paths: List[str],
                                                      overwrite) -> None:
        """
        Args:
            image_path: list
                a list contains serveral paths to series image or nd.array of series image
            save_path_list: list
                a list contatins serveral paths for saving the result of lobe mask 
            overwrite: bool
                a flag to determine if overwriting the result of lobe mask when it exists, defulat = False
        """
        assert(len(image_paths) == len(lobe_save_paths))
        def error_when_running(error):
            print(error)
        pool = Pool(processes=os.cpu_count() // 4)
        try:
            logger.debug("Start of computing and saving the lobe mask")
            for image_path, lobe_save_path, lobe_info_save_path in zip(image_paths, lobe_save_paths, lobe_info_save_paths):
                pool.apply_async(LobeSegmentation.compute_and_save_lobe_mask_of_series, 
                                args = (image_path, lobe_save_path, lobe_info_save_path, overwrite),
                                error_callback = error_when_running)
            pool.close()
            pool.join()
            logger.debug("End of computing and saving the lobe mask")
        finally:
            pool.terminate()
    
    @staticmethod
    def compute_and_save_lobe_mask_of_series(image_path: str, 
                                             lobe_save_path: str, 
                                             lobe_info_save_path: str, 
                                             overwrite) -> None:
        """Compute lobe mask of series image and then save it as .npz file
        Args:
            images: str or NDArray[np.int16]
                if it is str, then it will be regard as a path to series image
            save_path: str
                a path for saving
            overwrite: bool
                a flag to determine if overwriting the result of lobe mask when it exists, defulat = False
        """
        assert(lobe_save_path.endswith('npz'))
        if not overwrite and os.path.exists(lobe_save_path):
            return
        
        # Check if gpu available
        import torch
        use_gpu = torch.cuda.is_available()
        # Compute lobe mask of series
        lobe_3d, lobe_bbox_3d = LobeSegmentation.compute_lobe_mask_of_series(image_path, use_gpu)
        # Save Lobe Mask
        os.makedirs(dirname(lobe_save_path), exist_ok=True)
        np.savez_compressed(lobe_save_path, image = lobe_3d)
        
        # Save txt which containing the inforamtion of lobe (e.g bbox)
        y_min, y_max = lobe_bbox_3d[0] # top-to-down
        x_min, x_max = lobe_bbox_3d[1] # left-to-right
        first_slice_id, end_slice_id = lobe_bbox_3d[2]
        lines = ['Valid slice id (z_min, z_max)\n',
                'Lobe Bbox (top_down_min, top_down_max, left_right_min, left_right_max)\n',
                '-----\n',
                f'{first_slice_id},{end_slice_id}\n',
                f'{y_min},{y_max},{x_min},{x_max}']
        
        with open(lobe_info_save_path, 'w') as f:
            f.writelines(lines)

    @staticmethod
    def compute_lobe_mask_of_series(image_path: str, use_gpu: bool) -> Tuple[NDArray[np.uint8], NDArray[np.int32]]:
        """
        Return: NDArray[np.uint8]
            lobe binary mask
        """
        dicom_3d = LobeSegmentation._load_series_image(image_path, use_gpu)
        binary_3d = LobeSegmentation._binarization(dicom_3d, use_gpu)
        lobe_3d, lobe_bbox_3d = LobeSegmentation._lobe_segmentation(binary_3d, use_gpu)
        return lobe_3d, lobe_bbox_3d

    @staticmethod
    def _load_series_image(image_path: str, use_gpu: bool):
        if use_gpu:
            import cupy as cp
            image = cp.load(image_path)
        else:
            image = np.load(image_path)
        return image

    @staticmethod
    def _binarization(dicom_3d: np.ndarray, use_gpu: bool) -> NDArray[np.uint8]:
        if use_gpu:
            from cupyx.scipy import ndimage as cupy_nd
            import cupy as cp
            # initialize gaussian kernel
            x, y, z = cp.mgrid[-2:3, -2:3, -2:3]
            gaussian_kernel = cp.exp(-(x**2 + y**2 + z**2))

            # Calculate threshold
            dicom_3d = cp.clip(dicom_3d, HU_MIN, HU_MAX)
            dicom_3d = cupy_nd.convolve(dicom_3d, gaussian_kernel, mode = 'constant', cval = 0.0) # gaussian smooth filter initialization
            threshold = cp.mean(dicom_3d)
            # Binarize
            binary_3d = cp.zeros(dicom_3d.shape, cp.uint8)
            binary_3d[dicom_3d <= -threshold] = 1
            binary_3d[dicom_3d > threshold] = 0
            binary_3d = cp.asnumpy(binary_3d)
        else:
            # initialize gaussian kernel
            x, y, z = np.mgrid[-2:3, -2:3, -2:3]
            gaussian_kernel = np.exp(-(x**2 + y**2 + z**2))

            # Calculate threshold
            dicom_3d = np.clip(dicom_3d, HU_MIN, HU_MAX)
            dicom_3d = nd.convolve(dicom_3d, gaussian_kernel, mode = 'constant', cval = 0.0) # gaussian smooth filter initialization
            threshold = dicom_3d.mean()
            # Binarize
            binary_3d = np.zeros(dicom_3d.shape, np.uint8)
            binary_3d[dicom_3d <= -threshold] = 1
            binary_3d[dicom_3d > threshold] = 0
        return binary_3d
    
    @staticmethod
    def _lobe_segmentation(lobe_3d: NDArray[np.uint8], use_gpu) -> Tuple[NDArray[np.uint8], NDArray[np.int32]]:
        _, _ , num_slice = lobe_3d.shape
        
        # Flood Fill
        for z in range(num_slice):
            flood_fill_msk = np.zeros((514, 514), np.uint8)
            temp = lobe_3d[:,:, z].copy()
            if lobe_3d[510, 510, z].astype(np.uint8) > 10:
                cv2.floodFill(temp, flood_fill_msk, (510, 510), 0)
            else:
                cv2.floodFill(temp, flood_fill_msk, (5, 5), 0)
            lobe_3d[:, :, z] = temp

        if use_gpu:
            import cupy as cp
            from cupyx.scipy import ndimage as cupy_nd
            lobe_3d = cp.asarray(lobe_3d)
            binary_closing_structue = cp.ones((5, 5, 5))
            lobe_3d = cupy_nd.binary_closing(lobe_3d, structure = binary_closing_structue)
            lobe_3d = lobe_3d.astype(cp.uint8)
            lobe_3d = cp.asnumpy(lobe_3d)
        else:
            binary_closing_structue = np.ones((5, 5, 5))
            lobe_3d = nd.binary_closing(lobe_3d, structure = binary_closing_structue)
            lobe_3d = lobe_3d.astype(np.uint8)

        # 3D C.C label selet largest area
        labels_out, N = cc3d.largest_k(lobe_3d, k=2, connectivity=26, delta=0, return_N=True)
        stats = cc3d.statistics(labels_out) # any element in stats 's shape is (3, ...)
        
        target = -1
        valid_indices = np.argsort(stats['voxel_counts'])[::-1]
        valid_indices = valid_indices[1:] # The biggest of component is always background and index is 0
        for component_idx in valid_indices:
            bbox = stats['bounding_boxes'][component_idx]
            y_range, x_range, _ = bbox
            y_min, y_max = y_range.start, y_range.stop
            x_min, x_max = x_range.start, x_range.stop
            y_length = y_max - y_min
            x_length = x_max - x_min
            
            center_y, center_x, _ = stats['centroids'][component_idx]
            if not((x_length > 480 or y_length > 480) and ((center_y > 412 or center_y < 100) or (center_x < 50 or center_x > 462))):
                target = component_idx
                break
        
        # There are not any component fitting condition, then using the union of all top2 compoent
        if target == -1:
            lobe_3d = (labels_out > 0)
        else:
            lobe_3d = (labels_out == target)
        
        # Dilation
        lobe_3d = lobe_3d.astype(np.uint8)
        top_down, left_right, z = np.nonzero(lobe_3d)
        start_slice_id = np.min(z)
        end_slice_id = min(np.max(z) + 1, num_slice) # nonzero of z means that the slice indexing by np.max(z) has some nonzero element, so we need to add 1.
        
        ellipse9x9 = LobeSegmentation.create_ellipse_structure(9, z_offset = 2, y_offset = 0, x_offset = 0)
        ellipse9x9_flat = LobeSegmentation.create_ellipse_structure(9, z_offset = 2, y_offset = 1, x_offset = 0)
        ellipse7x7_flat = LobeSegmentation.create_ellipse_structure(7, z_offset = 2, y_offset = 1, x_offset = 0)
        
        if use_gpu:
            import cupy as cp
            from cupyx.scipy import ndimage as cupy_nd
            lobe_3d = cp.asarray(lobe_3d)
            ellipse9x9_flat = cp.asarray(ellipse9x9_flat)
            ellipse9x9 = cp.asarray(ellipse9x9)
            ellipse7x7_flat = cp.asarray(ellipse7x7_flat)
            lobe_3d = lobe_3d.astype(cp.uint8)
            top_down, left_right, z = cp.nonzero(lobe_3d)
            start_slice_id = cp.min(z)
            end_slice_id = min(cp.max(z) + 1, num_slice)
            
            lobe_3d = cupy_nd.binary_opening(lobe_3d, structure = ellipse7x7_flat, iterations = 1) # Remove small struecture
            lobe_3d = cupy_nd.binary_dilation(lobe_3d, structure = ellipse9x9_flat, iterations = 2, brute_force = True)
            lobe_3d = cupy_nd.binary_closing(lobe_3d, structure = ellipse9x9_flat, iterations = 2, brute_force = True)
            lobe_3d = cupy_nd.binary_dilation(lobe_3d, structure = ellipse9x9, iterations = 2, brute_force = True) # Return dtype is bool
            
            lobe_3d[..., :start_slice_id] = 0
            lobe_3d[..., end_slice_id: ] = 0
            
            lobe_3d = lobe_3d.astype(cp.uint8)
            # Convert back to cpu
            lobe_3d = cp.asnumpy(lobe_3d)
            top_down = cp.asnumpy(top_down)
            left_right = cp.asnumpy(left_right)
        else:
            lobe_3d = lobe_3d.astype(np.uint8)
            top_down, left_right, z = np.nonzero(lobe_3d)
            start_slice_id = np.min(z)
            end_slice_id = min(np.max(z) + 1, num_slice)
            
            lobe_3d = nd.binary_opening(lobe_3d, structure = ellipse7x7_flat, iterations = 1) # Remove small struecture
            lobe_3d = nd.binary_dilation(lobe_3d, structure = ellipse9x9_flat, iterations = 2)
            lobe_3d = nd.binary_closing(lobe_3d, structure = ellipse9x9_flat, iterations = 2)
            lobe_3d = nd.binary_dilation(lobe_3d, structure = ellipse9x9, iterations = 2)
            
            binary_closing_structue = np.ones((7, 7, 7))
            lobe_3d = nd.binary_closing(lobe_3d, structure = binary_closing_structue)
            
            lobe_3d[..., :start_slice_id] = 0
            lobe_3d[..., end_slice_id: ] = 0
            lobe_3d = lobe_3d.astype(np.uint8)
        
        start_slice_id = int(start_slice_id)
        end_slice_id = int(end_slice_id)
        
        # Fill hole in lobe
        for z in range(start_slice_id, end_slice_id):
            flood_fill_msk = np.zeros((514, 514), np.uint8)
            temp = lobe_3d[:,:, z].copy()
            if lobe_3d[510, 510, z].astype(np.uint8) > 10:
                cv2.floodFill(temp, flood_fill_msk, (510, 510), 0)
            else:
                cv2.floodFill(temp, flood_fill_msk, (5, 5), 0)
            lobe_3d[:, :, z] = (1 - flood_fill_msk)[1:513, 1:513]
        top_down, left_right, z = np.nonzero(lobe_3d)
        
        y_min, y_max = np.min(top_down), np.max(top_down)
        x_min, x_max = np.min(left_right), np.max(left_right)
        height_bbox3d = y_max - y_min
        width_bbox3d = x_max - x_min
        # Find meaningful first slice
        for first_meaningful_slice_id in range(start_slice_id, end_slice_id):
            lobe_2d = lobe_3d[..., first_meaningful_slice_id]
            if LobeSegmentation.is_slice_meaningful(lobe_2d, height_bbox3d, width_bbox3d):
                break
        # Find meaningful end slice
        for end_meaningful_slice_id in range(end_slice_id - 1, max(start_slice_id - 1, 0), -1):
            lobe_2d = lobe_3d[..., end_meaningful_slice_id]
            if LobeSegmentation.is_slice_meaningful(lobe_2d, height_bbox3d, width_bbox3d):
                break
        
        end_meaningful_slice_id = min(end_meaningful_slice_id + 1, num_slice)
        lobe_3d[..., :first_meaningful_slice_id] = 0
        lobe_3d[..., end_meaningful_slice_id:] = 0
        lobe_bbox_3d = np.array([[y_min, y_max], [x_min, x_max], [first_meaningful_slice_id, end_meaningful_slice_id]], dtype = np.int32)
        return lobe_3d, lobe_bbox_3d

    @staticmethod
    def is_slice_meaningful(lobe_2d, height_bbox3d: int, width_bbox3d: int):
        """Check whether the number of componet is larger than 3 or the compoent in lobe_2d is bigger than threshold.
        
        1. If the number of componet is larger than 3, it means that the left and right side of lung appear in the lobe image.
        2. If the compoent in lobe_2d is bigger than threshold, it means that the left and right side of lung appear in the lobe image but they are connected with each other.
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(lobe_2d, connectivity = 8)
        if num_labels >= 3:
            return True
        elif num_labels == 2:
            # The order of return value 'stats' is different from cc3d.
            # Variable of 'x' here is also left-to-right, and Variable of 'y' here is top-down
            x, y, w_bbox2d, h_bbox2d, areas = stats[1] # index 0 is background
            if (w_bbox2d > width_bbox3d / 4) or (h_bbox2d > height_bbox3d / 4):
                return True
        else:
            return False

    @staticmethod
    def create_ellipse_structure(size_of_element: int, z_offset = 2, y_offset = 1, x_offset = 0) -> NDArray[np.uint8]:
        size = (size_of_element, size_of_element, size_of_element)
        center = tuple(np.array(size) // 2)
        radius = size_of_element // 2 + 1
        radius = (radius - z_offset, radius - y_offset, radius - x_offset)  # (depth, height, width)
        ellipse_structure = np.zeros(size, dtype=bool)
        for z in range(size[0]):
            for y in range(size[1]):
                for x in range(size[2]):
                    if ((z - center[0]) / radius[0]) ** 2 + ((y - center[1]) / radius[1]) ** 2 + ((x - center[2]) / radius[2]) ** 2 <= 1:
                        ellipse_structure[z, y, x] = 1
        return ellipse_structure