import cv2
import os
import math
import numpy as np
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt

from fl_modules.dataset.utils import ALL_LOC, ALL_RAD, load_image, load_label, gen_dicom_path, gen_label_path
from fl_modules.eval.nodule_finding import NoduleFinding
from .convert import noduleFinding2cude

MAX_IMAGES = 15
SUBPLOT_WIDTH = 3.5
SUBPLOT_HEIGHT = 3

def draw_bbox(image: np.ndarray, bboxes: np.ndarray, color = (255, 0, 0)) -> np.ndarray:
    """
    Args:
        image: a 3D image with shape [Z, Y, X, 3]
    """
    for z1, y1, x1, z2, y2, x2 in bboxes:
        for z in range(int(z1), int(z2)):
            image[z] = cv2.rectangle(image[z].copy(), (x1, y1), (x2, y2), color, 1)
    return image

def draw_bbox_on_image(image: np.ndarray, bboxes: np.ndarray, color = (255, 0, 0), half_image = True, axis_off = True, 
                       save_path = None, extra_sup_title = None, show = False, offset = 0, bbox_offset = 0, z_offset = 0) -> None:
    """
    Args:
        image: a 3D image with shape [Z, Y, X, 3]
        bbox: a 2D array with shape [N, 6] where N is the number of bounding boxes and 6 is [z1, y1, x1, z2, y2, x2]
    """
    if len(image.shape) == 3:
        image = image[..., np.newaxis]
        image = np.repeat(image, 3, axis=-1)
    
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    if bboxes.dtype != np.int32:
        bboxes = bboxes.astype(np.int32)
    if len(bboxes.shape) == 3:
        bboxes = bboxes.reshape(-1, 6)
    
    max_z, max_y, max_x, _ = image.shape
    bboxes = np.clip(bboxes, 0, [max_z - 1, max_y - 1, max_x - 1, max_z, max_y, max_x])
        
    for bbox in bboxes:
        z1, y1, x1, z2, y2, x2 = bbox
        center_x = (x1 + x2) / 2
        expaned_bbox = bbox.copy()
        if bbox_offset > 0: # expand bbox for better visualization, but only on x and y axis
            expaned_bbox[1:3] = np.maximum(0, expaned_bbox[1:3] - bbox_offset)
            expaned_bbox[4:6] = np.minimum([max_y, max_x], expaned_bbox[4:6] + bbox_offset)
        bboxed_image = draw_bbox(image.copy(), expaned_bbox[np.newaxis, ...], color)
        if z_offset > 0:
            z0 = max(0, z1 - z_offset)
            z3 = min(max_z, z2 + z_offset)
            z_expanded_bbox = expaned_bbox.copy()
            z_expanded_bbox[0] = z0
            z_expanded_bbox[3] = z1
            if color == (255, 0, 0):
                z_color = (0, 255, 0)
            else:
                z_color = (255, 0, 0)
            bboxed_image = draw_bbox(bboxed_image, z_expanded_bbox[np.newaxis, ...], z_color)
            
            z_expanded_bbox[0] = z2
            z_expanded_bbox[3] = z3
            bboxed_image = draw_bbox(bboxed_image, z_expanded_bbox[np.newaxis, ...], z_color)
        # Crop image to save drawing time and space
        if half_image:
            if center_x < image.shape[2] / 2: # left
                bboxed_image = bboxed_image[:, :, :bboxed_image.shape[2] // 2]
            else: # right
                bboxed_image = bboxed_image[:, :, bboxed_image.shape[2] // 2:]
        
        expaned_z1 = max(0, z1 - offset) # expand z axis for better visualization
        expaned_z2 = min(max_z, z2 + offset) # expand z axis for better visualization
        if (expaned_z2 - expaned_z1) > MAX_IMAGES:
            zs = list(range(expaned_z1, expaned_z2, (expaned_z2 - expaned_z1) // MAX_IMAGES))
        else:
            zs = list(range(expaned_z1, expaned_z2))
            
        # Draw
        n_row = max(int(math.sqrt(len(zs))), 1)
        n_col = int(math.ceil(len(zs) / n_row))
        fig = plt.figure(figsize=(int(n_col * SUBPLOT_WIDTH), n_row * SUBPLOT_HEIGHT))
        for i, z in enumerate(zs):
            ax = plt.subplot(n_row, n_col, i+1)
            ax.imshow(bboxed_image[z], cmap='gray')
            if i == len(zs) // 2:
                title = f'z={z}(center)'
            else:
                title = f'z={z}'
            ax.set_title(title)
            if axis_off:
                ax.axis('off')
        sup_title = 'ctrXYZ=({}, {}, {}), whd=({}, {}, {})'.format(int(center_x), int((y1 + y2) / 2), int((z1 + z2) / 2),
                                                        int(x2 - x1), int(y2 - y1), int(z2 - z1))
        if extra_sup_title is not None:
            sup_title += ', ' + extra_sup_title
        plt.suptitle(sup_title)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
def draw_pseu_bbox_and_label_on_image(image: np.ndarray, bboxes: np.ndarray, bboxes_pseu: np.ndarray, color = (0, 255, 0), color_pseu = (255, 0, 0), 
                                      half_image = True, axis_off = True, save_path = None, extra_sup_title = None) -> None:
    """
    """
    if len(image.shape) == 3:
        image = image[..., np.newaxis]
        image = np.repeat(image, 3, axis=-1)
    
    bboxed_image = image.copy()
    for bbox in bboxes:
        bboxed_image = draw_bbox(bboxed_image, bbox[np.newaxis, ...], color)
        
    for bbox_pseu in bboxes_pseu:
        bboxed_image = draw_bbox(bboxed_image, bbox_pseu[np.newaxis, ...], color_pseu)
    
    for bbox in bboxes:
        z1, y1, x1, z2, y2, x2 = bbox
        center_x = (x1 + x2) / 2
        # Crop image to save drawing time and space
        if half_image:
            if center_x < image.shape[2] / 2: # left
                bboxed_image = bboxed_image[:, :, :bboxed_image.shape[2] // 2]
            else: # right
                bboxed_image = bboxed_image[:, :, bboxed_image.shape[2] // 2:]
        
        if (z2 - z1) > MAX_IMAGES:
            zs = list(range(z1, z2, (z2 - z1) // MAX_IMAGES))
        else:
            zs = list(range(z1, z2))
            
        if (z2 - z1) > MAX_IMAGES:
            zs = list(range(z1, z2, (z2 - z1) // MAX_IMAGES))
        else:
            zs = list(range(z1, z2))
        # Draw
        n_row = max(int(math.sqrt(len(zs))), 1)
        n_col = int(math.ceil(len(zs) / n_row))
        fig = plt.figure(figsize=(int(n_col * SUBPLOT_WIDTH), n_row * SUBPLOT_HEIGHT))
        for i, z in enumerate(zs):
            ax = plt.subplot(n_row, n_col, i+1)
            ax.imshow(bboxed_image[z], cmap='gray')
            ax.set_title(f'z={z}')
            if axis_off:
                ax.axis('off')
        sup_title = 'ctrXYZ=({}, {}, {}), whd=({}, {}, {})'.format(int(center_x), int((y1 + y2) / 2), int((z1 + z2) / 2),
                                                                    int(x2 - x1), int(y2 - y1), int(z2 - z1))
        if extra_sup_title is not None:
            sup_title += ', ' + extra_sup_title
        plt.suptitle(sup_title)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        plt.close(fig)
        
    if save_path is None:
        for bbox_pseu in bboxes_pseu:
            z1, y1, x1, z2, y2, x2 = bbox_pseu
            if (z2 - z1) > MAX_IMAGES:
                zs = list(range(z1, z2, (z2 - z1) // MAX_IMAGES))
            else:
                zs = list(range(z1, z2))
            # Draw
            n_row = max(int(math.sqrt(len(zs))), 1)
            n_col = int(math.ceil(len(zs) / n_row))
            fig = plt.figure(figsize=(int(n_col * SUBPLOT_WIDTH), n_row * SUBPLOT_HEIGHT))
            for i, z in enumerate(zs):
                ax = plt.subplot(1, len(zs), i+1)
                ax.imshow(bboxed_image[z], cmap='gray')
                ax.set_title(f'z={z} pseu')
                if axis_off:
                    ax.axis('off')
            sup_title = 'ctrXYZ=({}, {}, {}), whd=({}, {}, {})'.format(int(center_x), int((y1 + y2) / 2), int((z1 + z2) / 2),
                                                            int(x2 - x1), int(y2 - y1), int(z2 - z1))
            if extra_sup_title is not None:
                sup_title += ', ' + extra_sup_title
            plt.suptitle(sup_title)
            plt.tight_layout()
            plt.close(fig)
        
def draw_bbox_on_image_with_label(img_path: str, label_path: str, color = (255, 0, 0)) -> np.ndarray:
    """
    Args:
        img_path: path to the image
        label_path: path to the label
    """
    image = load_image(img_path)
    image = image * 2 - 1 # normalize to [-1, 1]
    
    label = load_image(load_label(label_path, np.array([1.0, 1.0, 1.0]))) # use pixel rather than physical spacing
    
    nodule_findings = label2nodulefinding(label)
    bboxes = noduleFinding2cude(nodule_findings, image.shape)
    draw_bbox_on_image(image, bboxes, color)
    
    
if __name__ == '__main__':
    # draw FN
    ME_ROOT = r'D:\workspace\medical_dataset\ME_dataset'
    LDCT_ROOT = r'D:\workspace\medical_dataset\LDCT_test_dataset'

    def get_dicom_and_label_path(series_name: str) -> Tuple[str, str]:
        """
        Return:
            A tuple of (dicom_path, label_path)
        """
        if 'Test' in series_name:
            root = LDCT_ROOT
        else:
            root = ME_ROOT
        folder = os.path.join(root, series_name) 
        # load dicom and label
        dicom_path = gen_dicom_path(folder, series_name)
        label_path = gen_label_path(folder, series_name)
        return dicom_path, label_path

    # load FN series list
    # csv_path = './save/[2024-02-14-1706]_all_ME_bs3_ns5/annotation/[2024-02-14-1706]_all_ME_bs3_ns5/annotation/predict_epoch_160/FN_0.1.txt'
    csv_path = r'D:\workspace\python\My_CPM_Net\save\[2024-02-14-1706]_all_ME_bs3_ns5\val_temp\annotation\epoch_0\FN_0.1.csv'
    with open(csv_path, 'r') as f:
        if csv_path.endswith('.csv'):
            lines = f.readlines()[1:] # skip header
            fn_nodules_infos = [line.strip().split(',') for line in lines]
        else:
            lines = f.readlines()
            fn_nodules_infos = [line.strip().split(',') for line in lines]
            temp = []
            for n in fn_nodules_infos:
                temp.append([n[0].replace('_crop.npy', ''), n[2], n[3], n[4], n[5], n[6], n[7]])
            fn_nodules_infos = temp
    fn_dicom_paths = []
    fn_nodule_findings = []
    for i, nodule in enumerate(fn_nodules_infos):
        series_name, x, y, z, w, h, d = nodule
        dicom_path, label_path = get_dicom_and_label_path(series_name)
        fn_dicom_paths.append(dicom_path)
        nodule_finding = NoduleFinding(coordX=x, coordY=y, coordZ=z, w=w, h=h, d=d)
        fn_nodule_findings.append(nodule_finding)
        
    for idx in range(len(fn_dicom_paths)):
        fn_nodule_finding = fn_nodule_findings[idx]
        if fn_nodule_finding.d >= 3:
            continue
        image = load_image(fn_dicom_paths[idx])
        image = image * 2 - 1

        print(str(fn_nodule_finding))
        fn_bboxes = noduleFinding2cude(fn_nodule_finding, image.shape)

        mapped_image = ((image + 1) * 127.5).astype(np.uint8)
        # copy 3D image to 3 channels
        mapped_image = np.stack([mapped_image, mapped_image, mapped_image], axis=-1) # [Z, Y, X, 3]
        draw_bbox_on_image(mapped_image, fn_bboxes, color=(255, 0, 0))