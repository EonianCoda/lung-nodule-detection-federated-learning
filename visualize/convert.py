from typing import List, Tuple, Dict

import numpy as np
from fl_modules.dataset.utils import ALL_LOC, ALL_RAD, ALL_PROB
from fl_modules.eval.nodule_finding import NoduleFinding
from fl_modules.eval.nodule_typer import NoduleTyper
from typing import List, Tuple

# def output2nodulefinding(output: np.ndarray) -> List[NoduleFinding]:
#     pred_nodules = []
#     for n in output:
#         prob, z, y, x, d, h, w = n
#         nodule_finding = NoduleFinding(coordX=x, coordY=y, coordZ=z, w=w, h=h, d=d, CADprobability=prob)
#         nodule_finding.auto_nodule_type()
#         pred_nodules.append(nodule_finding)
#     return pred_nodules

# def label2nodulefinding(label: Dict[str, np.ndarray]) -> List[NoduleFinding]:
#     """
#     Args:
#         label: a dictionary with keys 'all_loc' and 'all_rad'
#     """
#     nodules = []
#     loc = label[ALL_LOC]
#     rad = label[ALL_RAD]
#     nodule_typer = NoduleTyper()
#     for (z, y, x), (d, h, w), r in zip(loc, rad, rad):
#         nodule_finding = NoduleFinding(ctr_x=x, ctr_y=y, ctr_z=z, w=w, h=h, d=d, prob=1.0, is_gt=True)
#         nodules.append(nodule_finding)
#     return nodules

def annot2bboxes(annot):
    annot = annot[annot[:, -1] != -1]
    if len(annot) == 0:
        return np.zeros((0, 6))
    else:
        # ctr_z, ctr_y, ctr_x, d, h, w
        bboxes = np.stack([annot[:, :3] - annot[:, 3:6] / 2, annot[:, :3] + annot[:, 3:6] / 2], axis=1)
        bboxes = bboxes.reshape(-1, 6)
        return bboxes

def label2bboxes(label: dict, conf_threshold=0.0) -> np.ndarray:
    """
    Return: A numpy array of shape (N, 2, 3) where N is the number of nodules
    """
    ctr = label[ALL_LOC]
    rad = label[ALL_RAD]
    if ALL_PROB in label:
        conf = label[ALL_PROB]
        ctr = ctr[conf >= conf_threshold]
        rad = rad[conf >= conf_threshold]
    if len(ctr) != 0:
        return np.stack([ctr - rad / 2, ctr + rad / 2], axis=1) # (N, 2, 3)
    else:
        return np.zeros((0, 2, 3))

def noduleFinding2cude(nodules: List[NoduleFinding], shape: Tuple[int, int, int]) -> np.ndarray:
    if not isinstance(nodules, list):
        nodules = [nodules]
    bboxes = []
    for nodule in nodules:
        z, y, x, d, h, w = nodule.ctr_z, nodule.ctr_y, nodule.ctr_x, nodule.d, nodule.h, nodule.w
        z1 = max(round(z - d/2), 0)
        y1 = max(round(y - h/2), 0)
        x1 = max(round(x - w/2), 0)

        z2 = min(round(z + d/2), shape[0] - 1)
        y2 = min(round(y + h/2), shape[1] - 1)
        x2 = min(round(x + w/2), shape[2] - 1)
        bboxes.append((z1, y1, x1, z2, y2, x2))
    bboxes = np.array(bboxes, dtype=np.int32)
    return bboxes

def gtNoduleFinding2cube(nodules: List[NoduleFinding], shape: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return: A tuple of [pred_bboxes, gt_bboxes]
    """
    if not isinstance(nodules, list):
        nodules = [nodules]

    pred_bboxes = []
    gt_bboxes = []
    for nodule in nodules:
        # Get predicted nodule
        pred_ctr_x, pred_ctr_y, pred_ctr_z, pred_w, pred_h, pred_d = nodule.ctr_x, nodule.ctr_y, nodule.ctr_z, nodule.w, nodule.h, nodule.d
        
        pred_z1 = max(round(pred_ctr_z - pred_d/2), 0)
        pred_y1 = max(round(pred_ctr_y - pred_h/2), 0)
        pred_x1 = max(round(pred_ctr_x - pred_w/2), 0)
        
        pred_z2 = min(round(pred_ctr_z + pred_d/2), shape[0] - 1)
        pred_y2 = min(round(pred_ctr_y + pred_h/2), shape[1] - 1)
        pred_x2 = min(round(pred_ctr_x + pred_w/2), shape[2] - 1)
        
        pred_bboxes.append((pred_z1, pred_y1, pred_x1, pred_z2, pred_y2, pred_x2))
        # Get ground truth nodule
        gt_nodule = nodule.match_nodule_finding
        gt_ctr_x, gt_ctr_y, gt_ctr_z, gt_w, gt_h, gt_d = gt_nodule.ctr_x, gt_nodule.ctr_y, gt_nodule.ctr_z, gt_nodule.w, gt_nodule.h, gt_nodule.d
        
        gt_z1 = max(round(gt_ctr_z - gt_d/2), 0)
        gt_y1 = max(round(gt_ctr_y - gt_h/2), 0)
        gt_x1 = max(round(gt_ctr_x - gt_w/2), 0)
        
        gt_z2 = min(round(gt_ctr_z + gt_d/2), shape[0] - 1)
        gt_y2 = min(round(gt_ctr_y + gt_h/2), shape[1] - 1)
        gt_x2 = min(round(gt_ctr_x + gt_w/2), shape[2] - 1)
        
        gt_bboxes.append((gt_z1, gt_y1, gt_x1, gt_z2, gt_y2, gt_x2))
    pred_bboxes = np.array(pred_bboxes, dtype=np.int32)
    gt_bboxes = np.array(gt_bboxes, dtype=np.int32)
    return pred_bboxes, gt_bboxes