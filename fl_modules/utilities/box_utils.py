import torch

import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple

def zyxdhw2zyxzyx(box, dim=-1):
    ctr_zyx, dhw = torch.split(box, 3, dim)
    z1y1x1 = ctr_zyx - dhw/2
    z2y2x2 = ctr_zyx + dhw/2
    return torch.cat((z1y1x1, z2y2x2), dim)  # zyxzyx bbox

def bbox_decode(anchor_points: torch.Tensor, pred_offsets: torch.Tensor, pred_shapes: torch.Tensor, stride_tensor: torch.Tensor, dim=-1) -> torch.Tensor:
    """Apply the predicted offsets and shapes to the anchor points to get the predicted bounding boxes.
    anchor_points is the center of the anchor boxes, after applying the stride, new_center = (center + pred_offsets) * stride_tensor
    Args:
        anchor_points: torch.Tensor
            A tensor of shape (bs, num_anchors, 3) containing the coordinates of the anchor points, each of which is in the format (z, y, x).
        pred_offsets: torch.Tensor
            A tensor of shape (bs, num_anchors, 3) containing the predicted offsets in the format (dz, dy, dx).
        pred_shapes: torch.Tensor
            A tensor of shape (bs, num_anchors, 3) containing the predicted shapes in the format (d, h, w).
        stride_tensor: torch.Tensor
            A tensor of shape (bs, 3) containing the strides of each dimension in format (z, y, x).
    Returns:
        A tensor of shape (bs, num_anchors, 6) containing the predicted bounding boxes in the format (z, y, x, d, h, w).
    """
    center_zyx = (anchor_points + pred_offsets) * stride_tensor
    return torch.cat((center_zyx, 2*pred_shapes), dim)  # zyxdhw bbox

def make_anchors(feat: torch.Tensor, input_size: List[float], grid_cell_offset=0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate anchors from a feature.
    Returns:
        num_anchor is the number of anchors in the feature map, which is d * h * w.
        A tuple of two tensors:
            anchor_points: torch.Tensor
                A tensor of shape (num_anchors, 3) containing the coordinates of the anchor points, each of which is in the format (z, y, x).
            stride_tensor: torch.Tensor
                A tensor of shape (num_anchors, 3) containing the strides of the anchor points, the strides of each anchor point are same.
    """
    dtype, device = feat.dtype, feat.device
    _, _, d, h, w = feat.shape
    strides = torch.tensor([input_size[0] / d, input_size[1] / h, input_size[2] / w], dtype=dtype, device=device)
    sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
    sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
    sz = torch.arange(end=d, device=device, dtype=dtype) + grid_cell_offset  # shift z
    anchor_points = torch.cartesian_prod(sz, sy, sx)
    stride_tensor = strides.repeat(d * h * w, 1)
    return anchor_points, stride_tensor

def nms_3D(dets: NDArray[np.float32], overlap=0.5, top_k=200):
    """
    Args:
        dets: 
            (N, 7) [prob, ctr_z, ctr_y, ctr_x, d, h, w]
        overlap: 
            iou threshold
        top_k: 
            keep top_k results
    """
    # det {prob, ctr_z, ctr_y, ctr_x, d, h, w}
    dd, hh, ww = dets[:, 4], dets[:, 5], dets[:, 6]
    z1 = dets[:, 1] - 0.5 * dd
    y1 = dets[:, 2] - 0.5 * hh
    x1 = dets[:, 3] - 0.5 * ww
    
    z2 = dets[:, 1] + 0.5 * dd
    y2 = dets[:, 2] + 0.5 * hh
    x2 = dets[:, 3] + 0.5 * ww
    
    scores = dets[:, 0]
    areas = dd * hh * ww
    
    _, idx = scores.sort(0, descending=True)
    keep = []
    while idx.size(0) > 0:
        i = idx[0]
        keep.append(int(i.cpu().numpy()))
        if idx.size(0) == 1 or len(keep) == top_k:
            break
        xx1 = torch.max(x1[idx[1:]], x1[i].expand(len(idx)-1))
        yy1 = torch.max(y1[idx[1:]], y1[i].expand(len(idx)-1))
        zz1 = torch.max(z1[idx[1:]], z1[i].expand(len(idx)-1))

        xx2 = torch.min(x2[idx[1:]], x2[i].expand(len(idx)-1))
        yy2 = torch.min(y2[idx[1:]], y2[i].expand(len(idx)-1))
        zz2 = torch.min(z2[idx[1:]], z2[i].expand(len(idx)-1))

        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        d = torch.clamp(zz2 - zz1, min=0.0)

        inter = w*h*d
        IoU = inter / (areas[i] + areas[idx[1:]] - inter)
        inds = IoU <= overlap
        idx = idx[1:][inds]
    return torch.from_numpy(np.array(keep))

def iou_3D(box1, box2):
    # need z_ctr, y_ctr, x_ctr, d
    z1 = np.maximum(box1[0] - 0.5 * box1[3], box2[0] - 0.5 * box2[3])
    y1 = np.maximum(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
    x1 = np.maximum(box1[2] - 0.5 * box1[3], box2[2] - 0.5 * box2[3])

    z2 = np.minimum(box1[0] + 0.5 * box1[3], box2[0] + 0.5 * box2[3])
    y2 = np.minimum(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3])
    x2 = np.minimum(box1[2] + 0.5 * box1[3], box2[2] + 0.5 * box2[3])

    w = np.maximum(x2 - x1, 0.)
    h = np.maximum(y2 - y1, 0.)
    d = np.maximum(z2 - z1, 0.)

    inters = w * h * d
    uni = box1[3] * box1[3] * box1[3] + box2[3] * box2[3] * box2[3] - inters
    uni = np.maximum(uni, 1e-8)
    ious = inters / uni
    return ious
