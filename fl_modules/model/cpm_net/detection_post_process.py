from typing import List
import torch.nn as nn
import torch
from fl_modules.utilities.box_utils import nms_3D, bbox_decode, make_anchors

class DetectionPostprocess(nn.Module):
    def __init__(self, topk: int=60, threshold: float=0.15, nms_threshold: float=0.05, nms_topk: int=20, crop_size: List[int]=[96, 96, 96], min_size: int=-1):
        super(DetectionPostprocess, self).__init__()
        self.topk = topk
        self.threshold = threshold
        self.nms_threshold = nms_threshold
        self.nms_topk = nms_topk
        self.crop_size = crop_size
        self.min_size = min_size

    def forward(self, output, device, is_logits=True, threshold=None, nms_topk=None, lobe_mask=None):
        Cls = output['Cls']
        Shape = output['Shape']
        Offset = output['Offset']
        batch_size = Cls.size()[0]
        anchor_points, stride_tensor = make_anchors(Cls, self.crop_size, 0)
        
        # Apply lobe
        if lobe_mask is not None:
            if is_logits:
                Cls[lobe_mask == 0] = -20 # ignore the lobe 0, -20 indicates the background, sigmoid(-20) is close to 0
            else:
                Cls[lobe_mask == 0] = 1e-4
        
        # view shape
        pred_scores = Cls.view(batch_size, 1, -1)
        pred_shapes = Shape.view(batch_size, 3, -1)
        pred_offsets = Offset.view(batch_size, 3, -1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        if is_logits:
            pred_scores = pred_scores.sigmoid()
        pred_shapes = pred_shapes.permute(0, 2, 1).contiguous()
        pred_offsets = pred_offsets.permute(0, 2, 1).contiguous()
        
        # recale to input_size
        pred_bboxes = bbox_decode(anchor_points, pred_offsets, pred_shapes, stride_tensor)
        # Get the topk scores and indices
        topk_scores, topk_indices = torch.topk(pred_scores.squeeze(), self.topk, dim=-1, largest=True)
        
        dets = (-torch.ones((batch_size, self.topk, 8))).to(device)
        for j in range(batch_size):
            # Get indices of scores greater than threshold
            topk_score = topk_scores[j]
            topk_idx = topk_indices[j]
            if threshold is not None:
                keep_box_mask = (topk_score > threshold)
            else:
                keep_box_mask = (topk_score > self.threshold)
            keep_box_n = keep_box_mask.sum()
            
            if keep_box_n > 0:
                keep_topk_score = topk_score[keep_box_mask]
                keep_topk_idx = topk_idx[keep_box_mask]
                
                # 1, prob, ctr_z, ctr_y, ctr_x, d, h, w
                det = (-torch.ones((keep_box_n, 8))).to(device)
                det[:, 0] = 1
                det[:, 1] = keep_topk_score
                det[:, 2:] = pred_bboxes[j][keep_topk_idx]
                if nms_topk is not None:
                    keep = nms_3D(det[:, 1:], overlap=self.nms_threshold, top_k=nms_topk)
                else:
                    keep = nms_3D(det[:, 1:], overlap=self.nms_threshold, top_k=self.nms_topk)
                dets[j][:len(keep)] = det[keep.long()]

        if self.min_size > 0:
            dets_volumes = dets[:, :, 5] * dets[:, :, 6] * dets[:, :, 7]
            dets[dets_volumes < self.min_size] = -1
                
        return dets