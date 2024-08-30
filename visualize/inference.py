import os
import math
from typing import List
import numpy as np
import torch

from networks.ResNet_3D_CPM import DetectionPostprocess
from utils.box_utils import nms_3D
from logic.utils import load_model
from evaluationScript.nodule_finding_original import NoduleFinding

from dataload.split_combine import SplitComb
from dataload.utils import load_image
from .convert import output2nodulefinding

class Inference:
    def __init__(self, model, 
                 device: torch.device, 
                 crop_size = [64, 128, 128], 
                 overlap_ratio = 0.25,
                 det_threshold = 0.7,
                 final_mns_topk = 40,
                 val_mixed_precision = False,
                 batch_size = 16):
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.final_mns_topk = final_mns_topk
        
        self.device = device
        self.val_mixed_precision = val_mixed_precision
        # Build model
        if isinstance(model, str):
            self.model = load_model(model)
        else:
            self.model = model
        self.model.to(self.device)
        self.model.eval()
        
        self.overlap_size = [int(s * overlap_ratio) for s in crop_size]
        self.split_comber = SplitComb(self.crop_size, self.overlap_size, pad_value=-1)

        self.detection_postprocess = DetectionPostprocess(topk=60, threshold=det_threshold, nms_threshold=0.05, nms_topk=20, crop_size=self.crop_size)
        
    def _predict(self, data) -> List[np.ndarray]:
        patch_outputs = []
        data = data.to(self.device)
        for i in range(int(math.ceil(data.size(0) / self.batch_size))):
            end = (i+1) * self.batch_size
            if end > data.size(0):
                end = data.size(0)
            if self.val_mixed_precision:
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        output = self.model(data[i * self.batch_size:end])
                        output = self.detection_postprocess(output, device=self.device) #1, prob, ctr_z, ctr_y, ctr_x, d, h, w
            else:
                with torch.no_grad():
                    output = self.model(data[i * self.batch_size:end])
                    output = self.detection_postprocess(output, device=self.device) #1, prob, ctr_z, ctr_y, ctr_x, d, h, w
            patch_outputs.append(output.data.cpu().numpy())
        del data
        return patch_outputs
    
    def predict(self, img_path:int) -> List[NoduleFinding]:
        # Load image
        image = load_image(img_path)
        image = image * 2.0 - 1.0 # convert to -1 ~ 1  note ste pad_value to -1 for SplitComb
        
        # Split image
        split_images, nzhw = self.split_comber.split(image) # split_images [N, 1, crop_z, crop_y, crop_x]
        data = torch.from_numpy(split_images)
        
        # Predict
        patch_outputs = self._predict(data)
        
        # Combine patch outputs
        output = np.concatenate(patch_outputs, 0)
        output = self.split_comber.combine(output, nzhw=nzhw)
        output = torch.from_numpy(output).view(-1, 8)
        object_ids = output[:, -1] != -1.0
        output = output[object_ids]
        if len(output) > 0:
            keep = nms_3D(output[:, 1:], overlap=0.05, top_k=self.final_mns_topk)
            output = output[keep]
        output = output.numpy()[:, 1:] # remove id
        
        # Convert to nodule finding
        nodules = output2nodulefinding(output)
        return image, nodules