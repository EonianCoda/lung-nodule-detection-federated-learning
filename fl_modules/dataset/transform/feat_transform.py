import numpy as np
from typing import Tuple
import torch
import logging

class AbstractFeatTransform(object):
    def __init__(self, params):
        pass
    def forward(self, feat):
        return feat
    
    def backward(self, feat):
        return feat

class FlipFeatTransform(AbstractFeatTransform):
    def __init__(self, flip_axes: Tuple[int]):
        self.flip_axes = list(flip_axes)
        for i in self.flip_axes:
            if i > 0:
                raise ValueError("flip_axes should be negative index")
        
    def forward(self, feat, sign_value=False):
        if sign_value:
            sign = [1, 1, 1]
            for i in self.flip_axes:
                sign[i] = -1
            sign = np.array(sign)
        if isinstance(feat, np.ndarray):
            if sign_value:
                for _ in range(3):
                    sign = np.expand_dims(sign, -1)
                if len(feat.shape) > 4:
                    for _ in range(len(feat.shape) - 4):
                        sign = np.expand_dims(sign, 0)
                return np.flip(feat * sign, self.flip_axes) 
            else:
                return np.flip(feat, self.flip_axes)
        elif isinstance(feat, torch.Tensor):
            if sign_value:
                sign = torch.from_numpy(sign).to(feat.device).to(feat.dtype)
                for _ in range(3):
                    sign = sign.unsqueeze(-1)
                if len(feat.shape) > 4:
                    for _ in range(len(feat.shape) - 4):
                        sign = sign.unsqueeze(0)
                return torch.flip(feat * sign, self.flip_axes)
            else:
                return torch.flip(feat, self.flip_axes)
    
    def backward(self, feat, sign_value=False):
        return self.forward(feat, sign_value)

class Rot90FeatTransform(AbstractFeatTransform):
    def __init__(self, rot_angle: int, rot_axes: Tuple[int]):
        self.rot_axes = list(rot_axes)
        self.rot_angle = rot_angle
        for i in self.rot_axes:
            if i > 0:
                raise ValueError("rot_axes should be negative index")
        assert rot_angle % 90 == 0, "rot_angle must be multiple of 90"
        
    def forward(self, feat):
        if isinstance(feat, np.ndarray):
            return np.rot90(feat,  self.rot_angle // 90, self.rot_axes)
        elif isinstance(feat, torch.Tensor):
            return torch.rot90(feat, self.rot_angle // 90, self.rot_axes)
    
    def backward(self, feat):
        if isinstance(feat, np.ndarray):
            rot_feat = np.rot90(feat, -self.rot_angle // 90, self.rot_axes)
            if len(feat.shape) == 4 and feat.shape[0] == 3: # 4 is (channel, depth, height, width), 3 is d, h, w or z, y, x
                # Change rotation axes
                rot_axis1 = self.rot_axes[0]
                rot_axis2 = self.rot_axes[1]
                # swap axis in the channel dimension
                rot_feat[[rot_axis1, rot_axis2]] = rot_feat[[rot_axis2, rot_axis1]]
            return rot_feat
        elif isinstance(feat, torch.Tensor):
            rot_feat = torch.rot90(feat, -self.rot_angle // 90, self.rot_axes)
            if len(feat.shape) == 4 and feat.shape[0] == 3:
                # Change rotation axes
                rot_axis1 = self.rot_axes[0]
                rot_axis2 = self.rot_axes[1]
                # swap axis in the channel dimension
                rot_feat[[rot_axis1, rot_axis2]] = rot_feat[[rot_axis2, rot_axis1]]
            return rot_feat

class TransposeFeatTransform(AbstractFeatTransform):
    def __init__(self, transpose_order: Tuple[int]):
        self.transpose_order = np.array(transpose_order)
        
    def forward(self, feat):
        assert len(feat.shape) >= 3
        transpose_order = np.arange(len(feat.shape))
        transpose_order[-3:] = self.transpose_order
        transpose_order = transpose_order.tolist()
        if isinstance(feat, np.ndarray):
            return np.transpose(feat, transpose_order)
        elif isinstance(feat, torch.Tensor):
            return feat.permute(transpose_order)
        
    def backward(self, feat):
        assert len(feat.shape) >= 3
        transpose_order = np.arange(len(feat.shape))
        
        transpose_order[-3:] = np.argsort(self.transpose_order) + len(feat.shape) - 3
        transpose_order = transpose_order.tolist()
        if isinstance(feat, np.ndarray):
            return np.transpose(feat, transpose_order)
        elif isinstance(feat, torch.Tensor):
            return feat.permute(transpose_order)