import numpy as np
from typing import Tuple

class AbstractCTRTransform(object):
    def __init__(self, params):
        pass
    def forward_ctr(self, ctr):
        return ctr
    
    def backward_ctr(self, ctr):
        return ctr
    
    def forward_rad(self, coords):
        return coords
    
    def backward_rad(self, coords):
        return coords
    
    def forward_spacing(self, spacing):
        return spacing

    def backward_spacing(self, spacing):
        return spacing

class OffsetMinusCTR(AbstractCTRTransform):
    def __init__(self, offset: np.ndarray):
        if not isinstance(offset, np.ndarray):
            offset = np.array([offset])
        self.offset = offset
        self.axes = np.where(offset != 0)[0]
        
    def forward_ctr(self, ctr):
        """new = offset - old
        """
        if len(ctr) == 0:
            return ctr
        
        new_ctr = ctr.copy()
        new_ctr[:, self.axes] = self.offset[self.axes] - ctr[:, self.axes]
        return new_ctr
    
    def backward_ctr(self, ctr):
        """old = offset - new
        """
        return self.forward_ctr(ctr)

class OffsetPlusCTR(AbstractCTRTransform):
    def __init__(self, offset: np.ndarray):
        if not isinstance(offset, np.ndarray):
            offset = np.array([offset])
        self.offset = offset
        
    def forward_ctr(self, ctr):
        """new = offset + old
        """
        if len(ctr) == 0:
            return ctr
        return ctr + self.offset
    
    def backward_ctr(self, ctr):
        """old = new - offset
        """
        if len(ctr) == 0:
            return ctr
        return ctr - self.offset    

class TransposeCTR(AbstractCTRTransform):
    def __init__(self, transpose_order: Tuple[int]):
        if not isinstance(transpose_order, np.ndarray):
            transpose_order = np.array([transpose_order], dtype=np.int32)
        self.transpose_order = transpose_order.astype(np.int32)
        
    def forward_ctr(self, ctr):
        if len(ctr) == 0:
            return ctr
        elif len(ctr.shape) == 1:
            return ctr[self.transpose_order]
        else:
            return ctr[..., self.transpose_order]
    
    def backward_ctr(self, ctr):
        if len(ctr) == 0:
            return ctr
        elif len(ctr.shape) == 1:
            return ctr[np.argsort(self.transpose_order)]
        else:
            return ctr[..., np.argsort(self.transpose_order)]

    def forward_rad(self, rads):
        return self.forward_ctr(rads)
    
    def backward_rad(self, rads):
        return self.backward_ctr(rads)
    
    def forward_spacing(self, spacing):
        return spacing[self.transpose_order]
    
    def backward_spacing(self, spacing):
        return spacing[np.argsort(self.transpose_order)]

class RotateCTR(AbstractCTRTransform):
    def __init__(self, angle: float, axes: Tuple[int, int], image_shape: np.ndarray):
        self.angle = angle
        self.radian = np.deg2rad(angle)
        self.image_center = np.array(image_shape) / 2
        self.cos = np.cos(self.radian)
        self.sin = np.sin(self.radian)
        self.axes = axes
        
        if self.angle % 90 != 0:
            raise ValueError("angle must be multiple of 90")
        
    def forward_ctr(self, ctr):
        if len(ctr) == 0:
            return ctr
        new_ctr = ctr.copy()
        new_ctr[:, self.axes[0]] = (ctr[:, self.axes[0]] - self.image_center[self.axes[0]]) * self.cos - (ctr[:, self.axes[1]] - self.image_center[self.axes[1]]) * self.sin + self.image_center[self.axes[0]]
        new_ctr[:, self.axes[1]] = (ctr[:, self.axes[0]] - self.image_center[self.axes[0]]) * self.sin + (ctr[:, self.axes[1]] - self.image_center[self.axes[1]]) * self.cos + self.image_center[self.axes[1]]
        return new_ctr
    
    def backward_ctr(self, ctr):
        if len(ctr) == 0:
            return ctr
        new_ctr = ctr.copy()
        new_ctr[:, self.axes[0]] = (ctr[:, self.axes[0]] - self.image_center[self.axes[0]]) * self.cos + (ctr[:, self.axes[1]] - self.image_center[self.axes[1]]) * self.sin + self.image_center[self.axes[0]]
        new_ctr[:, self.axes[1]] = (ctr[:, self.axes[0]] - self.image_center[self.axes[0]]) * -self.sin + (ctr[:, self.axes[1]] - self.image_center[self.axes[1]]) * self.cos + self.image_center[self.axes[1]]
        return new_ctr
    
    def forward_rad(self, rads):
        if len(rads) == 0:
            return rads
        if self.angle == 90 or self.angle == 270:
            new_coords = rads.copy()
            new_coords[:, self.axes[0]] = rads[:, self.axes[1]]
            new_coords[:, self.axes[1]] = rads[:, self.axes[0]]
            return new_coords
        else:
            return rads
        
    def backward_rad(self, coords):
        return self.forward_rad(coords)
    
    def forward_spacing(self, spacing):
        if self.angle == 90 or self.angle == 270:
            new_spacing = spacing.copy()
            new_spacing[self.axes[0]] = spacing[self.axes[1]]
            new_spacing[self.axes[1]] = spacing[self.axes[0]]
            return new_spacing
        else:
            return spacing
    
    def backward_spacing(self, spacing):
        return self.forward_spacing(spacing)