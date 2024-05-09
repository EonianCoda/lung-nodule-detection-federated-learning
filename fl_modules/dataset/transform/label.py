import numpy as np
from .abstract_transform import AbstractTransform

class CoordToAnnot(AbstractTransform):
    """Convert one-channel label map to one-hot multi-channel probability map"""
    def __call__(self, sample):
        ctr = sample['ctr']
        rad = sample['rad']
        cls = sample['cls']
        
        spacing = sample['spacing']
        n = ctr.shape[0]
        spacing = np.tile(spacing, (n, 1))
        
        annot = np.concatenate([ctr, rad.reshape(-1, 3), spacing.reshape(-1, 3), cls.reshape(-1, 1)], axis=-1).astype('float32') # (n, 10)

        sample['annot'] = annot
        return sample

class ClassificationCoordToAnnot(AbstractTransform):
    """Convert one-channel label map to one-hot multi-channel probability map"""
    def __call__(self, sample):
        ctr = sample['ctr']
        rad = sample['rad']
        cls = sample['cls']
        
        spacing = sample['spacing']
        n = ctr.shape[0]
        spacing = np.tile(spacing, (n, 1))
        
        annot = np.concatenate([ctr, rad.reshape(-1, 3), spacing.reshape(-1, 3), cls.reshape(-1, 5)], axis=-1).astype('float32') # (n, 14)

        sample['annot'] = annot
        return sample

class SemiCoordToAnnot(AbstractTransform):
    """Convert one-channel label map to one-hot multi-channel probability map"""
    def __call__(self, sample):
        ctr = sample['ctr']
        rad = sample['rad']
        cls = sample['cls']
        
        gt_ctr = sample['gt_ctr']
        gt_rad = sample['gt_rad']
        gt_cls = sample['gt_cls']
        
        spacing = sample['spacing']
        
        annot = np.concatenate([ctr, rad.reshape(-1, 3), np.tile(spacing, (ctr.shape[0], 1)).reshape(-1, 3), cls.reshape(-1, 1)], axis=-1).astype('float32') # (n, 10)
        gt_annot = np.concatenate([gt_ctr, gt_rad.reshape(-1, 3), np.tile(spacing, (gt_ctr.shape[0], 1)).reshape(-1, 3), gt_cls.reshape(-1, 1)], axis=-1).astype('float32') # (n, 10)
        sample['annot'] = annot
        sample['gt_annot'] = gt_annot
        return sample
