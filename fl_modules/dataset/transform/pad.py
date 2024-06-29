# -*- coding: utf-8 -*-
from __future__ import print_function, division
import numpy as np
import json
import math
from .abstract_transform import AbstractTransform
from .ctr_transform import OffsetPlusCTR
class Pad(AbstractTransform):
    """
    Pad the image (shape [C, D, H, W] or [C, H, W]) to an new spatial shape, 
    the real output size will be max(image_size, output_size)
    """

    def __init__(self, output_size, ceil_mode=False, pad_value: float = 0):
        """
        output_size (tuple/list): the size along each spatial axis. 
        ceil_mode (bool): if true, the real output size is integer multiples of output_size.
        """
        self.output_size = output_size
        self.ceil_mode = ceil_mode
        self.pad_value = pad_value

    def __call__(self, sample):
        image = sample['image']
        input_shape = image.shape
        
        # if the input shape is already the same as the output shape, return the sample
        if input_shape[1] == self.output_size[0] and input_shape[2] == self.output_size[1] and input_shape[3] == self.output_size[2]:
            return sample
        
        input_dim = len(input_shape) - 1
        assert (len(self.output_size) == input_dim)
        if (self.ceil_mode):
            multiple = [int(math.ceil(float(input_shape[1 + i]) / self.output_size[i])) \
                        for i in range(input_dim)]
            output_size = [multiple[i] * self.output_size[i] \
                           for i in range(input_dim)]
        else:
            output_size = self.output_size
        margin = [max(0, output_size[i] - input_shape[1 + i]) for i in range(input_dim)]
        margin_lower = [int(margin[i] / 2) for i in range(input_dim)]
        margin_upper = [margin[i] - margin_lower[i] for i in range(input_dim)]
        pad = [(margin_lower[i], margin_upper[i]) for i in range(input_dim)]
        pad = tuple([(0, 0)] + pad)

        image_t = np.pad(image, pad, 'constant', constant_values=self.pad_value) if (max(margin) > 0) else image
        sample['image'] = image_t
        sample['ctr_transform'].append(OffsetPlusCTR(margin_lower))
        ##TODO add feature transform
        if 'ctr' in sample and len(sample['ctr']) > 0 and np.any(np.array(margin_lower) > 0):
            sample['ctr'] = sample['ctr'].copy() + margin_lower
        return sample