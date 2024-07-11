# -*- coding: utf-8 -*-
from __future__ import print_function, division

from .abstract_transform import AbstractTransform
from scipy import ndimage
from skimage.util import random_noise
import random
import numpy as np
import cv2

class RandomIntensity(AbstractTransform):
    def __init__(self, p=0.5):
        self.random_blur = RandomBlur(sigma_range=(0.2, 0.6), p=1.0)
        self.random_gamma = RandomGamma(gamma_range=[0.92, 1.08], p=1.0)
        self.intensity_transforms = [self.random_blur, self.random_gamma]
        self.p = p
    def __call__(self, sample):
        if random.random() < self.p:
            aug_idx = np.random.choice(len(self.intensity_transforms), 1)[0]
            sample = self.intensity_transforms[aug_idx](sample)
        return sample

class RandomBlur(AbstractTransform):
    """
    Randomly applies Gaussian blur to the input image.

    Args:
        sigma_range (tuple): Range of sigma values for Gaussian blur. Default is (0.4, 0.8).
        p (float): Probability of applying the transform. Default is 0.5.
        channel_apply (int): Index of the channel to apply the transform on. Default is 0.

    Returns:
        dict: Transformed sample with the blurred image.

    """
    def __init__(self, sigma_range=(0.4, 0.8), p=0.5):
        """
        Initializes the RandomBlur transform.

        Args:
            sigma_range (tuple): Range of sigma values for Gaussian blur. Default is (0.4, 0.8).
            p (float): Probability of applying the transform. Default is 0.5.
        """
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, sample):
        """
        Applies the RandomBlur transform to the input sample.

        Args:
            sample (dict): Input sample containing the image.

        Returns:
            dict: Transformed sample with the blurred image.

        """
        if random.random() < self.p:
            image = sample['image']
            sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
            if len(sample['image'].shape) == 3: # depth, height, width
                image_t = ndimage.gaussian_filter(image, sigma)
                sample['image'] = image_t
            elif len(sample['image'].shape) == 4:
                image_t = ndimage.gaussian_filter(image[0], sigma)
                sample['image'][0] = image_t

        return sample

class RandomSharpen(AbstractTransform):
    def __init__(self, p=0.5, sigma_range=(0.4, 0.8), alpha_range=(1.5, 2.0)):
        self.p = p
        self.sigma_range = sigma_range
        self.alpha_range = alpha_range
        
    def __call__(self, sample):
        if random.random() < self.p:
            image = sample['image']
            sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
            alpha = np.random.uniform(self.alpha_range[0], self.alpha_range[1])
            if len(sample['image'].shape) == 3:
                image = image * 255
                image = image.astype(np.uint8)
                blur_img = cv2.GaussianBlur(image, (3, 3), sigma)
                image = cv2.addWeighted(image, alpha, blur_img, 1 - alpha, 0)
                image = image.astype(np.float32) / 255
                sample['image'] = image
            elif len(sample['image'].shape) == 4:
                image = image[0] * 255
                image = image.astype(np.uint8)
                blur_img = cv2.GaussianBlur(image, (3, 3), sigma)
                image = cv2.addWeighted(image, alpha, blur_img, 1 - alpha, 0)
                image = image.astype(np.float32) / 255
                sample['image'][0] = image
                
        return sample

class RandomSharpenNodule(AbstractTransform):
    def __init__(self, p=0.5, sigma_range=(0.4, 0.8), alpha_range=(1.5, 2.0), offset = 1):
        self.p = p
        self.sigma_range = sigma_range
        self.alpha_range = alpha_range
        
    def __call__(self, sample):
        if random.random() < self.p and len(sample['ctr']) > 0:
            image = sample['image']
            ctr = sample['ctr'] # shape: [n, 3]
            rad = sample['rad'] # shape: [n, 3]
            sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
            alpha = np.random.uniform(self.alpha_range[0], self.alpha_range[1])
            
            # [z1, y1, x1, z2, y2, x2] = [ctr - rad / 2, ctr + rad / 2
            bboxes = np.array([ctr - rad / 2, ctr + rad / 2]).transpose(1, 0, 2).reshape(-1, 6)
            for box in bboxes:
                sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
                box[:3] = np.maximum(np.floor(box[:3] - self.offset), 0)
                box[3:] = np.minimum(np.ceil(box[3:] + self.offset), image.shape[1:])
                box = box.astype(np.int32)
                z1, y1, x1, z2, y2, x2 = box
                image_t = image[int(z1):int(z2), int(y1):int(y2), int(x1):int(x2)]
                image_t = image_t * 255
                image_t = image_t.astype(np.uint8)
                blur_img = cv2.GaussianBlur(image_t, (3, 3), sigma)
                image_t = cv2.addWeighted(image_t, alpha, blur_img, 1 - alpha, 0)
                image_t = image_t.astype(np.float32) / 255
                image[int(z1):int(z2), int(y1):int(y2), int(x1):int(x2)] = image_t
            sample['image'] = image
        return sample

class RandomNoiseNodule(AbstractTransform):
    def __init__(self, p=0.5, hu_offset = 15 / 1400, offset = 1):
        self.p = p
        self.hu_offset = hu_offset
        self.offset = offset
    def __call__(self, sample):
        if random.random() < self.p and len(sample['ctr']) > 0:
            image = sample['image']
            ctr = sample['ctr'] # shape: [n, 3]
            rad = sample['rad']
            bboxes = np.array([ctr - rad / 2, ctr + rad / 2]).transpose(1, 0, 2).reshape(-1, 6)
            for box in bboxes:
                box[:3] = np.maximum(np.floor(box[:3] - self.offset), 0)
                box[3:] = np.minimum(np.ceil(box[3:] + self.offset), image.shape[1:])
                box = box.astype(np.int32)
                z1, y1, x1, z2, y2, x2 = box
                image_t = image[int(z1):int(z2), int(y1):int(y2), int(x1):int(x2)]
                noise = np.random.uniform(-self.hu_offset, self.hu_offset, image_t.shape)
                image_t = image_t + noise
                image[int(z1):int(z2), int(y1):int(y2), int(x1):int(x2)] = image_t
        return sample
    
class RandomBlurNodule(AbstractTransform):
    """
    Randomly applies Gaussian blur to the input image.

    Args:
        sigma_range (tuple): Range of sigma values for Gaussian blur. Default is (0.4, 0.8).
        p (float): Probability of applying the transform. Default is 0.5.
        channel_apply (int): Index of the channel to apply the transform on. Default is 0.

    Returns:
        dict: Transformed sample with the blurred image.

    """
    def __init__(self, sigma_range=(0.4, 0.6), p=0.5, offset = 1):
        """
        Initializes the RandomBlur transform.

        Args:
            sigma_range (tuple): Range of sigma values for Gaussian blur. Default is (0.4, 0.8).
            p (float): Probability of applying the transform. Default is 0.5.
        """
        self.sigma_range = sigma_range
        self.p = p
        self.offset = offset

    def __call__(self, sample):
        """
        Applies the RandomBlur transform to the input sample.

        Args:
            sample (dict): Input sample containing the image.

        Returns:
            dict: Transformed sample with the blurred image.

        """
        if random.random() < self.p and len(sample['ctr']) > 0:
            image = sample['image']
            ctr = sample['ctr'] # shape: [n, 3]
            rad = sample['rad'] # shape: [n, 3]
            
            # [z1, y1, x1, z2, y2, x2] = [ctr - rad / 2, ctr + rad / 2
            bboxes = np.array([ctr - rad / 2, ctr + rad / 2]).transpose(1, 0, 2).reshape(-1, 6)
            for box in bboxes:
                sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
                box[:3] = np.maximum(np.floor(box[:3] - self.offset), 0)
                box[3:] = np.minimum(np.ceil(box[3:] + self.offset), image.shape[1:])
                box = box.astype(np.int32)
                z1, y1, x1, z2, y2, x2 = box
                image[0, z1:z2, y1:y2, x1:x2] = ndimage.gaussian_filter(image[0, z1:z2, y1:y2, x1:x2], sigma)    

        return sample

class RandomAugmentNodule(AbstractTransform):
    def __init__(self, p = 0.5, offset = 1, sigma_range=(0.4, 0.6), alpha_range=(1.5, 2.0), hu_offset = 15 / 1400):
        self.p = p
        self.offset = offset
        self.sigma_range = sigma_range
        self.alpha_range = alpha_range
        self.hu_offset = hu_offset
        
    def __call__(self, sample):
        if random.random() < self.p and len(sample['ctr']) > 0:
            image = sample['image']
            ctr = sample['ctr']
            rad = sample['rad']
            
            # [z1, y1, x1, z2, y2, x2] = [ctr - rad / 2, ctr + rad / 2
            bboxes = np.array([ctr - rad / 2, ctr + rad / 2]).transpose(1, 0, 2).reshape(-1, 6)
            for box in bboxes:
                box[:3] = np.maximum(np.floor(box[:3] - self.offset), 0)
                box[3:] = np.minimum(np.ceil(box[3:] + self.offset), image.shape[1:])
                box = box.astype(np.int32)
                z1, y1, x1, z2, y2, x2 = box
                d, h, w = z2 - z1, y2 - y1, x2 - x1
                if d * h * w == 0:
                    continue
                if d == 1 or h == 1 or w == 1:
                    aug_type = 'blur'
                else:
                    aug_type = random.choice(['blur', 'sharpen'])
                sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
                
                if aug_type == 'blur':
                    image[0, z1:z2, y1:y2, x1:x2] = ndimage.gaussian_filter(image[0, z1:z2, y1:y2, x1:x2], sigma)
                # elif aug_type == 'noise':
                #     box = box.astype(np.int32)
                #     z1, y1, x1, z2, y2, x2 = box
                #     noise = np.random.uniform(-self.hu_offset, self.hu_offset, (z2 - z1, y2 - y1, x2 - x1))
                #     image[0, z1:z2, y1:y2, x1:x2] = image[0, z1:z2, y1:y2, x1:x2] + noise
                elif aug_type == 'sharpen':
                    alpha = np.random.uniform(self.alpha_range[0], self.alpha_range[1])
                    image_t = image[0, z1:z2, y1:y2, x1:x2] * 255
                    image_t = image_t.astype(np.uint8)
                    blur_img = cv2.GaussianBlur(image_t, (0, 0), sigma)
                    image_t = cv2.addWeighted(image_t, alpha, blur_img, 1 - alpha, 0)
                    image_t = image_t.astype(np.float32) / 255
                    image[0, z1:z2, y1:y2, x1:x2] = image_t
                    
                sample['image'] = image
        return sample            

class RandomGamma(AbstractTransform):
    def __init__(self, gamma_range=[0.92, 1.08], p=0.5):
        """
        gamma range: gamme will be in [1/gamma_range, gamma_range]
        """
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            image = sample['image']
            gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
            if len(sample['image'].shape) == 3:
                image_t = np.power(image, gamma)
                sample['image'] = image_t
            elif len(sample['image'].shape) == 4:
                image_t = np.power(image[0], gamma)
                sample['image'][0] = image_t

        return sample

class RandomNoise(AbstractTransform):
    def __init__(self, p=0.5, gamma_range=(1e-4, 5e-4)):
        self.p = p
        self.gamma_range = gamma_range

    def __call__(self, sample):
        if random.random() < self.p:
            image = sample['image']
            gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
            if len(sample['image'].shape) == 3: # depth, height, width
                image_t = random_noise(image, var=gamma)
                sample['image'] = image_t
            elif len(sample['image'].shape) == 4: # channel, depth, height, width
                image_t = random_noise(image[0], var=gamma)
                sample['image'][0] = image_t

        return sample
