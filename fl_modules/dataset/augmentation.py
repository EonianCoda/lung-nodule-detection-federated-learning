import random
import numpy as np
from typing import List, Tuple, Union
import albumentations as A
import cv2
import torch
class RandomFlipYXZ:
    def __init__(self, p: float = 0.5):
        self.p = p
    def __call__(self, images: Union[Tuple[np.ndarray], Tuple[np.ndarray]]) -> Union[Tuple[np.ndarray], Tuple[np.ndarray]]:
        flip_axes = []
        for i in range(3):
            if random.random() < self.p:
                flip_axes.append(i)
        if len(flip_axes) != 0:
            for i in range(len(images)):
                if isinstance(images[i], torch.Tensor):
                    images[i] = torch.flip(images[i], dims = flip_axes)
                else:
                    images[i] = np.flip(images[i], axis = flip_axes)
        return images
    
def random_color(image, p: float = 0.3, intensity: float = 1.0):
    image = A.RandomBrightnessContrast(brightness_limit = 0.1 * intensity, 
                                       contrast_limit = 0.1 * intensity,
                                       p = p)(image = image)
    return image['image']

def random_gauss_noise(image, p: float = 0.3):
    image = A.GaussNoise(p = p, var_limit=0.01)(image = image)
    return image['image']

def random_blur(image, p: float = 0.3):
    blur_image = A.Blur(p = p, blur_limit = 5)(image = image)['image']
    image = blur_image * 0.5 + image * 0.5 
    return image

class RandomRotation:
    def __init__(self, angle_range: list = [-10, 10], p=0.5):
        self.angle_range = angle_range
        self.p = p
    def __call__(self, images, is_masks: list):
        if random.random() < self.p:
            angle = random.choice(list(range(self.angle_range[0], self.angle_range[1] + 1)))
            for i, is_mask in enumerate(is_masks):
                if not is_mask:
                    images[i] = A.rotate(images[i], angle, border_mode=cv2.BORDER_CONSTANT, value=0)
                else:
                    images[i] = A.rotate(images[i], angle, border_mode=cv2.BORDER_CONSTANT, value=0, interpolation=cv2.INTER_NEAREST)
        return images

class RandomCutout:
    def __init__(self, 
                 p: float = 0.5,
                 num_of_crop: int = 3,
                 cut_block_size: int = 5, 
                 img_size = (40, 40, 30)) -> None:
        self.p = p
        self.cut_block_size = cut_block_size
        self.img_size = img_size
        self.num_of_crop = num_of_crop
    
    def get_random_box(self):
        cut_center = [random.randrange(max_value) for max_value in self.img_size]  
        lt = [max(center - (self.cut_block_size // 2), 0)  for center in cut_center]
        rb = [min(center + (self.cut_block_size // 2) + 1, max_value)  for center, max_value in zip(cut_center, self.img_size)]
        
        box = np.array([lt, rb], dtype = np.int32)
        return box 

    def __call__(self, images):
        for i, img in enumerate(images):
            img = img.copy()
            for _ in range(self.num_of_crop):
                if random.random() < self.p:
                    box = self.get_random_box()
                    img[box[0,0]: box[1,0], box[0,1]: box[1,1], box[0,2]: box[1,2]] = 0.0
            images[i] = img
        return images

class RandomRotation90:
    def __init__(self, p = 0.5) -> None:
        self.p = p
        self.angles = [90, 180, 270]
    def __call__(self, images):
        if random.random() < self.p:
            rot_angle = random.choice(self.angles)
            for i, img in enumerate(images):
                img = A.rotate(img, angle=rot_angle)
                images[i] = img
        return images    