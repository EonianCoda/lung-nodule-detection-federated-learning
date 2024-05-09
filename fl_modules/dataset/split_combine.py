from typing import List, Tuple
import numpy as np
import scipy

class SplitComb():
    def __init__(self, crop_size: List[int]=[64, 128, 128], overlap_size:List[int]=[16, 32, 32], do_padding:bool=True, pad_value:int=0):
        self.stride_size = [crop_size[0]-overlap_size[0], 
                            crop_size[1]-overlap_size[1], 
                            crop_size[2]-overlap_size[2]]
        self.overlap = overlap_size
        self.crop_size = np.array(crop_size, dtype=np.int32)
        self.pad_value = pad_value
        self.do_padding = do_padding

    def split(self, image, lobe = None, out_stride: int = 4):
        do_padding = self.do_padding
        apply_lobe = isinstance(lobe, np.ndarray)
        image_splits = []
        d, h, w = image.shape

        # Number of splits in each dimension
        nz = int(np.ceil(float(d) / self.stride_size[0]))
        ny = int(np.ceil(float(h) / self.stride_size[1]))
        nx = int(np.ceil(float(w) / self.stride_size[2]))

        nzyx = [nz, ny, nx]
        if do_padding or (d < self.crop_size[0]) or (h < self.crop_size[1]) or (w < self.crop_size[2]): # Last 3 condition for fail lobe
            pad = [[0, int(nz * self.stride_size[0] + self.overlap[0] - d)],
                    [0, int(ny * self.stride_size[1] + self.overlap[1] - h)],
                    [0, int(nx * self.stride_size[2] + self.overlap[2] - w)]]
            image = np.pad(image, pad, 'constant', constant_values=self.pad_value)
            if apply_lobe:
                lobe = np.pad(lobe, pad, 'constant', constant_values=0)
            do_padding = True
            
        if apply_lobe:
            # Make sure the shape of lobe is divisible by out_stride
            lobe_shape = np.array(lobe.shape)
            lobe_pad = (np.ceil(lobe_shape / out_stride) * out_stride - lobe_shape).astype(np.int32)
            if np.any(lobe_pad != 0):
                lobe = np.pad(lobe, [(0, lobe_pad[0]), (0, lobe_pad[1]), (0, lobe_pad[2])], mode='constant', constant_values=0)
                lobe_shape = lobe.shape
                
            zoom_ratio = [1 / out_stride, 1 / out_stride, 1 / out_stride]
            resized_lobe = scipy.ndimage.zoom(lobe, zoom_ratio, order=0, mode='nearest')
            strided_crop_size = [int(x / out_stride) for x in self.crop_size] 
            lobe_splits = []
            
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    start_z = int(iz * self.stride_size[0])
                    end_z = start_z + self.crop_size[0]
                    if end_z > d and not do_padding:
                        start_z = d - self.crop_size[0]
                        end_z = d
                        
                    start_y = int(iy * self.stride_size[1])
                    end_y = start_y + self.crop_size[1]
                    if end_y > h and not do_padding:
                        start_y = h - self.crop_size[1]
                        end_y = h
                    
                    start_x = int(ix * self.stride_size[2])
                    end_x = start_x + self.crop_size[2]
                    if end_x > w and not do_padding:
                        start_x = w - self.crop_size[2]
                        end_x = w

                    image_split = image[np.newaxis, np.newaxis, start_z:end_z, start_y:end_y, start_x:end_x]
                    if apply_lobe:
                        lobe_start_z = start_z // out_stride
                        lobe_start_y = start_y // out_stride
                        lobe_start_x = start_x // out_stride
                        
                        lobe_end_z = lobe_start_z + strided_crop_size[0]
                        lobe_end_y = lobe_start_y + strided_crop_size[1]
                        lobe_end_x = lobe_start_x + strided_crop_size[2]
                        
                        lobe_split = resized_lobe[np.newaxis, np.newaxis, lobe_start_z:lobe_end_z, lobe_start_y:lobe_end_y, lobe_start_x:lobe_end_x]
                        lobe_splits.append(lobe_split)
                    image_splits.append(image_split)

        image_splits = np.concatenate(image_splits, 0)
        
        if apply_lobe:
            lobe_splits = np.concatenate(lobe_splits, 0)
            return image_splits, lobe_splits, nzyx, [d, h, w]
        else:
            return image_splits, nzyx, [d, h, w]

    def combine(self, output, nzhw, image_shape: Tuple[int, int, int]):
        nz, nh, nw = nzhw
        idx = 0
        for iz in range(nz): 
            for ih in range(nh):
                for iw in range(nw):
                    sz = int(iz * self.stride_size[0])
                    sh = int(ih * self.stride_size[1])
                    sw = int(iw * self.stride_size[2])
                    
                    if sz + self.crop_size[0] > image_shape[0] and not self.do_padding:
                        sz = image_shape[0] - self.crop_size[0]
                        
                    if sh + self.crop_size[1] > image_shape[1] and not self.do_padding:
                        sh = image_shape[1] - self.crop_size[1]
                        
                    if sw + self.crop_size[2] > image_shape[2] and not self.do_padding:
                        sw = image_shape[2] - self.crop_size[2]
                    
                    # [N, 8]
                    # 8-> id, prob, z_min, y_min, x_min, d, h, w 
                    output[idx][:, 2] += sz
                    output[idx][:, 3] += sh
                    output[idx][:, 4] += sw
                    idx += 1
        return output