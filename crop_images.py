import os
import json
import numpy as np
from multiprocessing import Pool
root = 'D:\workspace\medical_dataset\LN_dataset'

def crop_image(series_folder: str, series_name:str, margins = [0, 0, 0]): # margins = [y, x, z]
    npy_path = os.path.join(series_folder, 'npy', f'{series_name}.npy')
    lobe_path = os.path.join(series_folder, 'npy', f'{series_name}_lobe.npz')
    lobe_info_path = os.path.join(series_folder, 'npy', 'lobe_info.txt')
    mask_path = os.path.join(series_folder, 'mask', f'{series_name}.npz')
    nodule_count_path = os.path.join(series_folder, 'mask', f'{series_name}_nodule_count.json')
    
    crop_npy_path = os.path.join(series_folder, 'npy', f'{series_name}_crop.npy')
    crop_lobe_path = os.path.join(series_folder, 'npy', f'{series_name}_lobe_crop.npz')
    crop_mask_path = os.path.join(series_folder, 'mask', f'{series_name}_crop.npz')
    crop_nodule_count_path = os.path.join(series_folder, 'mask', f'{series_name}_nodule_count_crop.json')
    
    # Get lobe info
    with open(lobe_info_path, 'r') as f:
        lobe_info = f.readlines()
        lobe_info = [i.strip() for i in lobe_info]
    z_min, z_max = [int(z) for z in lobe_info[-2].split(',')]
    y_min, y_max, x_min, x_max = [int(z) for z in lobe_info[-1].split(',')]
    
    y_min = max(0, y_min - margins[0])
    y_max = min(512, y_max + margins[0])
    x_min = max(0, x_min - margins[1])
    x_max = min(512, x_max + margins[1])
    
    offset = np.array([y_min, x_min, z_min], dtype=np.int32)
    offset = np.expand_dims(offset, axis=(0,1))
    
    nodule_count = json.load(open(nodule_count_path, 'r'))
    # (N,2, 3)
    if len(nodule_count['bboxes']) != 0:
        bboxes = np.array(nodule_count['bboxes'], dtype=np.int32) # [y_min, x_min, z_min], [y_max, x_max, z_max]
        bboxes = bboxes - offset
        nodule_count['bboxes'] = bboxes.tolist()
    
        nodule_start_slice_ids = np.array(nodule_count['nodule_start_slice_ids'], dtype=np.int32)
        nodule_start_slice_ids = nodule_start_slice_ids - z_min
        nodule_count['nodule_start_slice_ids'] = nodule_start_slice_ids.tolist()
    
    with open(crop_nodule_count_path, 'w') as f:
        json.dump(nodule_count, f)
    # crop image
    image = np.load(npy_path)
    image = image[y_min:y_max, x_min:x_max, z_min:z_max]
    np.save(crop_npy_path, image)
    
    mask = np.load(mask_path)['image']
    mask = mask[y_min:y_max, x_min:x_max, z_min:z_max]
    np.savez_compressed(crop_mask_path, image=mask)
    
    lobe = np.load(lobe_path)['image']
    lobe = lobe[y_min:y_max, x_min:x_max, z_min:z_max]
    np.savez_compressed(crop_lobe_path, image=lobe)

if __name__ == '__main__':
    tasks = []
    series_names = os.listdir(root)
    for name in series_names:
        series_folder = os.path.join(root, name)
        tasks.append((series_folder, name))
        
    pool = Pool(os.cpu_count() // 2)
    pool.starmap(crop_image, tasks)