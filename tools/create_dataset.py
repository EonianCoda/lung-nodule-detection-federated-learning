import os
import shutil

npy_root = 'D:\\workspace\\medical_dataset\\LDCT_test_lobe_0.8x0.8x1_npz'
mask_root = 'D:\\workspace\\medical_dataset\\LN_mask'
dst_root = 'D:\\workspace\\medical_dataset\\LN_dataset'

if __name__ == '__main__':
    series_names = os.listdir(npy_root)
    for name in series_names:
        npy_folder = os.path.join(npy_root, name)
        mask_folder = os.path.join(mask_root, name)
        dst_folder = os.path.join(dst_root, name)
        
        dst_npy_folder = os.path.join(dst_folder, 'npy')
        dst_mask_folder = os.path.join(dst_folder, 'mask')
        
        if os.listdir(npy_folder) == 0:
            os.rmdir(npy_folder)
            os.rmdir(mask_folder)
            continue
        
        if not os.path.exists(mask_folder):
            print('mask folder not exist: ', mask_folder)
            continue
        
        os.makedirs(dst_npy_folder, exist_ok=True)
        os.makedirs(dst_mask_folder, exist_ok=True)
        
        for file in os.listdir(npy_folder):
            shutil.move(os.path.join(npy_folder, file), os.path.join(dst_npy_folder, file))
        
        for file in os.listdir(mask_folder):
            shutil.move(os.path.join(mask_folder, file), os.path.join(dst_mask_folder, file))
            
        os.rmdir(npy_folder)
        os.rmdir(mask_folder)