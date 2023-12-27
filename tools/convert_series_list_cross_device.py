import os
import argparse

ME_ROOT = 'D:\\workspace\\medical_dataset\\LN_dataset'
LDCT_TEST_ROOT = 'D:\\workspace\\medical_dataset\\LDCT_test_dataset'
LN_ROOT = 'D:\\workspace\\medical_dataset\\LN_dataset'
SERIES_LIST_TEXT_ROOT = 'D:\\workspace\\python\\lung-nodule-detection-federated-learning\\data\\lung_nodule'
SAVE_ROOT = 'D:\\workspace\\python\\lung-nodule-detection-federated-learning\\data\\lung_nodule_cross_device'

key_words = {'LDCT_test_dataset': LDCT_TEST_ROOT,
             'LN_dataset': LN_ROOT,
             'ME_dataset': ME_ROOT}

if __name__ == '__main__':
    
    for series_list_txt in os.listdir(SERIES_LIST_TEXT_ROOT):
        path = os.path.join(SERIES_LIST_TEXT_ROOT, series_list_txt)
        with open(path, 'r') as f:
            header = f.readline()
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            folder, filename = line.split(',')
            for key_word in key_words.keys():
                if key_word in folder:
                    root = key_words[key_word]
                    new_lines.append('{},{}'.format(os.path.join(root, os.path.basename(folder)), filename))
                    break
                
        save_path = os.path.join(SAVE_ROOT, series_list_txt)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(header)
            f.writelines(new_lines)