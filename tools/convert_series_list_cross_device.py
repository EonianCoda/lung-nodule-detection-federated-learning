import os
import shutil

ME_ROOT = '/disk/HDD1/FL/ME_dataset'
LDCT_TEST_ROOT = '/disk/HDD1/FL/LDCT_test_dataset'
LN_ROOT = '/disk/HDD1/FL/LN_dataset'
SERIES_LIST_TEXT_ROOT = '../data/lung_nodule'
SAVE_ROOT = '../data/lung_nodule_cross_device'

key_words = {'LDCT_test_dataset': LDCT_TEST_ROOT,
             'LN_dataset': LN_ROOT,
             'ME_dataset': ME_ROOT}

white_list = ['splitted_info.txt']

if __name__ == '__main__':
    
    for series_list_txt in os.listdir(SERIES_LIST_TEXT_ROOT):
        if series_list_txt in white_list:
            save_path = os.path.join(SAVE_ROOT, series_list_txt)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            shutil.copy(os.path.join(SERIES_LIST_TEXT_ROOT, series_list_txt), save_path)
            continue
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
                    new_lines.append('{},{}'.format(os.path.join(root, filename.strip()), filename))
                    break
                
        save_path = os.path.join(SAVE_ROOT, series_list_txt)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(header)
            f.writelines(new_lines)