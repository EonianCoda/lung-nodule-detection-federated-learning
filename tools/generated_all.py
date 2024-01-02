import os
import random

paths = []
roots = ['D:\\workspace\\medical_dataset\\LN_dataset',
         'D:\\workspace\\medical_dataset\\LDCT_test_dataset',
        'D:\\workspace\\medical_dataset\\ME_dataset']


if __name__ == '__main__':
    for root in roots:
        for name in os.listdir(root):
            folder = os.path.join(root, name)
            paths.append('{},{}'.format(folder, name))
            
    random.seed(1029)
    random.shuffle(paths)
    header = 'Folder,Filename\n'

    with open('./all.txt', 'w') as f:
        f.write(header)
        f.writelines('\n'.join(paths))