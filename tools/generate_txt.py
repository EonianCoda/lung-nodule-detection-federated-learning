import os
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_dir', type=str, help='path to dataset')
    parser.add_argument('--file_root', type=str, help='path to dataset')
    parser.add_argument('--save_dir', type=str, default='./datasplit', help='path to save')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    
    split_dir = args.split_dir
    save_dir = args.save_dir
    file_root = args.file_root
    
    os.makedirs(save_dir, exist_ok=True)
    for file_name in os.listdir(split_dir):
        txt_path = os.path.join(split_dir, file_name)
        save_path = os.path.join(save_dir, file_name)
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]
        save_lines = ['Folder,File_Name\n']
        for series_name in lines:
            line = '{},{}\n'.format(os.path.join(file_root, series_name), series_name)
            save_lines.append(line)
        
        # Remove the last '\n'
        save_lines[-1] = save_lines[-1][:-1]
        with open(save_path, 'w') as f:
            f.writelines(save_lines)