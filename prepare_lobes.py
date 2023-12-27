import os
import argparse
import logging

from fl_modules.inference.lobe3d_segmentation import prepare_lobe_of_series
from fl_modules.utilities import load_yaml

logger = logging.getLogger(__name__)

### Read Config
data_config = load_yaml('./config/data_config.yaml')
co_config = load_yaml('./config/co_config.yaml')
root = data_config['root']
training_config = co_config['training']
nodule_size_ranges = training_config['nodule_size_ranges']
co_name = training_config['envoy']['shard_name']

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_set', default = 'fl_cmp_trainABC.txt')
    parser.add_argument('--val_set', default = 'fl_cmp_valABC.txt')
    parser.add_argument('--test_set', default = 'val.txt')
    args = parser.parse_args()
    return args

def load_series_list(series_list_path: str):
    """
    Return:
        series_list: list of tuples (series_folder, file_name)

    """
    with open(series_list_path, 'r') as f:
        lines = f.readlines()
        lines = lines[1:] # Remove the line of description
        
    series_list = []
    for series_info in lines:
        series_info = series_info.strip()
        series_folder, file_name = series_info.split(',')
        series_list.append([series_folder, file_name])
    return series_list

if __name__ == '__main__':
    args = get_parser()
    
    # Dataset path
    train_set_path = os.path.join(root, args.train_set)
    val_set_path = os.path.join(root, args.val_set)
    test_set_path = os.path.join(root, args.test_set)
    
    series_paths = [os.path.join(series_info[0], 'npy', series_info[1] + '.npy') for series_info in load_series_list(train_set_path)]
    lobe_paths, first_and_last_slice_of_lobe_paths = prepare_lobe_of_series(series_paths)
    
    series_paths = [os.path.join(series_info[0], 'npy', series_info[1] + '.npy') for series_info in load_series_list(val_set_path)]
    lobe_paths, first_and_last_slice_of_lobe_paths = prepare_lobe_of_series(series_paths, overwrite=True)
    
    series_paths = [os.path.join(series_info[0], 'npy', series_info[1] + '.npy') for series_info in load_series_list(test_set_path)]
    lobe_paths, first_and_last_slice_of_lobe_paths = prepare_lobe_of_series(series_paths)