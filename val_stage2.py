import os
import argparse
import logging
import datetime
import numpy as np

import re
from typing import List, Union

import torch
from offline_fl.dataset.stage2_dataset import Stage2Dataset
from offline_fl.client.stage2_logic import test

from offline_fl.inference.nodule_counter import NoduleCounter
from offline_fl.utilities import build_instance, load_yaml, setup_logging
from offline_fl.utilities.nodule_metrics import NoduleMetrics

logger = logging.getLogger(__name__)
cache_folder = './cache/'


def write_lines(result_csv_path: str, lines: Union[List[str], str]):
    if isinstance(lines, str):
        lines = [lines]
    
    lines = [line + '\n' for line in lines]
    with open(result_csv_path, 'a') as f:
        f.writelines(lines)
        
def get_newer_folder(root: str) -> str:
    """Get the newest folder in the root folder
    Return:
        str: the path of the newest folder
    """
    subfolders = [f.path for f in os.scandir(root) if f.is_dir()]
    creation_dates = {}
    for folder in subfolders:
        creation_timestamp = os.path.getctime(folder)
        creation_date =  datetime.datetime.fromtimestamp(creation_timestamp).strftime('%Y-%m-%d %H:%M:%S')
        creation_dates[folder] = creation_date

    sorted_creation_dates = sorted(creation_dates.items(), key = lambda kv: kv[1], reverse = True)
    return sorted_creation_dates[0][0]
    
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_set', default = 'fl_cmp_valABC.txt')
    parser.add_argument('--model_path', type=str, default='auto')
    parser.add_argument('--model', default='offline_fl.model.stage2.stage2_model.Stage2Model')
    parser.add_argument('--config_path', default='./config/stage2.yaml')
    parser.add_argument('--result_save_path', type=str, default='')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    setup_logging()
    # Read Config
    data_config = load_yaml('./config/data_config.yaml')
    co_config = load_yaml('./config/co_config.yaml')
    root = data_config['root']
    training_config = co_config['training']
    nodule_size_ranges = training_config['nodule_size_ranges']
    co_name = training_config['envoy']['shard_name']


    args = get_parser()
    fl_config = load_yaml(args.config_path)
    iou_threshold = args.iou_threshold
    nodule_3d_minimum_size = args.nodule_3d_minimum_size
    model_path = args.model_path
    result_save_path = args.result_save_path
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if model_path == 'auto':
        model_folder = get_newer_folder('./save')
        model_path = os.path.join(model_folder, 'best_model.pth')
    
    logger.info('Loading model from {}'.format(model_path))
    checkpoint = torch.load(model_path)
    if 'model_structure' in checkpoint:
        model = checkpoint['model_structure']
    else:
        model = build_instance(args.model, dict())
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)    

    # Dataset path
    val_set_path = os.path.join(root, args.val_set)
    
    # Number of Nodules
    nodule_counter = NoduleCounter()
    num_nodule_in_val_set = nodule_counter.count_and_analyze_nodules_of_multi_series(val_set_path, nodule_size_ranges)

    setup_logging()
    val_dataset = Stage2Dataset(dataset_type = 'test',
                                nodule_size_ranges = nodule_size_ranges,
                                num_nodules = num_nodule_in_val_set,
                                series_list_path = val_set_path,
                                crop_setting = fl_config['client']['dataset']['params']['crop_setting'],
                                cache_folder = cache_folder,
                                reset_data_in_disk = True)
    
    # Init Experiment folder
    if result_save_path == '':
        result_save_path = os.path.join(os.path.dirname(os.path.dirname(model_path)), 'result.csv')
    if not result_save_path.endswith('.csv'):
        result_save_path = result_save_path + '.csv'
    result_csv_path = result_save_path.replace('.csv', '_thrs{:.2f}.csv'.format(iou_threshold))
    
    logger.info("Saving result to '{}'".format(result_csv_path))
    # Validating
    stage1_results = test(model, val_dataset, iou_threshold, device = device, log_metric = True)
    write_lines(result_csv_path, 'iou_threshold,val_txt_path')
    write_lines(result_csv_path, '{:.2f},{}'.format(iou_threshold, val_set_path))
    nodule_metrics = NoduleMetrics(stage1_results)
    write_lines(result_csv_path, nodule_metrics.generate_metric_csv_lines())