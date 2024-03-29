import os
import argparse
import logging
import datetime
import numpy as np

import re
from typing import List, Union

import torch
from fl_modules.dataset import Stage1Dataset
from fl_modules.client.stage1_logic import test

from fl_modules.inference.nodule_counter import NoduleCounter
from fl_modules.utilities import build_instance, load_yaml, setup_logging
from fl_modules.utilities.nodule_metrics import NoduleMetrics
from fl_modules.inference.predictor import PredictorStage1 
logger = logging.getLogger(__name__)

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
    parser.add_argument('--iou_threshold', type = float, default = 0.01)
    parser.add_argument('--nodule_3d_minimum_size', type = int, default = 5)
    parser.add_argument('--model_path', type=str, default='auto')
    parser.add_argument('--model', default='fl_modules.model.stage1.stage1_model.Stage1Model')
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
    iou_threshold = args.iou_threshold
    nodule_3d_minimum_size = args.nodule_3d_minimum_size
    model_path = args.model_path
    result_save_path = args.result_save_path
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if model_path == 'auto':
        model_folder = get_newer_folder('./save')
        model_path = os.path.join(model_folder, 'best.pth')
    
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
    # Init Experiment folder
    val_dataset = Stage1Dataset(dataset_type = 'valid',
                                depth = 32,
                                nodule_size_ranges = nodule_size_ranges,
                                num_nodules = num_nodule_in_val_set,
                                series_list_path = val_set_path)
    
    if result_save_path == '':
        result_save_path = os.path.join(os.path.dirname(os.path.dirname(model_path)), 'result.csv')
    if not result_save_path.endswith('.csv'):
        result_save_path = result_save_path + '.csv'
    result_csv_path = result_save_path.replace('.csv', '_thrs{:.2f}.csv'.format(iou_threshold))
    
    logger.info('Validating with iou_threshold = {:.2f}'.format(iou_threshold))
    logger.info("Saving result to '{}'".format(result_csv_path))
    # Validating
    
    
    model.eval()
    series_paths = val_dataset.series_paths
    gt_mask_maps_paths = val_dataset.gt_mask_maps_paths
    predictor = PredictorStage1(model, 
                                use_lobe = True, 
                                iou_threshold = iou_threshold,
                                nodule_3d_minimum_size = nodule_3d_minimum_size,
                                device = device, 
                                log_metrics = True)
    test_metrics = predictor.get_recall_precision_of_nodules_in_series(series_paths, gt_mask_maps_paths, return_each_series=True)
    for i in range(len(test_metrics)):
        tp = test_metrics[i]['all']['tp']
        fp = test_metrics[i]['all']['fp']
        fn = test_metrics[i]['all']['fn']
        info = ','.join(str(int(n)) for n in [tp, fp, fn])
        print(series_paths[i], info)