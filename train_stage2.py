import os
import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from offline_fl.client.stage2_logic import train, validation, test
from offline_fl.dataset.stage2_dataset import Stage2Dataset
from offline_fl.model.ema import EMA

from offline_fl.inference.nodule_counter import NoduleCounter
from offline_fl.utilities import build_instance, load_yaml, get_local_time_in_taiwan, setup_logging, write_yaml, init_seed
from offline_fl.utilities.nodule_metrics import NoduleMetrics
from val_stage2 import write_lines

logger = logging.getLogger(__name__)
cache_folder = './cache/'
### Read Config
data_config = load_yaml('./config/data_config.yaml')
co_config = load_yaml('./config/co_config.yaml')
root = data_config['root']
training_config = co_config['training']
nodule_size_ranges = training_config['nodule_size_ranges']
co_name = training_config['envoy']['shard_name']

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extra_info', default = '')
    parser.add_argument('--train_set', default = 'fl_cmp_trainABC.txt')
    parser.add_argument('--val_set', default = 'fl_cmp_valABC.txt')
    parser.add_argument('--test_set', default = 'val.txt')
    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--num_epoch', type = int, default = 80)
    parser.add_argument('--lr', type = float, default = 0.0001)
    parser.add_argument('--seed', type = int, default = 1029)
    parser.add_argument('--model', default='offline_fl.model.stage2.stage2_model.Stage2Model')
    parser.add_argument('--base_planes', type = int, default = 16)
    parser.add_argument('--apply_ema', action='store_true', default=False)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--resume_model_path', type=str, default='')
    parser.add_argument('--best_model_metric_name', type=str, default='f1_score')
    parser.add_argument('--config_path', default='./config/stage2.yaml')
    args = parser.parse_args()
    return args

def write_metrics(metrics: dict, 
                epoch: int,
                prefix: str,
                writer: SummaryWriter):
    for metric, value in metrics.items():
        writer.add_scalar(f'{prefix}/{metric}', value, global_step = epoch)
    writer.flush()

def save_states(model: nn.Module, 
                            optimizer: optim.Optimizer,
                            save_path: str = './model.pth',
                            ema: EMA = None):
    save_dict = {'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'model_structure': model}
    if ema is not None:
        save_dict['ema'] = ema.state_dict()
    torch.save(save_dict, save_path)

        
if __name__ == '__main__':
    args = get_parser()
    seed = args.seed
    extra_info = args.extra_info
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    learning_rate = args.lr
    base_planes = args.base_planes
    apply_ema = args.apply_ema
    resume_model_path = args.resume_model_path
    best_model_metric_name = args.best_model_metric_name
    fl_config = load_yaml(args.config_path)
    # Set seed
    init_seed(seed)
    # Experiment Name
    cur_time = get_local_time_in_taiwan()
    timestamp = "[%d-%02d-%02d-%02d%02d]" % (cur_time.year, 
                                            cur_time.month, 
                                            cur_time.day, 
                                            cur_time.hour, 
                                            cur_time.minute)
    exp_name = timestamp
    if extra_info != '':
        exp_name = exp_name + f'_{extra_info}'
    exp_root = f'./save/{exp_name}'
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Build model and optimizer
    if resume_model_path != '': # Resume model
        checkpoint = torch.load(resume_model_path, map_location = device)
        if 'model_structure' in checkpoint:
            model = checkpoint['model_structure']
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'ema' in checkpoint:
            ema = EMA(model, decay = args.ema_decay)
            ema.load_state_dict(checkpoint['ema'])
            # Restore model weights from shadow weights to original weights
            ema.restore()
        else:
            ema = None
        # Get start epoch and end epoch from model name, e.g. 1.pth
        model_name = os.path.basename(resume_model_path)
        start_epoch = int(model_name.split('.')[0])
        end_epoch = start_epoch + num_epoch
        
        exp_name = os.path.basename(os.path.dirname(resume_model_path))
        exp_root = os.path.dirname(os.path.dirname(resume_model_path))
    else: 
        # Build new model
        model = build_instance(args.model, {'base_planes': base_planes})
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        start_epoch = 0
        end_epoch = num_epoch
        # Register EMA
        if apply_ema:
            ema = EMA(model, decay = args.ema_decay)  
            ema.register()
        else:
            ema = None
    
    # Dataset path
    train_set_path = os.path.join(root, args.train_set)
    val_set_path = os.path.join(root, args.val_set)
    test_set_path = os.path.join(root, args.test_set)
    
    # Number of Nodules
    nodule_counter = NoduleCounter()
    num_nodule_in_train_set = nodule_counter.count_and_analyze_nodules_of_multi_series(train_set_path, nodule_size_ranges)
    num_nodule_in_val_set = nodule_counter.count_and_analyze_nodules_of_multi_series(val_set_path, nodule_size_ranges)
    num_nodule_in_test_set = nodule_counter.count_and_analyze_nodules_of_multi_series(test_set_path, nodule_size_ranges)

    saving_model_root = os.path.join(exp_root, 'model')
    log_txt_path = os.path.join(exp_root, 'log.txt')
    setup_logging(log_file=log_txt_path)
    # Init Experiment folder
    logger.info("Save model into '{}'".format(saving_model_root))
    os.makedirs(saving_model_root, exist_ok=True)
    if resume_model_path == '':
        write_yaml(os.path.join(exp_root, 'setting.yaml'), vars(args))
    
    # Get dataset
    train_dataset = Stage2Dataset(dataset_type = 'train',
                                    nodule_size_ranges = nodule_size_ranges,
                                    num_nodules = num_nodule_in_train_set,
                                    crop_setting = fl_config['client']['dataset']['params']['crop_setting'],
                                    cache_folder = cache_folder,
                                    series_list_path = train_set_path)
    
    val_dataset = Stage2Dataset(dataset_type = 'valid',
                                nodule_size_ranges = nodule_size_ranges,
                                num_nodules = num_nodule_in_val_set,
                                crop_setting = fl_config['client']['dataset']['params']['crop_setting'],
                                cache_folder = cache_folder,
                                series_list_path = val_set_path)
    
    test_dataset = Stage2Dataset(dataset_type = 'test',
                                nodule_size_ranges = nodule_size_ranges,
                                num_nodules = num_nodule_in_test_set,
                                crop_setting = fl_config['client']['dataset']['params']['crop_setting'],
                                cache_folder = cache_folder,
                                series_list_path = test_set_path)

    # Training
    writer = SummaryWriter(log_dir = os.path.join(exp_root, 'tensorboard'))
    best_epoch = 0
    best_metric = 0.0
    last_best_txt = ''
    
    model = model.to(device)
    for epoch in range(start_epoch, end_epoch):
        logger.info("Epoch {}/{}:".format(epoch + 1, end_epoch))
        # Train
        train_metrics = train(model = model, 
                                     dataset = train_dataset, 
                                     optimizer = optimizer, 
                                     num_epoch = 1, 
                                     batch_size = batch_size, 
                                     device = device, 
                                     enable_progress_bar=True,
                                     log_metric=True,
                                     ema = ema)
        write_metrics(train_metrics, epoch, 'train', writer)
        
        # Use Shadow model to validate and save model
        if ema is not None:
            ema.apply_shadow()
            
        save_states(model = model, 
                    optimizer = optimizer, 
                    save_path = os.path.join(saving_model_root, f'{epoch + 1}.pth'),
                    ema = ema)
        # Validating
        val_metrics = validation(model, val_dataset, device=device, enable_progress_bar=True, log_metric=True, batch_size=batch_size)
        write_metrics(val_metrics, epoch, 'val', writer)
        metric = val_metrics[best_model_metric_name]
        if metric >= best_metric:
            best_metric = metric
            best_epoch = epoch + 1
            logger.info(f'Save best model in epoch {epoch + 1} for {best_model_metric_name} = {best_metric:.3f}')
            with open(os.path.join(exp_root, 'best.txt'), 'w') as f:
                f.write(f'{epoch + 1}\n')
                f.write(f'{best_model_metric_name} = {best_metric:.3f}\n')
            save_states(model = model, 
                        optimizer = optimizer, 
                        save_path = os.path.join(exp_root, 'best.pth'),
                        ema = ema)
        
        # Restore model
        if ema is not None:
            ema.restore()
    logger.info(f'Best model is at epoch {best_epoch} for {best_model_metric_name} = {best_metric:.3f}')
    # Testing
    logger.info('Start to test with best model')
    
    # Load best model
    model_state_dict = torch.load(os.path.join(exp_root, 'best.pth'), map_location=device)
    model.load_state_dict(model_state_dict['model_state_dict'])
    
    result_csv_path = os.path.join(exp_root, f'{exp_name}.csv')
    
    # Validating with validation set
    stage2_results = test(model, val_dataset, device = device, batch_size=batch_size)
    write_lines(result_csv_path, 'val_txt_path')
    write_lines(result_csv_path, '{}'.format(val_set_path))
    nodule_metrics = NoduleMetrics(stage2_results)
    write_lines(result_csv_path, nodule_metrics.generate_metric_csv_lines())
    
    # Validating with test set
    stage2_results = test(model, test_dataset, device = device, batch_size=batch_size)
    write_lines(result_csv_path, 'val_txt_path')
    write_lines(result_csv_path, '{}'.format(test_set_path))
    nodule_metrics = NoduleMetrics(stage2_results)
    write_lines(result_csv_path, nodule_metrics.generate_metric_csv_lines())
    
    # Write metrics to tensorboard
    write_metrics(stage2_results['all'], epoch, 'test', writer)
    writer.close()