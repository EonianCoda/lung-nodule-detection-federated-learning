import os
import shutil
import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from fl_modules.client.stage1_logic import train, validation, test
from fl_modules.dataset import Stage1Dataset
from fl_modules.model.ema import EMA

from fl_modules.inference.nodule_counter import NoduleCounter
from fl_modules.utilities import build_instance, load_yaml, get_local_time_in_taiwan, setup_logging, write_yaml, init_seed
from fl_modules.utilities.nodule_metrics import NoduleMetrics
from val_stage1 import write_lines


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
    parser.add_argument('--extra_info', default = '')
    parser.add_argument('--train_set', default = 'pretrained_train.txt')
    parser.add_argument('--val_set', default = 'pretrained_val.txt')
    parser.add_argument('--test_set', nargs='+', type=str, default=['client0_test.txt', 'client1_test.txt', 'client2_test.txt'])
    parser.add_argument('--batch_size', type = int, default = 1)
    parser.add_argument('--num_epoch', type = int, default = 50)
    parser.add_argument('--iou_threshold', type = float, default = 0.2)
    parser.add_argument('--test_iou_threshold', type = float, default = 0.01)
    parser.add_argument('--test_nodule_3d_minimum_size', type = int, default = 5)
    parser.add_argument('--lr', type = float, default = 0.0001)
    parser.add_argument('--seed', type = int, default = 1029)
    parser.add_argument('--model', default='fl_modules.model.stage1.stage1_model.Stage1Model')
    parser.add_argument('--normalization', default='instance')
    parser.add_argument('--apply_ema', action='store_true', default=False)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--resume_model_path', type=str, default='')
    parser.add_argument('--mixed_precision', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', action='store_true', default=False)
    parser.add_argument('--best_model_metric_name', type=str, default='f1_score')
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
    iou_threshold = args.iou_threshold
    test_iou_threshold = args.test_iou_threshold
    test_nodule_3d_minimum_size = args.test_nodule_3d_minimum_size
    num_epoch = args.num_epoch
    learning_rate = args.lr * batch_size
    apply_ema = args.apply_ema
    resume_model_path = args.resume_model_path
    best_model_metric_name = args.best_model_metric_name
    mixed_precision = args.mixed_precision
    num_workers = args.num_workers
    pin_memory = args.pin_memory
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
        model = build_instance(args.model, {'normalization': args.normalization})
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
    
    # Init best model
    best_txt_path = os.path.join(exp_root, 'best.txt')
    if os.path.exists(best_txt_path): # resume training
        with open(best_txt_path, 'r') as f:
            best_epoch = int(f.readline())
            best_model_line = f.readline()
            best_model_metric_name = best_model_line.split('=')[0].strip()
            best_metric = float(best_model_line.split('=')[1])
    else:
        best_epoch = 0
        best_metric = 0.0
    
    # Dataset path
    train_set_path = os.path.join(root, args.train_set)
    val_set_path = os.path.join(root, args.val_set)
    
    test_set_paths = []
    for i in range(len(args.test_set)):
        test_set_paths.append(os.path.join(root, args.test_set[i]))
    
    # Copy dataset txt to experiment folder
    dataset_txt_folder = os.path.join(exp_root, 'dataset_txt')
    os.makedirs(dataset_txt_folder, exist_ok=True)
    dataset_txt_paths = []
    dataset_txt_paths.append(train_set_path)
    dataset_txt_paths.append(val_set_path)
    dataset_txt_paths.extend(test_set_paths)
    for dataset_txt_path in dataset_txt_paths:
        dataset_txt_name = os.path.basename(dataset_txt_path)
        shutil.copy(dataset_txt_path, os.path.join(dataset_txt_folder, dataset_txt_name))
    
    # Number of Nodules
    nodule_counter = NoduleCounter()
    num_nodule_in_train_set = nodule_counter.count_and_analyze_nodules_of_multi_series(train_set_path, nodule_size_ranges)
    num_nodule_in_val_set = nodule_counter.count_and_analyze_nodules_of_multi_series(val_set_path, nodule_size_ranges)
    num_nodule_in_test_sets = []
    for test_set_path in test_set_paths:
        num_nodule_in_test_sets.append(nodule_counter.count_and_analyze_nodules_of_multi_series(test_set_path, nodule_size_ranges))

    saving_model_root = os.path.join(exp_root, 'model')
    log_txt_path = os.path.join(exp_root, 'log.txt')
    setup_logging(log_file=log_txt_path)
    # Init Experiment folder
    logger.info("Save model into '{}'".format(saving_model_root))
    os.makedirs(saving_model_root, exist_ok=True)
    if resume_model_path == '':
        write_yaml(os.path.join(exp_root, 'setting.yaml'), vars(args))
    
    # Get dataset
    train_dataset = Stage1Dataset(dataset_type = 'train',
                                    depth = 32,
                                    nodule_size_ranges = nodule_size_ranges,
                                    num_nodules = num_nodule_in_train_set,
                                    series_list_path = train_set_path)
    
    val_dataset = Stage1Dataset(dataset_type = 'valid',
                                depth = 32,
                                nodule_size_ranges = nodule_size_ranges,
                                num_nodules = num_nodule_in_val_set,
                                series_list_path = val_set_path)
    
    test_datasets = []
    for i in range(len(test_set_paths)):
        test_set_path = test_set_paths[i]
        num_nodule_in_test_set = num_nodule_in_test_sets[i]
        test_datasets.append(Stage1Dataset(dataset_type = 'test',
                                            depth = 32,
                                            nodule_size_ranges = nodule_size_ranges,
                                            num_nodules = num_nodule_in_test_set,
                                            series_list_path = test_set_path))

    # Training
    writer = SummaryWriter(log_dir = os.path.join(exp_root, 'tensorboard'))
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size = batch_size, 
                                  num_workers=num_workers,
                                  prefetch_factor = 1,
                                  pin_memory=pin_memory)
    val_dataloader = DataLoader(val_dataset, batch_size = 1)
    
    model = model.to(device)
    for epoch in range(start_epoch, end_epoch):
        logger.info("Epoch {}/{}:".format(epoch + 1, end_epoch))
        # Train
        train_metrics = train(model = model, 
                                dataloader = train_dataloader, 
                                optimizer = optimizer, 
                                mixed_precision = mixed_precision,
                                device = device, 
                                enable_progress_bar=True,
                                log_metric=True,
                                ema = ema)
        write_metrics(train_metrics, epoch, 'train', writer)
        scheduler.step()
        # Use Shadow model to validate and save model
        if ema is not None:
            ema.apply_shadow()
            
        save_states(model = model, 
                    optimizer = optimizer, 
                    save_path = os.path.join(saving_model_root, f'{epoch + 1}.pth'),
                    ema = ema)
        # Validating
        val_metrics = validation(model = model, 
                                 dataloader = val_dataloader, 
                                 iou_threshold = iou_threshold, 
                                 device = device, 
                                 enable_progress_bar = True,
                                 log_metric = True)
        write_metrics(val_metrics, epoch, 'val', writer)
        metric = val_metrics[best_model_metric_name]
        if metric >= best_metric:
            best_metric = metric
            best_epoch = epoch + 1
            logger.info(f'Save best model in epoch {epoch + 1} for {best_model_metric_name} = {best_metric:.3f}')
            with open(best_txt_path, 'w') as f:
                f.write(f'{epoch + 1}\n')
                f.write(f'{best_model_metric_name} = {best_metric:.3f}\n')
                f.write('-' * 20 + '\n')
                for metric, value in val_metrics.items():
                    f.write(f'{metric} = {value:.3f}\n')
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
    
    result_csv_path = os.path.join(exp_root, f'{exp_name}_thrs{test_iou_threshold:.2f}.csv')
    
    # Validating with validation set
    stage1_results = test(model = model,
                          dataloader = val_dataloader,
                          iou_threshold = test_iou_threshold, 
                          nodule_3d_minimum_size = test_nodule_3d_minimum_size, 
                          device = device, 
                          log_metric = True)
    write_lines(result_csv_path, 'iou_threshold,val_txt_path')
    write_lines(result_csv_path, '{:.2f},{}'.format(test_iou_threshold, val_set_path))
    nodule_metrics = NoduleMetrics(stage1_results)
    write_lines(result_csv_path, nodule_metrics.generate_metric_csv_lines())
    
    # Validating with test set
    for i in range(len(test_set_paths)):
        test_set_path = test_set_paths[i]
        test_dataset = test_datasets[i]
        test_dataloader = DataLoader(test_dataset, batch_size = 1)
        
        stage1_results = test(model = model,
                            dataloader = test_dataloader,
                            iou_threshold = test_iou_threshold, 
                            nodule_3d_minimum_size = test_nodule_3d_minimum_size, 
                            device = device, 
                            log_metric = True)
        write_lines(result_csv_path, 'iou_threshold,val_txt_path')
        write_lines(result_csv_path, '{:.2f},{}'.format(test_iou_threshold, test_set_path))
        nodule_metrics = NoduleMetrics(stage1_results)
        write_lines(result_csv_path, nodule_metrics.generate_metric_csv_lines())
    
    # Write metrics to tensorboard
    write_metrics(stage1_results['all'], 0, 'test', writer)
    writer.close()