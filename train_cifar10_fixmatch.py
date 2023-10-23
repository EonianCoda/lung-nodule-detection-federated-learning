import os
import argparse
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from fl_modules.client.cirfar10_logic import train_fixmatch, validation, test
from fl_modules.dataset.cifar10_dataset import Cifar10SupervisedDataset, Cifar10UnsupervisedDataset
from fl_modules.dataset.utils import prepare_cifar10_datasets
from fl_modules.model.ema import EMA

from fl_modules.utilities import build_instance, get_local_time_in_taiwan, setup_logging, write_yaml, init_seed

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default = '')
    parser.add_argument('--bs', type = int, default = 64)
    parser.add_argument('--num_epoch', type = int, default = 200)
    parser.add_argument('--lr', type = float, default = 0.03)
    parser.add_argument('--optimizer', default = 'sgd')
    parser.add_argument('--weight_decay', type = float, default = 0.0005)
    parser.add_argument('--unsupervised_conf_thrs', type = float, default = 0.95)
    parser.add_argument('--supervised_ratio', type = float, default = 0.05)
    parser.add_argument('--supervised_augment', action='store_true', default=False)
    parser.add_argument('--seed', type = int, default = 1029)
    parser.add_argument('--scheduler', action='store_true', default=False)
    parser.add_argument('--model', default='fl_modules.model.wide_resnet.WideResNet')
    parser.add_argument('--merge_supervised', action='store_true', default=False)
    parser.add_argument('--unsupervised_bs_multiplier', type = int, default = 7)
    parser.add_argument('--apply_ema', action='store_true', default=False)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--resume_model_path', type=str, default='')
    parser.add_argument('--best_model_metric_name', type=str, default='accuracy')
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
    
def get_optimizer(model: nn.Module, 
                  optimizer_type: str,
                  learning_rate: float, 
                  weight_decay: float = 0.0):
    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(grouped_parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(grouped_parameters, lr=learning_rate, weight_decay=weight_decay, momentum=0.9, nesterov=True)
    return optimizer
    
if __name__ == '__main__':
    args = get_parser()
    seed = args.seed
    exp_name = args.exp_name
    train_bs = args.bs
    val_bs = 64 if train_bs <= 64 else train_bs
    num_epoch = args.num_epoch
    weight_decay = args.weight_decay
    supervised_ratio = args.supervised_ratio
    supervised_augment = args.supervised_augment
    unsupervised_conf_thrs = args.unsupervised_conf_thrs
    merge_supervised = args.merge_supervised
    learning_rate = args.lr
    apply_ema = args.apply_ema
    resume_model_path = args.resume_model_path
    best_model_metric_name = args.best_model_metric_name
    unsupervised_bs_multiplier= args.unsupervised_bs_multiplier
    
    # Set seed
    init_seed(seed)
    
    # Experiment Name
    cur_time = get_local_time_in_taiwan()
    timestamp = "[%d-%02d-%02d-%02d%02d]" % (cur_time.year, 
                                            cur_time.month, 
                                            cur_time.day, 
                                            cur_time.hour, 
                                            cur_time.minute)
    if exp_name == '':
        exp_name = timestamp
    else:
        exp_name = f'{timestamp}_{exp_name}'
    exp_root = f'./save/cifar10_fixmatch/{exp_name}'
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Build model and optimizer
    if resume_model_path != '': # Resume model
        checkpoint = torch.load(resume_model_path, map_location = device)
        if 'model_structure' in checkpoint:
            model = checkpoint['model_structure']
        
        optimizer = get_optimizer(model, args.optimizer, learning_rate, weight_decay)
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
        model = build_instance(args.model)
        model = model.to(device)
        
        optimizer = get_optimizer(model, args.optimizer, learning_rate, weight_decay)
        start_epoch = 0
        end_epoch = num_epoch
        # Register EMA
        if apply_ema:
            ema = EMA(model, decay = args.ema_decay)  
            ema.register()
        else:
            ema = None
    
    saving_model_root = os.path.join(exp_root, 'model')
    log_txt_path = os.path.join(exp_root, 'log.txt')
    setup_logging(log_file=log_txt_path)
    # Init Experiment folder
    logger.info("Save model into '{}'".format(saving_model_root))
    os.makedirs(saving_model_root, exist_ok=True)
    if resume_model_path == '':
        write_yaml(os.path.join(exp_root, 'setting.yaml'), vars(args))
    
    # Get dataset
    train_s, train_u, val_set, test_set = prepare_cifar10_datasets(train_val_test_split = [0.8, 0.1, 0.1], 
                                                                   s_u_split=[supervised_ratio, 1.0 - supervised_ratio],
                                                                   seed=seed, 
                                                                   num_clients=1)
    train_s = train_s[0]
    train_u = train_u[0]
    train_s_dataset = Cifar10SupervisedDataset(dataset_type = 'train',
                                               data = train_s,
                                               do_augment = supervised_augment)
    # Merge supervised data into unsupervised data
    if merge_supervised:
        train_u['x'] = np.concatenate([train_u['x'], train_s['x'].copy()])
    
    train_u_dataset = Cifar10UnsupervisedDataset(dataset_type = 'train',
                                                 data = train_u,
                                                targets = ['weak', 'strong'])
    val_dataset = Cifar10SupervisedDataset(dataset_type = 'val', 
                                           data = val_set)
    test_dataset = Cifar10SupervisedDataset(dataset_type = 'test',
                                            data = test_set)


    # Get dataloader
    dataloader_s = DataLoader(train_s_dataset, batch_size = train_bs, shuffle = True, num_workers = os.cpu_count() // 2, drop_last = True)
    dataloader_u = DataLoader(train_u_dataset, batch_size = train_bs * unsupervised_bs_multiplier, shuffle = True, num_workers = os.cpu_count() // 2, drop_last = True)
    iter_dataloader_s = iter(dataloader_s)
    iter_dataloader_u = iter(dataloader_u)

    # Training
    if args.scheduler:
        num_of_total_training_steps = (len(train_s_dataset) / train_bs) * num_epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = num_of_total_training_steps, eta_min=0)
    else:
        scheduler = None
        
    writer = SummaryWriter(log_dir = os.path.join(exp_root, 'tensorboard'))
    best_epoch = 0
    best_metric = 0.0
    last_best_txt = ''
    
    model = model.to(device)
    for epoch in range(start_epoch, end_epoch):
        logger.info("Epoch {}/{}:".format(epoch + 1, end_epoch))
        # Train
        train_metrics = train_fixmatch(model = model,
                                       iter_dataloader_s = iter_dataloader_s,
                                        iter_dataloader_u = iter_dataloader_u,
                                        optimizer = optimizer,
                                        scheduler = scheduler,
                                        epoch = epoch,
                                        num_epochs = end_epoch,
                                        unsupervised_conf_thrs=unsupervised_conf_thrs,
                                        device = device,
                                        enable_progress_bar=True,
                                        log_metric=True,
                                        ema = ema)
        write_metrics(train_metrics, epoch, 'train', writer)
        
        # Use Shadow model to validate and save model
        if ema is not None:
            ema.apply_shadow()
        
        if (epoch != 0 and epoch % 50 == 0) or (epoch == end_epoch - 1):
            save_states(model = model, 
                        optimizer = optimizer, 
                        save_path = os.path.join(saving_model_root, f'{epoch + 1}.pth'),
                        ema = ema)
        # Validating
        val_metrics = validation(model, val_dataset, bs=val_bs,  device=device, enable_progress_bar=True,log_metric=True)
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
    
    test_metrics = test(model, test_dataset, bs=val_bs, device=device, enable_progress_bar=True,log_metric=True)
    # Write metrics to tensorboard
    write_metrics(test_metrics, epoch, 'test', writer)
    writer.close()