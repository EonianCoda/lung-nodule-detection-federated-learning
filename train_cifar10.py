import os
import argparse
import logging

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from fl_modules.client.cirfar10_logic import train_normal, validation, test
from fl_modules.dataset.cifar10_dataset import Cifar10SupervisedDataset
from fl_modules.dataset.utils import prepare_cifar10_datasets

from fl_modules.utilities import setup_logging, write_yaml
from train import build_train, save_states, write_metrics
logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default = '')
    parser.add_argument('--bs', type = int, default = 64)
    
    parser.add_argument('--num_epoch', type = int, default = 200)
    parser.add_argument('--lr', type = float, default = 0.03)
    parser.add_argument('--optimizer', default = 'sgd')
    parser.add_argument('--weight_decay', type = float, default = 0.0001)
    
    parser.add_argument('--model', default='wide_resnet')
    
    parser.add_argument('--apply_scheduler', action='store_true', default=False)
    
    parser.add_argument('--no_ema', action='store_false', default=False)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    
    parser.add_argument('--seed', type = int, default = 1029)
    parser.add_argument('--best_model_metric_name', type=str, default='accuracy')
    parser.add_argument('--resume_model_path', type=str, default='')
    args = parser.parse_args()
    return args

def get_dataloder(args):
    train_bs = args.bs
    val_bs = train_bs * 2
    
    train_s, _ , val_set, test_set = prepare_cifar10_datasets(train_val_test_split = [0.8, 0.1, 0.1], s_u_split=[1.0, 0.0], num_clients = 1, seed=args.seed)
    train_dataset = Cifar10SupervisedDataset(dataset_type = 'train',
                                            data = train_s[0],
                                            do_augment = True)
    val_dataset = Cifar10SupervisedDataset(dataset_type = 'val', 
                                           data = val_set)
    test_dataset = Cifar10SupervisedDataset(dataset_type = 'test',
                                            data = test_set)

    num_workers = os.cpu_count() // 2
    train_dataloader = DataLoader(train_dataset, batch_size = train_bs, shuffle = True, num_workers = num_workers, drop_last = True)
    val_dataloader = DataLoader(val_dataset, batch_size = val_bs, shuffle = False, num_workers = num_workers, drop_last = False)
    test_dataloader = DataLoader(test_dataset, batch_size = val_bs, shuffle = False, num_workers = num_workers, drop_last = False)

    return train_dataloader, val_dataloader, test_dataloader
if __name__ == '__main__':
    args = get_parser()
    best_model_metric_name = args.best_model_metric_name
    val_bs = args.bs * 2
    
    model, optimizer, scheduler, ema, start_epoch, end_epoch, device, exp_root, exp_name = build_train(args)
    saving_model_root = os.path.join(exp_root, 'model')
    setup_logging(log_file=os.path.join(exp_root, 'log.txt'))
    # Init Experiment folder
    logger.info("Save model into '{}'".format(saving_model_root))
    os.makedirs(saving_model_root, exist_ok=True)
    if args.resume_model_path == '':
        write_yaml(os.path.join(exp_root, 'setting.yaml'), vars(args))
    
    # Get dataset
    train_dataloader, val_dataloader, test_dataloader = get_dataloder(args)
    # Training
    writer = SummaryWriter(log_dir = os.path.join(exp_root, 'tensorboard'))
    best_epoch = 0
    best_metric = 0.0
    last_best_txt = ''
    
    model = model.to(device)
    for epoch in range(start_epoch, end_epoch):
        logger.info("Epoch {}/{}:".format(epoch + 1, end_epoch))
        # Train
        train_metrics = train_normal(model = model,
                                       dataloader = train_dataloader,
                                        optimizer = optimizer,
                                        scheduler=scheduler,
                                        ema = ema,
                                        device = device,
                                        log_metric=True,
                                        enable_progress_bar=True)
        write_metrics(train_metrics, epoch, 'train', writer)
        # Use Shadow model to validate and save model
        if ema is not None:
            ema.apply_shadow()
        
        if (epoch != 0 and epoch % 100 == 0) or (epoch == end_epoch - 1):
            save_states(model = model, 
                        optimizer = optimizer, 
                        save_path = os.path.join(saving_model_root, f'{epoch + 1}.pth'),
                        ema = ema)
        # Validating
        val_metrics = validation(model=model, 
                                 dataloader=val_dataloader,
                                 device=device, 
                                 enable_progress_bar=True,
                                 log_metric=True)
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
    
    test_metrics = validation(model=model, 
                                dataloader=test_dataloader,
                                device=device, 
                                enable_progress_bar=True,
                                log_metric=True)
    # Write metrics to tensorboard
    write_metrics(test_metrics, epoch, 'test', writer)
    writer.close()