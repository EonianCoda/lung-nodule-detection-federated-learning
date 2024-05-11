import os
from os.path import join
import copy
import json
import logging
from typing import List, Dict, Any, Tuple

import torch
import torchvision
from torch.utils.data import DataLoader

import fl_modules.dataset.transform as transform
from fl_modules.utilities import build_instance, write_yaml, build_config
from fl_modules.inference.nodule_counter import NoduleCounter
from fl_modules.dataset.collate import train_collate_fn, infer_collate_fn
logger = logging.getLogger(__name__)

def build_train_augmentation(crop_size: Tuple[int, int, int]):
    rot_zy = (crop_size[0] == crop_size[1] == crop_size[2])
    rot_zx = (crop_size[0] == crop_size[1] == crop_size[2])
        
    transform_list_train = [transform.RandomFlip(p=0.5, flip_depth=True, flip_height=True, flip_width=True)]
    transform_list_train.append(transform.RandomRotate90(p=0.5, rot_xy=True, rot_xz=rot_zx, rot_yz=rot_zy))
    transform_list_train.append(transform.CoordToAnnot())
                            
    train_transform = torchvision.transforms.Compose(transform_list_train)
    return train_transform

class Client:
    def __init__(self, 
                 name: str,
                 client_folder: str,
                 client_config: Dict[str, Any],
                 dataset_params_config: Dict[str, Dict[str, Any]],
                 model,
                 optimizer,
                 ema,
                 device: torch.device):
        self.name = name
        # Create client folder
        self.client_folder = client_folder
        os.makedirs(self.client_folder, exist_ok = True)
        
        self.client_config = client_config
        self.dataset_params_config = dataset_params_config
        
        self.model = model
        self.optimizer = optimizer
        self.ema = ema
        self.device = device
    
    def prepare(self):
        self._build_dataset_config()    
        
    def _build_dataset_config(self):
        # Prepare dataset config
        counter = NoduleCounter()
        self.dataset_config = dict()
        min_size = self.client_config['shared_params']['min_size']
        
        for key in self.dataset_params_config.keys():
            series_list_path = self.dataset_params_config[key]['series_list_path']
            num_nodule = counter.count_and_analyze_nodules_of_multi_series(series_list_path, self.client_config['nodule_size_ranges'], min_size = min_size)
            
            if key == 'train':
                target = 'train_dataset'
            else:
                target = 'val_dataset'
            
            # Create dataset config for different dataset
            config = copy.deepcopy(self.client_config[target]['params'])
            config.update(self.dataset_params_config[key])
            config.update(self.client_config['shared_params'])
            
            setattr(self, f'{key}_series_list_path', series_list_path)
            setattr(self, f'num_nodule_of_{key}_set', num_nodule)
            self.dataset_config[key] = config
        # Write dataset config to yaml file
        write_yaml(join(self.client_folder, 'plan', 'dataset_config.yaml'), self.dataset_config, default_flow_style = None)

    def build_action(self, action_fn, action_config: Dict[str, Any], action_name: str):
        action_config = copy.deepcopy(action_config) if action_config != None else dict()
        action_config['device'] = self.device
        
        setattr(self, f'{action_name}_config', action_config)
        setattr(self, f'{action_name}_fn', action_fn)
        
    def train(self, round_number: int, num_epoch:int, model, optimizer, ema):
        logger.info(f"Client '{self.name}' starts training!")
        # Lazy initialize dataset
        if self.train_config.get('dataset', None) == None:
            config = copy.deepcopy(self.dataset_config['train'])
            config = build_config(config)
            transform_post = build_train_augmentation(self.client_config['train_dataset']['params']['crop_fn']['params']['crop_size'])
            config['transform_post'] = transform_post
            self.train_set = build_instance(self.client_config['train_dataset']['template'], config)
            
            batch_size = self.train_config.get('batch_size', 1)
            num_workers = min(batch_size, 4)
            
            self.train_dataloader = DataLoader(self.train_set, 
                                                batch_size = batch_size,
                                                num_workers = num_workers,
                                                collate_fn = train_collate_fn,
                                                shuffle = True,
                                                drop_last=True,
                                                pin_memory = True)
            self.train_config['dataloader'] = self.train_dataloader
        
        self.train_config['model'] = model
        self.train_config['optimizer'] = optimizer
        self.train_config['ema'] = ema
        for epoch in range(num_epoch):
            train_metrics = self.train_fn(**self.train_config)
        
        self.save_metrics(train_metrics, 'train', round_number)
        return train_metrics
    
    def val(self, round_number: int, model, detection_postprocess, is_global: bool = False):
        logger.info(f"Client '{self.name}' starts validation!")
        # Lazy initialize dataset
        if self.val_config.get('dataset', None) == None:
            config = copy.deepcopy(self.dataset_config['val'])
            config = build_config(config)
            
            self.val_set = build_instance(self.client_config['val_dataset']['template'], config)
            batch_size = self.val_config.get('batch_size', 1)
            num_workers = batch_size
            self.val_dataloader = DataLoader(self.val_set, 
                                              batch_size = batch_size,
                                              shuffle = False,
                                              num_workers = num_workers,
                                              collate_fn=infer_collate_fn,
                                              drop_last=False,
                                              pin_memory = True)
            self.val_config['dataloader'] = self.val_dataloader
            
        task = 'val_global' if is_global else 'val_local'
        exp_folder = join(self.client_folder, 'results', task)
        self.val_config['model'] = model
        self.val_config['detection_postprocess'] = detection_postprocess
        val_metrics = self.val_fn(**self.val_config, exp_folder = exp_folder, epoch = round_number, series_list_path = getattr(self, 'val_series_list_path'))
        
        # Save metrics
        self.save_metrics(val_metrics, task, round_number)
        return val_metrics
    
    def test(self, model, detection_postprocess):
        logger.info(f'Client {self.name} starts testing!')
        # Lazy initialize dataset
        if self.test_config.get('dataset', None) == None:
            config = copy.deepcopy(self.dataset_config['test'])
            config = build_config(config)
            
            self.test_set = build_instance(self.client_config['val_dataset']['template'], config)
            batch_size = self.val_config.get('batch_size', 1)
            num_workers = batch_size
            self.test_dataloader = DataLoader(self.test_set, 
                                              batch_size = batch_size,
                                              shuffle = False,
                                              num_workers = num_workers,
                                              collate_fn=infer_collate_fn,
                                              drop_last=False,
                                              pin_memory = True)
            self.test_config['dataloader'] = self.test_dataloader
            
        exp_folder = join(self.client_folder, 'results', 'test')
        self.test_config['model'] = model
        self.test_config['detection_postprocess'] = detection_postprocess
        test_metrics = self.test_fn(**self.test_config, exp_folder = exp_folder, epoch = 'test', series_list_path = getattr(self, 'test_series_list_path'))
        
        return test_metrics
    
    def save_model_state(self, model, round_number: int):
        save_path = join(self.client_folder, 'model', f'{round_number}.pt')
        os.makedirs(os.path.dirname(save_path), exist_ok = True)
        torch.save(model.state_dict(), save_path)
        
    def load_model_state(self, model, round_number: int, device: torch.device):
        save_path = join(self.client_folder, 'model', f'{round_number}.pt')
        model.load_state_dict(torch.load(save_path, map_location = device))
        
    def save_optimizer_state(self, optimizer, round_number: int):
        save_path = join(self.client_folder, 'optimizer', f'{round_number}.pt')
        os.makedirs(os.path.dirname(save_path), exist_ok = True)
        torch.save(optimizer.state_dict(), save_path)
    
    def load_optimizer_state(self, optimizer, round_number: int, device: torch.device):
        save_path = join(self.client_folder, 'optimizer', f'{round_number}.pt')
        optimizer.load_state_dict(torch.load(save_path, map_location = device))
        
    def save_scheduler_state(self, scheduler, round_number: int):
        save_path = join(self.client_folder, 'scheduler', f'{round_number}.pt')
        os.makedirs(os.path.dirname(save_path), exist_ok = True)
        torch.save(scheduler.state_dict(), save_path)
        
    def load_scheduler_state(self, scheduler, round_number: int, device: torch.device):
        save_path = join(self.client_folder, 'scheduler', f'{round_number}.pt')
        scheduler.load_state_dict(torch.load(save_path, map_location = device))
        
    def save_ema_state(self, ema, round_number: int):
        save_path = join(self.client_folder, 'ema', f'{round_number}.pt')
        os.makedirs(os.path.dirname(save_path), exist_ok = True)
        torch.save(ema.state_dict(), save_path)
    
    def load_ema_state(self, ema, round_number: int, device: torch.device):
        save_path = join(self.client_folder, 'ema', f'{round_number}.pt')
        ema.load_state_dict(torch.load(save_path, map_location = device))
    
    def save_metrics(self, metrics: Dict[str, float], task: str, round_number: int):
        """Save metrics to json file
        Args:
            metrics: Dict[str, float]
                metrics to save, e.g. {'loss': 0.1, 'acc': 0.9}
            task: str
                task name, e.g. 'train', 'val_local', 'val_global', 'test'
            round_number: int
                round number
        """
        metrics = copy.deepcopy(metrics)
        for key in metrics.keys():
            metrics[key] = float(metrics[key])
        
        save_path = join(self.client_folder, 'metrics', f'{task}_{round_number}.json')
        os.makedirs(os.path.dirname(save_path), exist_ok = True)
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent = 4)