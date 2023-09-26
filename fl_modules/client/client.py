import os
from os.path import join
import copy
import json
import logging
from typing import List, Dict, Any

import torch
from fl_modules.utilities import build_instance, write_yaml
from fl_modules.inference.nodule_counter import NoduleCounter
logger = logging.getLogger(__name__)

class Client:
    def __init__(self, 
                 name: str,
                 client_folder: str,
                 client_config: Dict[str, Any],
                 dataset_params_config: Dict[str, Dict[str, Any]],
                 model,
                 optimizer,
                 ema,
                 device: torch.device,
                 enable_progress_bar: bool):
        self.name = name
        self.client_folder = client_folder
        os.makedirs(self.client_folder, exist_ok = True)
        self.client_config = client_config
        self.dataset_params_config = dataset_params_config
        
        self.model = model
        self.optimizer = optimizer
        self.ema = ema
        self.device = device
        self.enable_progress_bar = enable_progress_bar
        self.prepare()
    
    def prepare(self):
        self._build_dataset_config()    
        
    def _build_dataset_config(self):
        # Prepare dataset config
        counter = NoduleCounter()
        self.dataset_config = dict()
        for key in self.dataset_params_config.keys():
            series_list_path = self.dataset_params_config[key]['series_list_path']
            num_nodule = counter.count_and_analyze_nodules_of_multi_series(series_list_path, self.client_config['nodule_size_ranges'])
            
            # Create dataset config for different dataset
            config = copy.deepcopy(self.client_config['dataset']['params'])
            config.update(self.dataset_params_config[key])
            config['dataset_type'] = key
            config['nodule_size_ranges'] = self.client_config['nodule_size_ranges']
            config['num_nodules'] = num_nodule
            
            setattr(self, f'{key}_series_list_path', series_list_path)
            setattr(self, f'num_nodule_of_{key}_set', num_nodule)
            self.dataset_config[key] = config
        # Write dataset config to yaml file
        write_yaml(join(self.client_folder, 'plan', 'dataset_config.yaml'), self.dataset_config, default_flow_style = None)

    def build_action(self, action_fn, action_config: Dict[str, Any], action_name: str):
        action_config = copy.deepcopy(action_config) if action_config != None else dict()
        action_config['device'] = self.device
        action_config['enable_progress_bar'] = self.enable_progress_bar
        
        setattr(self, f'{action_name}_config', action_config)
        setattr(self, f'{action_name}_fn', action_fn)
        
    def train(self, round_number: int, model, optimizer, ema):
        logger.info(f"Client '{self.name}' starts training!")
        # Lazy initialize dataset
        if self.train_config.get('dataset', None) == None:
            self.train_set = build_instance(self.client_config['dataset']['template'], self.dataset_config['train'])
            self.train_config['dataset'] = self.train_set
        
        self.train_config['model'] = model
        self.train_config['optimizer'] = optimizer
        self.train_config['ema'] = ema
        train_metrics = self.train_fn(**self.train_config)
        
        self.save_metrics(train_metrics, 'train', round_number)
        return train_metrics
    
    def val(self, round_number: int, model, is_global: bool = False):
        logger.info(f"Client '{self.name}' starts validation!")
        # Lazy initialize dataset
        if self.val_config.get('dataset', None) == None:
            self.val_set = build_instance(self.client_config['dataset']['template'], self.dataset_config['val'])
            self.val_config['dataset'] = self.val_set
            
        self.val_config['model'] = model
        val_metrics = self.val_fn(**self.val_config)
        
        # Save metrics
        task = 'val_global' if is_global else 'val_local'
        self.save_metrics(val_metrics, task, round_number)
        return val_metrics
    
    def test(self, model):
        logger.info(f'Client {self.name} starts testing!')
        # Lazy initialize dataset
        if self.test_config.get('dataset', None) == None:
            self.test_set = build_instance(self.client_config['dataset']['template'], self.dataset_config['test'])
            self.test_config['dataset'] = self.test_set
            
        self.test_config['model'] = model
        test_metrics = self.test_fn(**self.test_config)
        
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