import os
from os.path import join
import shutil
import copy
import json
import logging
from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader
from fl_modules.utilities import build_instance
logger = logging.getLogger(__name__)

class Client:
    def __init__(self, 
                 name: str,
                 client_folder: str,
                 client_config: Dict[str, Any],
                 model,
                 optimizer,
                 scheduler,
                 ema,
                 device: torch.device,
                 enable_progress_bar: bool):
        self.name = name
        self.client_folder = client_folder
        os.makedirs(self.client_folder, exist_ok = True)
        self.client_config = client_config
        self.dataset_config = self.client_config['dataset']
        
        self.train_config = dict()
        self.val_config = dict()
        self.test_config = dict()
                
        self.model = model
        self.optimizer = optimizer
        self.ema = ema
        self.scheduler = scheduler
        self.device = device
        self.enable_progress_bar = enable_progress_bar

        self.num_workers = max(os.cpu_count() // 4, 2)

    def build_action(self, action_fn, action_config: Dict[str, Any], action_name: str):
        action_config = copy.deepcopy(action_config) if action_config != None else dict()
        action_config['device'] = self.device
        action_config['enable_progress_bar'] = self.enable_progress_bar
        
        setattr(self, f'{action_name}_config', action_config)
        setattr(self, f'{action_name}_fn', action_fn)
        
    def train(self, round_number: int, model, optimizer, ema, scheduler):
        logger.info(f"Client '{self.name}' starts training!")
        # Lazy initialize dataset
        if self.train_config.get('dataset', None) == None:
            train_s_dataset = build_instance(self.dataset_config['train_s']['template'], self.dataset_config['train_s']['params'])
            train_u = build_instance(self.dataset_config['train_u']['template'], self.dataset_config['train_u']['params'])
            
            train_s_dataloder = DataLoader(train_s_dataset,
                                           batch_size=train_s_dataset.batch_size,
                                            shuffle = True,
                                            num_workers = self.num_workers,
                                            pin_memory = True,
                                            # persistent_workers = True,
                                            drop_last = True)
            train_u_dataloader = DataLoader(train_u,
                                            shuffle = True,
                                            num_workers = self.num_workers,
                                            pin_memory = True,
                                            # persistent_workers = True,
                                            drop_last = True)
            self.train_config['dataloader_s'] = train_s_dataloder
            self.train_config['dataloader_u'] = train_u_dataloader
            
        self.train_config['model'] = model
        self.train_config['optimizer'] = optimizer
        self.train_config['ema'] = ema
        self.train_config['scheduler'] = scheduler
        train_metrics = self.train_fn(**self.train_config)
        
        self.save_metrics(train_metrics, 'train', round_number)
        return train_metrics
    
    def val(self, round_number: int, model, is_global: bool = False):
        logger.info(f"Client '{self.name}' starts validation!")
        # Lazy initialize dataset
        if self.val_config.get('dataset', None) == None:
            val_dataset = build_instance(self.dataset_config['val']['template'], self.dataset_config['val']['params'])
            dataloader = DataLoader(val_dataset,
                                    batch_size = val_dataset.batch_size,
                                    shuffle = False,
                                    num_workers = self.num_workers,
                                    drop_last = False)
            self.val_config['dataloader'] = dataloader
            
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
            test_set = build_instance(self.dataset_config['test']['template'], self.dataset_config['test']['params'])
            dataloader = DataLoader(test_set,
                                    batch_size = test_set.batch_size,
                                    shuffle = False,
                                    num_workers = self.num_workers,
                                    drop_last = False)
            self.test_config['dataloader'] = dataloader
            
        self.test_config['model'] = model
        test_metrics = self.test_fn(**self.test_config)
        
        return test_metrics
    
    def save_state_dict(self, 
                        instance,
                        target: str, 
                        round_number: int,
                        remove_previous: bool = True):
        save_path = join(self.client_folder, target, f'{round_number}.pt')
        folder = os.path.dirname(save_path)
        if remove_previous:
            if os.path.exists(folder):
                shutil.rmtree(folder)
        os.makedirs(folder, exist_ok = True)
        torch.save(instance.state_dict(), save_path)
    
    def load_state_dict(self,
                        instance,
                        target: str,
                        round_number: int,
                        device: torch.device):
        save_path = join(self.client_folder, target, f'{round_number}.pt')
        state_dict = torch.load(save_path, map_location = device)
        instance.load_state_dict(state_dict)
    
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
    
    
    @property
    def weight(self):
        return len(self.train_config['dataloader_s']) + len(self.train_config['dataloader_u'])