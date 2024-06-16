import os
from os.path import join
import copy
import json
import pickle
import logging
from typing import List, Dict, Any, Tuple

import torch
import torchvision
from torch.utils.data import DataLoader

import fl_modules.dataset.transform as transform
from fl_modules.utilities import build_instance, write_yaml, build_config
from fl_modules.inference.nodule_counter import NoduleCounter
from fl_modules.dataset.collate import train_collate_fn, infer_collate_fn, unlabeled_tta_train_tracking_collate_fn, infer_aug_collate_fn

logger = logging.getLogger(__name__)

def build_train_augmentation(crop_size: Tuple[int, int, int]):
    rot_zy = (crop_size[0] == crop_size[1] == crop_size[2])
    rot_zx = (crop_size[0] == crop_size[1] == crop_size[2])
        
    transform_list_train = [transform.RandomFlip(p=0.5, flip_depth=True, flip_height=True, flip_width=True)]
    transform_list_train.append(transform.RandomRotate90(p=0.5, rot_xy=True, rot_xz=rot_zx, rot_yz=rot_zy))
        
    transform_list_train.append(transform.CoordToAnnot())
                            
    train_transform = torchvision.transforms.Compose(transform_list_train)
    return train_transform

def build_strong_augmentation(crop_size: Tuple[int, int, int]):
    rot_zy = (crop_size[0] == crop_size[1] == crop_size[2])
    rot_zx = (crop_size[0] == crop_size[1] == crop_size[2])
        
    transform_list_train = [transform.SemiRandomFlip(p=0.5, flip_depth=True, flip_height=True, flip_width=True)]
    transform_list_train.append(transform.RandomBlurNodule(p=0.5, offset=2))
    transform_list_train.append(transform.SemiRandomRotate90(p=0.5, rot_xy=True, rot_xz=rot_zx, rot_yz=rot_zy))
        
    transform_list_train.append(transform.SemiCoordToAnnot())
                            
    train_transform = torchvision.transforms.Compose(transform_list_train)
    return train_transform

class Client:
    def __init__(self, 
                 name: str,
                 client_folder: str,
                 client_config: Dict[str, Any],
                 dataset_params_config: Dict[str, Dict[str, Any]],
                 device: torch.device,
                 save_local_state: bool = False):
        self.name = name
        # Create client folder
        self.client_folder = client_folder
        os.makedirs(self.client_folder, exist_ok = True)
        
        self.client_config = client_config
        self.dataset_params_config = dataset_params_config
        
        self.device = device
        self.save_local_state = save_local_state
    
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
            elif key == 'unlabeled_train':
                target = 'unlabeled_train_dataset'
            elif key == 'unlabeled_det':
                target = 'unlabeled_det_dataset'
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
        
    def gen_pseudo_labels(self, model, detection_postprocess, epoch: int):
        logger.info(f"Client '{self.name}' starts generating pseudo labels!")
        # Lazy initialize dataset
        if self.pseudo_label_config.get('dataloader', None) == None:
            config = copy.deepcopy(self.dataset_config['unlabeled_det'])
            config = build_config(config)
            
            self.pseudo_label_set = build_instance(self.client_config['unlabeled_det_dataset']['template'], config)
            batch_size = self.pseudo_label_config.get('batch_size', 1)
            num_workers = min(batch_size, 4)
            
            self.pseudo_label_dataloader = DataLoader(self.pseudo_label_set,
                                                        batch_size = batch_size,
                                                        num_workers = num_workers,
                                                        collate_fn = infer_collate_fn,
                                                        shuffle = False,
                                                        drop_last = False,
                                                        pin_memory = True)
            self.pseudo_label_config['dataloader'] = self.pseudo_label_dataloader
        
        self.pseudo_label_config['model'] = model
        self.pseudo_label_config['detection_postprocess'] = detection_postprocess
        pseudo_labels = self.pseudo_label_fn(**self.pseudo_label_config)
        
        save_path = join(self.client_folder, 'pseudo_label', f'pseu_labels_epoch_{epoch}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(pseudo_labels, f)
        
        dataloader_u = self.train_config['dataloader_u']
        dataloader_u.dataset.set_pseu_labels(pseudo_labels)
        
    def train(self, round_number: int, num_epoch:int, model_t, model_s, loss_fn, semi_loss_fn, optimizer, detection_postprocess):
        logger.info(f"Client '{self.name}' starts training!")
        # Lazy initialize dataset
        self._init_train_dataloader()
                                                    
        self.train_config['model_t'] = model_t
        self.train_config['model_s'] = model_s
        self.train_config['detection_loss'] = loss_fn
        self.train_config['unsupervised_detection_loss'] = semi_loss_fn
        self.train_config['detection_postprocess'] = detection_postprocess
        
        self.train_config['optimizer'] = optimizer
        for epoch in range(num_epoch):
            train_metrics = self.train_fn(**self.train_config)
        
        self.save_metrics(train_metrics, 'train', round_number)
        
        # Update Pseudo Labels
        train_loader_u = self.train_config['dataloader_u']
        original_num_unlabeled = len(train_loader_u.dataset)
        ema_update_labels_save_path = join(self.client_folder, 'ema_update_labels', f'ema_updated_labels_{epoch}.pkl')
        os.makedirs(os.path.dirname(ema_update_labels_save_path), exist_ok=True)
        with open(ema_update_labels_save_path, 'wb') as f:
            pickle.dump(train_loader_u.dataset.ema_updated_labels, f)
        
        train_loader_u.dataset.confirm_pseudo_labels()
        
        psuedo_label_save_path = os.path.join(self.client_folder, 'history_psuedo_labels', f'history_psuedo_labels_{epoch}.pkl')
        os.makedirs(os.path.dirname(psuedo_label_save_path), exist_ok=True)
        with open(psuedo_label_save_path, 'wb') as f:
            pickle.dump(train_loader_u.dataset.labels, f)
            
        new_num_unlabeled = len(train_loader_u.dataset)
        logger.info('After setting pseudo labels, the number of unlabeled samples is changed from {} to {}'.format(original_num_unlabeled, new_num_unlabeled))
        
        return train_metrics
    
    def _init_train_dataloader(self):
        if self.train_config.get('dataloader_l', None) == None or self.train_config.get('dataloader_u', None) == None:
            # Build labeled train dataset
            config = copy.deepcopy(self.dataset_config['train'])
            config = build_config(config)
            crop_size = self.client_config['train_dataset']['params']['crop_fn']['params']['crop_size']
            transform_post = build_train_augmentation(crop_size)
            config['transform_post'] = transform_post
            self.train_set = build_instance(self.client_config['train_dataset']['template'], config)
            
            batch_size = self.train_config.get('batch_size', 1)
            self.train_dataloader_l = DataLoader(self.train_set, 
                                                batch_size = batch_size,
                                                num_workers = 2, ##TODO: Test num_workers
                                                collate_fn = train_collate_fn,
                                                shuffle = True,
                                                drop_last=True,
                                                pin_memory = True,
                                                persistent_workers=True)
            self.train_config['dataloader_l'] = self.train_dataloader_l
            
            # Build unlabeled train dataset
            config = copy.deepcopy(self.dataset_config['unlabeled_train'])
            config = build_config(config)
            crop_size = self.client_config['unlabeled_train_dataset']['params']['crop_fn']['params']['crop_size']
            transform_post = build_strong_augmentation(crop_size)
            config['transform_post'] = transform_post
            self.unlabeled_train_set = build_instance(self.client_config['unlabeled_train_dataset']['template'], config)
            
            batch_size = self.train_config.get('batch_size', 1)
            num_workers = min(batch_size, 4)
            self.train_dataloader_u = DataLoader(self.unlabeled_train_set,
                                                 batch_size = batch_size,
                                                num_workers = num_workers,
                                                collate_fn = unlabeled_tta_train_tracking_collate_fn,
                                                shuffle=True,
                                                drop_last=True,
                                                pin_memory = True)
            self.train_config['dataloader_u'] = self.train_dataloader_u
    
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
    
    def save_model_state(self, model, round_number: int, save_teacher: bool = False):
        if save_teacher:
            model_folder = join(self.client_folder, 'model_t')
        else:
            model_folder = join(self.client_folder, 'model_s')
        
        save_path = join(model_folder, f'{round_number}.pt')
        os.makedirs(os.path.dirname(save_path), exist_ok = True)
        torch.save(model.state_dict(), save_path)
        if not self.save_local_state:
            for round_number in range(round_number - 1):
                if os.path.exists(join(model_folder, f'{round_number}.pt')):
                    os.remove(join(model_folder, f'{round_number}.pt'))
        
    def load_model_state(self, model, round_number: int, device: torch.device, load_teacher: bool = False):
        if load_teacher:
            model_folder = join(self.client_folder, 'model_t')
        else:
            model_folder = join(self.client_folder, 'model_s')
            
        save_path = join(model_folder, f'{round_number}.pt')
        
        model.load_state_dict(torch.load(save_path, map_location = device))
        
    def save_optimizer_state(self, optimizer, round_number: int):
        save_path = join(self.client_folder, 'optimizer', f'{round_number}.pt')
        os.makedirs(os.path.dirname(save_path), exist_ok = True)
        torch.save(optimizer.state_dict(), save_path)
        if not self.save_local_state:
            for round_number in range(round_number - 1):
                if os.path.exists(join(self.client_folder, 'optimizer', f'{round_number}.pt')):
                    os.remove(join(self.client_folder, 'optimizer', f'{round_number}.pt'))
    
    def load_optimizer_state(self, optimizer, round_number: int, device: torch.device):
        save_path = join(self.client_folder, 'optimizer', f'{round_number}.pt')
        optimizer.load_state_dict(torch.load(save_path, map_location = device))
        
    def save_scheduler_state(self, scheduler, round_number: int):
        save_path = join(self.client_folder, 'scheduler', f'{round_number}.pt')
        os.makedirs(os.path.dirname(save_path), exist_ok = True)
        torch.save(scheduler.state_dict(), save_path)
        if not self.save_local_state:
            for round_number in range(round_number - 1):
                if os.path.exists(join(self.client_folder, 'scheduler', f'{round_number}.pt')):
                    os.remove(join(self.client_folder, 'scheduler', f'{round_number}.pt'))
        
    def load_scheduler_state(self, scheduler, round_number: int, device: torch.device):
        save_path = join(self.client_folder, 'scheduler', f'{round_number}.pt')
        scheduler.load_state_dict(torch.load(save_path, map_location = device))
        
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