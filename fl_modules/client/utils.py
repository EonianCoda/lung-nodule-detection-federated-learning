from typing import Dict
import logging
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

def write_metrics(metrics: Dict[str, float], epoch: int, prefix: str, writer: SummaryWriter):
    for metric, value in metrics.items():
        writer.add_scalar(f'{prefix}/{metric}', value, global_step = epoch)
    writer.flush()

def save_states(save_path: str, model: nn.Module, optimizer: torch.optim.Optimizer = None, scheduler: torch.optim.lr_scheduler = None, ema = None, **kwargs):
    save_dict = {'model_state_dict': model.state_dict(), 'model_structure': model}
    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler_state_dict'] = scheduler.state_dict()
        if getattr(scheduler, 'after_scheduler', None) is not None:
            save_dict['after_scheduler_state_dict'] = scheduler.after_scheduler.state_dict()
    if ema is not None:
        save_dict['ema_state_dict'] = ema.state_dict()
    
    for key, value in kwargs.items():
        save_dict['{}_state_dict'.format(key)] = value.state_dict()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(save_dict, save_path)
    
def load_states(load_path: str, device: torch.device, model: nn.Module = None, optimizer: torch.optim.Optimizer = None, scheduler: torch.optim.lr_scheduler = None, ema = None, **kwargs):
    checkpoint = torch.load(load_path, map_location=device)
    
    if model is not None:
        if 'state_dict' not in checkpoint and 'model_state_dict' not in checkpoint:
            model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if getattr(scheduler, 'after_scheduler', None) is not None:
            scheduler.after_scheduler.load_state_dict(checkpoint['after_scheduler_state_dict'])
    
    if ema is not None:
        ema.load_state_dict(checkpoint['ema_state_dict'])
        
    for key, value in kwargs.items():
        state_dict_key = '{}_state_dict'.format(key)
        if state_dict_key not in checkpoint:
            logger.warning(f'Key {key} not found in checkpoint')
            continue
        kwargs[key].load_state_dict(checkpoint[state_dict_key])
        
def load_model(load_path: str):
    checkpoint = torch.load(load_path)
    # Build model
    if 'model_structure' in checkpoint:
        model = checkpoint['model_structure']
    else:
        raise ValueError('Model structure not found in checkpoint')
        
    # Load state dict
    if 'state_dict' not in checkpoint and 'model_state_dict' not in checkpoint:
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    return model

def load_teacher_model(load_path: str):
    checkpoint = torch.load(load_path)
    # Build model
    if 'model_structure' in checkpoint:
        model = checkpoint['model_structure']
    else:
        raise ValueError('Teacher model structure not found in checkpoint')
    # Load state dict
    if 'state_dict' not in checkpoint and 'model_t_state_dict' not in checkpoint:
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint['model_t_state_dict'])
    return model

def get_memory_format(memory_format: str):
    if memory_format == 'channels_last':
        return torch.channels_last_3d
    else:
        return None