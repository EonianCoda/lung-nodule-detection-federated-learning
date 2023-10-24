import os
import logging
from typing import Dict
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss

from fl_modules.utilities import get_progress_bar
from fl_modules.model.ema import EMA

logger = logging.getLogger(__name__)
PRINT_EVERY = 1000

LAMBDA_S = 1 # supervised learning loss ratio
LAMBDA_I = 1 # inter-client consistency
LAMBDA_U = 1 # unsupervised learning loss ratio

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_fixmatch(model: nn.Module, 
                    dataloader_s: DataLoader,
                    iter_dataloader_s,
                    dataloader_u: DataLoader,
                    iter_dataloader_u, 
                    optimizer: torch.optim.Optimizer,
                    num_steps: int,
                    device: torch.device,
                    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                    unsupervised_conf_thrs: float = 0.75,
                    ema: EMA = None,
                    enable_progress_bar = False,
                    log_metric = False) -> Dict[str, float]:
    model.train()
    optimizer.zero_grad()
    
    supervised_loss_fn = CrossEntropyLoss()
    progress_bar = get_progress_bar('Train', num_steps, 0, 1) if enable_progress_bar else None

    losses_s = AverageMeter()
    losses_u = AverageMeter()
    losses = AverageMeter()
    corrected_s = AverageMeter()
    for step in range(num_steps):
        try:
            x_s, y_s = next(iter_dataloader_s)
        except StopIteration:
            iter_dataloader_s = iter(dataloader_s)
            x_s, y_s = next(iter_dataloader_s)
        x_s, y_s = x_s.to(device), y_s.to(device)
        y_s_one_hot =  nn.functional.one_hot(y_s.long(), num_classes=10).float() 
        
        try:
            x_u = next(iter_dataloader_u)
        except StopIteration:
            iter_dataloader_u = iter(dataloader_u)
            x_u = next(iter_dataloader_u)
            
        for i in range(len(x_u)):
            x_u[i] = x_u[i].to(device)
        loss_final = 0
        
        # Calculate supervised loss
        y_pred_s = model(x_s)
        loss_s = supervised_loss_fn(y_pred_s, y_s_one_hot) * LAMBDA_S
        loss_final += loss_s
        
        # Calculate unsupervised loss
        y_pred = model(x_u[0])
        conf = torch.where(torch.max(y_pred, dim=1)[0] >= unsupervised_conf_thrs)[0]
        if len(conf) > 0:
            y_pseu = y_pred[conf]
            y_pseu = torch.argmax(y_pseu, dim=1)
            y_pseu_onehot = nn.functional.one_hot(y_pseu, num_classes=10).float()
            loss_u = supervised_loss_fn(model(x_u[1][conf]), y_pseu_onehot) * LAMBDA_U
            loss_final += loss_u
        else:
            loss_u = torch.tensor(0.0)
        loss_final.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
            
        # Update EMA
        if ema is not None:
            ema.update()
        
        # Log loss and metrics
        losses_s.update(loss_s.item())
        losses_u.update(loss_u.item())
        losses.update(loss_final.item())
        corrected_s.update(torch.sum(torch.argmax(y_pred_s, dim=1) == y_s).item()) 
        
        post_fix = {'loss': losses.avg,
                    'loss_s': losses_s.avg,
                    'loss_u': losses_u.avg,
                    'acc_s': corrected_s.avg}
        if scheduler is not None:
            post_fix['lr'] = scheduler.get_last_lr()[0]
        if progress_bar is not None:
            progress_bar.set_postfix(**post_fix)
            progress_bar.update()
    if progress_bar is not None:
        progress_bar.close()
                
    train_metrics = {'acc_s': corrected_s.avg,
                    'loss': losses.avg,
                    'loss_s': losses_s.avg,
                    'loss_u': losses_u.avg}
    
    if log_metric:
        for metric, value in train_metrics.items():
            logger.info("Train metric '{}' = {:.4f}".format(metric, value))
    return train_metrics

def train_normal(model: nn.Module, 
                dataloader: DataLoader,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                ema: EMA = None,
                enable_progress_bar = False,
                log_metric = False) -> Dict[str, float]:
                
    model.train()
    optimizer.zero_grad()
    
    supervised_loss_fn = CrossEntropyLoss()
    # Calculate unsupervised loss    
    if enable_progress_bar:
        progress_bar = get_progress_bar('Train', len(dataloader), 0, 1)
    else:
        progress_bar = None

    losses = AverageMeter()
    corrected = AverageMeter()
    for step, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        y_one_hot =  nn.functional.one_hot(y.long(), num_classes=10).float() 
        
        # Calculate supervised loss
        y_pred = model(x)
        loss = supervised_loss_fn(y_pred, y_one_hot)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()
        # Update EMA
        if ema is not None:
            ema.update()
        
        # Log loss and metrics
        losses.update(loss.item())
        corrected.update(torch.sum(torch.argmax(y_pred, dim=1) == y).item())
        
        post_fix = {'loss': losses.avg,
                    'acc': corrected.avg}
        if scheduler is not None:
            post_fix['lr'] = scheduler.get_last_lr()[0]
            
        if progress_bar is not None:
            progress_bar.set_postfix(**post_fix)
            progress_bar.update()
    if progress_bar is not None:
        progress_bar.close()
                
    train_metrics = {'acc_s': corrected.avg,
                    'loss': losses.avg}
    
    if log_metric:
        for metric, value in train_metrics.items():
            logger.info("Train metric '{}' = {:.4f}".format(metric, value))
    return train_metrics
    
def validation(model: nn.Module, 
               dataloader: DataLoader,
               device: torch.device = torch.device('cpu'),
               enable_progress_bar = False,
               log_metric = False) -> Dict[str, float]:
    model.eval()
    if enable_progress_bar:
        progress_bar = get_progress_bar('Validation', len(dataloader), 0, 1)
    else:
        progress_bar = None
    
    val_losses = AverageMeter()
    val_corrected = AverageMeter()
    for step, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        
        y_one_hot =  nn.functional.one_hot(y.long(), num_classes=10).float()
        val_loss = CrossEntropyLoss()(y_pred, y_one_hot)
    
        val_losses.update(val_loss.item())
        val_corrected.update(torch.sum(torch.argmax(y_pred, dim=1) == y).item())
        
        postfix = {'loss': val_losses.avg,
                    'accuracy': val_corrected.avg}
        if progress_bar is not None:
            progress_bar.set_postfix(**postfix)
            progress_bar.update()
    if progress_bar is not None:
        progress_bar.close()
    val_metrics = {'loss': val_losses.avg,
                    'accuracy': val_corrected.avg}
    if log_metric:
        for metric, value in val_metrics.items():
            logger.info("Val metric '{}' = {:.4f}".format(metric, value))
        
    return val_metrics

def test(model: nn.Module, 
        dataloader: DataLoader,
        device: torch.device = torch.device('cpu'),
        enable_progress_bar = False,
        log_metric = False) -> Dict[str, float]:
    test_metrics = validation(model, dataloader, device, enable_progress_bar, log_metric)
    return test_metrics