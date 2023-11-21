import os
import logging
from typing import Dict, List
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
import torch.nn.functional as F

from fl_modules.utilities import get_progress_bar
from fl_modules.model.ema import EMA

logger = logging.getLogger(__name__)
PRINT_EVERY = 1000

LAMBDA_S = 1 # supervised learning loss ratio
LAMBDA_I = 1e-2 # inter-client consistency
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

def train_fedmatch(model: nn.Module, 
                    dataloader_s: DataLoader,
                    dataloader_u: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    num_steps: int,
                    device: torch.device,
                    helper_agent_models: List[nn.Module] = [],
                    iter_dataloader_s = None,
                    iter_dataloader_u = None,
                    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                    unsupervised_conf_thrs: float = 0.95,
                    ema: EMA = None,
                    enable_progress_bar = False,
                    log_metric = False) -> Dict[str, float]:
    model.train()
    optimizer.zero_grad()
    
    progress_bar = get_progress_bar('Train', num_steps, 0, 1) if enable_progress_bar else None

    num_helper_agents = len(helper_agent_models)
    for helper_agent_model in helper_agent_models:
        helper_agent_model.eval()
        
    losses_s = AverageMeter()
    losses_u = AverageMeter()
    losses = AverageMeter()
    accs_s = AverageMeter()
    if num_helper_agents > 0:
        losses_icc = AverageMeter()
    
    
    if iter_dataloader_s is None:
        iter_dataloader_s = iter(dataloader_s)
    if iter_dataloader_u is None:
        iter_dataloader_u = iter(dataloader_u)
    
    for step in range(num_steps):
        try:
            x_s, targets_s = next(iter_dataloader_s)
        except StopIteration:
            iter_dataloader_s = iter(dataloader_s)
            x_s, targets_s = next(iter_dataloader_s)
        targets_s = targets_s.long().to(device)
        try:
            (x_u_w, x_u_s), _  = next(iter_dataloader_u)
        except StopIteration:
            iter_dataloader_u = iter(dataloader_u)
            (x_u_w, x_u_s), _ = next(iter_dataloader_u)
        
        x = torch.cat((x_s, x_u_w, x_u_s), dim=0).to(device)
        # Predict
        logits = model(x)
        logits_s = logits[:x_s.shape[0]]
        logits_u_w, logits_u_s = logits[x_s.shape[0]:].chunk(2)
        del logits
        # Calculate supervised loss
        loss_s = F.cross_entropy(logits_s, targets_s, reduction='mean') * LAMBDA_S
        loss_final = loss_s
        
        # Calculate unsupervised loss
        pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)
        max_probs_u, targets_u = torch.max(pseudo_label, dim=-1)
        mask = (max_probs_u >= unsupervised_conf_thrs).float()
        if num_helper_agents > 0 and torch.sum(mask) > 0:
            loss_u = 0
            loss_icc = 0
            x_u_w = x[x_s.shape[0]:x_s.shape[0] + x_u_w.shape[0]][mask == 1]
            logits_u_w = logits_u_w[mask == 1]
            with torch.no_grad():
                logits_helper = [helper_agent_model(x_u_w) for helper_agent_model in helper_agent_models]
                
            for logits in logits_helper:
                loss_icc = loss_icc + F.kl_div(F.log_softmax(logits_u_w, dim=-1), F.softmax(logits, dim=-1), reduction='batchmean') * LAMBDA_I / num_helper_agents
                
            # Agreement-based Pseudo Labeling
            with torch.no_grad():
                pseudo_labels = torch.sum(torch.stack([F.one_hot(torch.argmax(logits, dim=-1), num_classes=10) for logits in logits_helper], axis=0), axis=0)
                pseudo_label = F.one_hot(targets_u[mask == 1], num_classes=10)
                targets_u = pseudo_label + pseudo_labels
                targets_u = torch.argmax(targets_u, dim=-1)

            loss_u = F.cross_entropy(logits_u_s[mask == 1], targets_u, reduction='mean') * LAMBDA_U
            loss_final = loss_final + loss_u + loss_icc
        elif torch.sum(mask) > 0:
            loss_u = F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask
            loss_u = loss_u.mean() * LAMBDA_U
            loss_final = loss_final + loss_u
        else:
            loss_u = torch.tensor(0.0)
            if num_helper_agents > 0:
                loss_icc = torch.tensor(0.0)
        # Backward
        loss_final.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
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
        if num_helper_agents > 0:
            losses_icc.update(loss_icc.item())
        acc = torch.sum(torch.argmax(logits_s, dim=1) == targets_s).item() / x_s.shape[0]
        accs_s.update(acc)
        
        post_fix = {'loss': losses.avg,
                    'loss_s': losses_s.avg,
                    'loss_u': losses_u.avg,
                    'acc_s': accs_s.avg}
        if num_helper_agents > 0:
            post_fix['loss_icc'] = losses_icc.avg
        if scheduler is not None:
            post_fix['lr'] = scheduler.get_last_lr()[0]
        if progress_bar is not None:
            progress_bar.set_postfix(**post_fix)
            progress_bar.update()
    if progress_bar is not None:
        progress_bar.close()

    train_metrics = {'acc_s': accs_s.avg,
                    'loss': losses.avg,
                    'loss_s': losses_s.avg,
                    'loss_u': losses_u.avg}
    
    if log_metric:
        for metric, value in train_metrics.items():
            logger.info("Train metric '{}' = {:.4f}".format(metric, value))
    return train_metrics

def train_fixmatch(model: nn.Module, 
                    dataloader_s: DataLoader,
                    dataloader_u: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    num_steps: int,
                    device: torch.device,
                    iter_dataloader_s = None,
                    iter_dataloader_u = None,
                    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                    unsupervised_conf_thrs: float = 0.95,
                    ema: EMA = None,
                    enable_progress_bar = False,
                    log_metric = False) -> Dict[str, float]:
    model.train()
    optimizer.zero_grad()
    
    progress_bar = get_progress_bar('Train', num_steps, 0, 1) if enable_progress_bar else None

    losses_s = AverageMeter()
    losses_u = AverageMeter()
    losses = AverageMeter()
    accs_s = AverageMeter()
    
    if iter_dataloader_s is None:
        iter_dataloader_s = iter(dataloader_s)
    if iter_dataloader_u is None:
        iter_dataloader_u = iter(dataloader_u)
    
    for step in range(num_steps):
        try:
            x_s, targets_s = next(iter_dataloader_s)
        except StopIteration:
            iter_dataloader_s = iter(dataloader_s)
            x_s, targets_s = next(iter_dataloader_s)
        targets_s = targets_s.long().to(device)
        try:
            (x_u_w, x_u_s), _  = next(iter_dataloader_u)
        except StopIteration:
            iter_dataloader_u = iter(dataloader_u)
            (x_u_w, x_u_s), _ = next(iter_dataloader_u)
        
        x = torch.cat((x_s, x_u_w, x_u_s), dim=0).to(device)
        
        # Predict
        logits = model(x)
        logits_s = logits[:x_s.shape[0]]
        logits_u_w, logits_u_s = logits[x_s.shape[0]:].chunk(2)
        del logits
        # Calculate supervised loss
        loss_s = F.cross_entropy(logits_s, targets_s, reduction='mean') * LAMBDA_S
        loss_final = loss_s
        
        # Calculate unsupervised loss
        pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)
        max_probs_u, targets_u = torch.max(pseudo_label, dim=-1)
        mask = (max_probs_u >= unsupervised_conf_thrs).float()
        loss_u = F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask
        loss_u = loss_u.mean() * LAMBDA_U
        loss_final = loss_final + loss_u
        
        # Backward
        loss_final.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
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
        acc = torch.sum(torch.argmax(logits_s, dim=1) == targets_s).item() / x_s.shape[0]
        accs_s.update(acc)
        
        post_fix = {'loss': losses.avg,
                    'loss_s': losses_s.avg,
                    'loss_u': losses_u.avg,
                    'acc_s': accs_s.avg}
        if scheduler is not None:
            post_fix['lr'] = scheduler.get_last_lr()[0]
        if progress_bar is not None:
            progress_bar.set_postfix(**post_fix)
            progress_bar.update()
    if progress_bar is not None:
        progress_bar.close()
                
    train_metrics = {'acc_s': accs_s.avg,
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
                num_steps: int,
                scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                ema: EMA = None,
                enable_progress_bar = False,
                log_metric = False) -> Dict[str, float]:
                
    model.train()
    optimizer.zero_grad()
    
    progress_bar = get_progress_bar('Train', len(dataloader), 0, 1) if enable_progress_bar else None
    losses = AverageMeter()
    accs = AverageMeter()
    
    iter_dataloader = iter(dataloader)
    for step in range(num_steps):
        try:
            x, targets = next(iter_dataloader)
        except StopIteration:
            iter_dataloader = iter(dataloader)
            x, targets = next(iter_dataloader)
        x, targets = x.to(device), targets.long().to(device)
        
        # Calculate supervised loss
        logits = model(x)
        loss = F.cross_entropy(logits, targets, reduction='mean')
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
        # Update EMA
        if ema is not None:
            ema.update()
        
        # Log loss and metrics
        losses.update(loss.item())
        acc = torch.sum(torch.argmax(logits, dim=1) == targets).item() / x.shape[0]
        accs.update(acc)
        
        post_fix = {'loss': losses.avg,
                    'acc': accs.avg}
        if scheduler is not None:
            post_fix['lr'] = scheduler.get_last_lr()[0]
            
        if progress_bar is not None:
            progress_bar.set_postfix(**post_fix)
            progress_bar.update()
    if progress_bar is not None:
        progress_bar.close()
                
    train_metrics = {'acc_s': accs.avg,
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
    progress_bar = get_progress_bar('Validation', len(dataloader), 0, 1) if enable_progress_bar else None
    losses = AverageMeter()
    accs = AverageMeter()
    for step, (x, target) in enumerate(dataloader):
        x, target = x.to(device), target.long().to(device)
        with torch.no_grad():
            logits = model(x)
        
        loss = F.cross_entropy(logits, target, reduction='mean')
        losses.update(loss.item())
        acc = torch.sum(torch.argmax(logits, dim=1) == target).item() / x.shape[0]
        accs.update(acc)
        
        postfix = {'loss': losses.avg,
                    'accuracy': accs.avg}
        if progress_bar is not None:
            progress_bar.set_postfix(**postfix)
            progress_bar.update()
    if progress_bar is not None:
        progress_bar.close()
    val_metrics = {'loss': losses.avg,
                    'accuracy': accs.avg}
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