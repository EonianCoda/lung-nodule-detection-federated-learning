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
                    iter_dataloader_s,
                    iter_dataloader_u, 
                    optimizer: torch.optim.Optimizer,
                    num_steps: int,
                    device: torch.device,
                    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                    unsupervised_conf_thrs: float = 0.75,
                    epoch: int = 0,
                    num_epochs: int = 0,
                    ema: EMA = None,
                    enable_progress_bar = False,
                    log_metric = False) -> Dict[str, float]:
                
    model.train()
    optimizer.zero_grad()
    
    avg_total_loss, avg_loss_s, avg_loss_u = 0.0, 0.0, 0.0
    avg_supervised_acc = 0.0
    supervised_loss_fn = CrossEntropyLoss()

    if enable_progress_bar:
        progress_bar = get_progress_bar('Train', epoch, num_epochs)
    else:
        progress_bar = None

    running_total_loss, running_loss_s, running_loss_u = 0.0, 0.0, 0.0
    running_supervised_acc = 0.0
    
    losses_s = AverageMeter()
    losses_u = AverageMeter()
    losses = AverageMeter()
    corrected_s = AverageMeter()
    for step in range(num_steps):
        x_s, y_s = next(iter_dataloader_s)
        x_s, y_s = x_s.to(device), y_s.to(device)
        y_s_one_hot =  nn.functional.one_hot(y_s.long(), num_classes=10).float() 
        
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
        if scheduler is not None:
            scheduler.step()
        # Update EMA
        if ema is not None:
            ema.update()
        
        # Log loss and metrics
        losses_s.update(loss_s.item())
        losses_u.update(loss_u.item())
        losses.update(loss_final.item())
        corrected_s.update(torch.sum(torch.argmax(y_pred_s, dim=1) == y_s).item() / len(y_s)) 
        
        avg_loss_s = running_loss_s / (step + 1)
        avg_loss_u = running_loss_u / (step + 1)
        avg_total_loss = running_total_loss / (step + 1)
        avg_supervised_acc = running_supervised_acc / (step + 1)
        if progress_bar is not None:
            progress_bar.set_postfix(loss = avg_total_loss,
                                        loss_s = avg_loss_s,
                                    loss_u = avg_loss_u,
                                    acc_s = avg_supervised_acc)
            progress_bar.update()
    if progress_bar is not None:
        progress_bar.close()
                
    final_avg_loss_s = running_loss_s / num_steps
    final_avg_loss_u = running_loss_u / num_steps
    final_avg_total_loss = running_total_loss / num_steps 
    final_avg_supervised_acc = running_supervised_acc / num_steps
    
    
    train_metrics = {'acc_s': final_avg_supervised_acc,
                    'loss_s': final_avg_loss_s,
                    'loss_u': final_avg_loss_u,
                    'loss': final_avg_total_loss}
    
    if log_metric:
        for metric, value in train_metrics.items():
            logger.info("Train metric '{}' = {:.4f}".format(metric, value))
    return train_metrics

def train_normal(model: nn.Module, 
                dataset: Dataset,
                optimizer: torch.optim.Optimizer,
                num_epoch: int, 
                device: torch.device,
                ema: EMA = None,
                batch_size = 10,
                enable_progress_bar = False,
                log_metric = False) -> Dict[str, float]:
                
    model.train()
    optimizer.zero_grad()
    
    avg_total_loss = 0.0
    avg_acc = 0.0
    supervised_loss_fn = CrossEntropyLoss()
    # Calculate unsupervised loss    
    batch_size_s = batch_size
    num_steps = len(dataset) // batch_size_s

    for epoch in range(num_epoch):
        dataloader = DataLoader(dataset, batch_size = batch_size_s, shuffle = True, num_workers = os.cpu_count() // 2)
        if enable_progress_bar:
            progress_bar = get_progress_bar('Train', len(dataloader), epoch, num_epoch)
        else:
            progress_bar = None

        running_total_loss = 0.0
        running_acc = 0.0
        for step, (x_s, y_s) in enumerate(dataloader):
            x_s, y_s = x_s.to(device), y_s.to(device)
            y_s_one_hot =  nn.functional.one_hot(y_s.long(), num_classes=10).float() 
            
            # Calculate supervised loss
            y_pred_s = model(x_s)
            loss = supervised_loss_fn(y_pred_s, y_s_one_hot)
            acc = torch.sum(torch.argmax(y_pred_s, dim=1) == y_s).item() / len(y_s)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Update EMA
            if ema is not None:
                ema.update()
            
            # Log loss and metrics
            running_total_loss = running_total_loss + loss.item()
            running_acc = running_acc + acc
            
            avg_total_loss = running_total_loss / (step + 1)
            avg_acc = running_acc / (step + 1)
            if progress_bar is not None:
                progress_bar.set_postfix(loss = avg_total_loss,
                                        acc = avg_acc)
                progress_bar.update()
        if progress_bar is not None:
            progress_bar.close()
                
    final_avg_total_loss = running_total_loss / num_steps 
    final_avg_acc = running_acc / num_steps
    
    train_metrics = {'acc_s': final_avg_acc,
                    'loss': final_avg_total_loss}
    
    if log_metric:
        for metric, value in train_metrics.items():
            logger.info("Train metric '{}' = {:.4f}".format(metric, value))
    return train_metrics
    
def validation(model: nn.Module, 
               dataset: Dataset,
               device: torch.device = torch.device('cpu'),
               batch_size = 64,
               enable_progress_bar = False,
               log_metric = False) -> Dict[str, float]:
    model.eval()
    
    val_dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers = os.cpu_count() // 2)
    
    if enable_progress_bar:
        progress_bar = get_progress_bar('Validation', len(val_dataloader), 0, 1)
    else:
        progress_bar = None
    
    running_losses, running_accs = [], []
    for step, (x, y_gts) in enumerate(val_dataloader):
        x, y_gts = x.to(device), y_gts.to(device)
        y_preds = model(x)
        
        y_gts_one_hot =  nn.functional.one_hot(y_gts.long(), num_classes=10).float()
        val_loss = CrossEntropyLoss()(y_preds, y_gts_one_hot)
        acc = torch.sum(torch.argmax(y_preds, dim=1) == y_gts).item() / len(y_gts)
    
        running_losses.append(val_loss.item())
        running_accs.append(acc)
        
        if progress_bar is not None:
            avg_loss = sum(running_losses) / len(running_losses)
            avg_acc = sum(running_accs) / len(running_accs)
            progress_bar.set_postfix(loss = avg_loss, accuracy = avg_acc)
            progress_bar.update()
    if progress_bar is not None:
        progress_bar.close()
    val_metrics = {'loss': sum(running_losses) / len(running_losses),
                    'accuracy': sum(running_accs) / len(running_accs)}
    if log_metric:
        for metric, value in val_metrics.items():
            logger.info("Val metric '{}' = {:.4f}".format(metric, value))
        
    return val_metrics

def test(model: nn.Module, 
               dataset: Dataset,
               device: torch.device = torch.device('cpu'),
               batch_size = 64,
               enable_progress_bar = False,
               log_metric = False) -> Dict[str, float]:
    test_metrics = validation(model, dataset, device, batch_size, enable_progress_bar, log_metric)
    return test_metrics