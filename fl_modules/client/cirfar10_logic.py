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

LAMBDA_S = 10 # unsupervised learning
LAMBDA_I = 1e-2 # inter-client consistency
LAMBDA_A = 1e-2 # agreement-based pseudo labeling

def train_fixmatch(model: nn.Module, 
                    dataset_s: Dataset,
                    dataset_u: Dataset, 
                    optimizer: torch.optim.Optimizer,
                    num_epoch: int, 
                    device: torch.device,
                    unsupervised_conf_thrs: float = 0.95,
                    ema: EMA = None,
                    batch_size = 10,
                    enable_progress_bar = False,
                    log_metric = False) -> Dict[str, float]:
                
    model.train()
    optimizer.zero_grad()
    
    avg_total_loss, avg_loss_s, avg_loss_u = 0.0, 0.0, 0.0
    avg_supervised_acc = 0.0
    supervised_loss_fn = CrossEntropyLoss()
    # Calculate unsupervised loss    
    batch_size_s = batch_size
    num_steps = len(dataset_s) // batch_size_s
    if dataset_u is not None:
        batch_size_u = len(dataset_u) // num_steps

    for epoch in range(num_epoch):
        supervised_dataloader = DataLoader(dataset_s, batch_size = batch_size_s, shuffle = True, num_workers = os.cpu_count() // 2, drop_last=True)
        unsupervised_dataloader = DataLoader(dataset_u, batch_size = batch_size_u, shuffle = True, num_workers = os.cpu_count() // 2, drop_last=True)    
        if enable_progress_bar:
            progress_bar = get_progress_bar('Train', len(supervised_dataloader), epoch, num_epoch)
        else:
            progress_bar = None

        running_total_loss, running_loss_s, running_loss_u = 0.0, 0.0, 0.0
        running_supervised_acc = 0.0
        for step, ((x_s, y_s), (x_u)) in enumerate(zip(supervised_dataloader, unsupervised_dataloader)):
            x_s, y_s = x_s.to(device), y_s.to(device)
            y_s_one_hot =  nn.functional.one_hot(y_s.long(), num_classes=10).float() 
            for i in range(len(x_u)):
                x_u[i] = x_u[i].to(device)
            loss_final = 0
            
            # Calculate supervised loss
            y_pred_s = model(x_s)
            loss_s = supervised_loss_fn(y_pred_s, y_s_one_hot) * LAMBDA_S
            supervised_acc = torch.sum(torch.argmax(y_pred_s, dim=1) == y_s).item() / len(y_s)
            loss_final += loss_s
            
            # Calculate unsupervised loss
            y_pred = model(x_u[0])
            conf = torch.where(torch.max(y_pred, dim=1)[0] >= unsupervised_conf_thrs)[0]
            if len(conf) > 0:
                y_pseu = y_pred[conf]
                y_pseu = torch.argmax(y_pseu, dim=1)
                y_pseu_onehot = nn.functional.one_hot(y_pseu, num_classes=10).float()
                loss_u = supervised_loss_fn(model(x_u[1][conf]), y_pseu_onehot) * LAMBDA_A
                loss_final += loss_u
            else:
                loss_u = torch.tensor(0.0)
            loss_final.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Update EMA
            if ema is not None:
                ema.update()
            
            # Log loss and metrics
            running_loss_s = running_loss_s + loss_s.item()
            running_loss_u = running_loss_u + loss_u.item()
            running_total_loss = running_total_loss + loss_final.item()
            running_supervised_acc = running_supervised_acc + supervised_acc
            
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