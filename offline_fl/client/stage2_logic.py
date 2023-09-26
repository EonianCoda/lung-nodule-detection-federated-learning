import os
import numpy as np
import logging
from typing import Dict, Tuple

from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch

from fl_modules.utilities import get_progress_bar
from fl_modules.model.ema import EMA

from ..optimizer.losses import focal_loss
logger = logging.getLogger(__name__)

class Nodule2dMetrics:
    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def update(self, preds, gt, thresholds):
        preds = (preds >= thresholds).astype(np.int32)
        self.tp += np.count_nonzero((preds == 1) & (gt == 1))
        self.fp += np.count_nonzero((preds == 1) & (gt == 0))
        self.tn += np.count_nonzero((preds == 0) & (gt == 0))
        self.fn += np.count_nonzero((preds == 0) & (gt == 1))
    
    @property
    def recall(self) -> float:
        return self.tp / max(self.tp + self.fn, 1)
    @property
    def precision(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)
    
    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn)

    @property
    def f1_score(self) -> float:
        recall = self.recall
        precision = self.precision

        if recall + precision <= 0:
            return 0
        
        return 2 * ((recall * precision) / (recall + precision))

    @property
    def f2_score(self) -> float:
        recall = self.recall
        precision = self.precision

        if recall + precision <= 0:
            return 0
        
        return 5 * ((recall * precision) / (4 * recall + precision))

    def get_metrics(self) -> Dict[str, float]:
        return {'acc': self.accuracy,
                'recall': self.recall,
                'precision': self.precision,
                'f1': self.f1_score,
                'f2': self.f2_score}

class Nodule3dMetrics:
    def __init__(self, dataset) -> None:
        self.preds = []
        self.indices = []
        self.dataset = dataset

    def update(self, preds, indices):
        preds = np.squeeze(preds).tolist()
        indices = np.squeeze(indices).tolist()
        
        if not isinstance(preds, list):
            preds = [preds]
        if not isinstance(indices, list):
            indices = [indices]
        
        self.preds.extend(preds)
        self.indices.extend(indices)

    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        return self.dataset.compute_nodule_3d_metrics(self.indices, self.preds)
    
class NoduleMetrics:
    def __init__(self, dataset):
        self.dataset = dataset
        self.metrics_2d = Nodule2dMetrics()
        self.metrics_3d = Nodule3dMetrics(dataset)

    def update(self, preds: torch.Tensor, labels: torch.Tensor, thresholds: torch.Tensor, indices: torch.Tensor):
        preds = preds.cpu().detach().numpy()
        labels = labels.cpu().numpy()
        thresholds = thresholds.numpy()
        indices = indices.numpy()
        
        self.metrics_2d.update(preds, labels, thresholds)
        self.metrics_3d.update(preds, indices)

    def get_metrics_2d(self) -> Dict[str, float]:
        return self.metrics_2d.get_metrics()
    
    def get_metrics_overall(self) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        Returns: Tuple[dict, dict]
            A tuple of (metric_2d, metrics_3d_of_different_type), metric_2d is a dict of serveral ('metric_name', metric_value)
            pairs, metrics_3d_of_different_type is a dict of serveral ('nodule_type', dict of ('metric_name', metric_value))
        """
        metrics_2d = dict()
        for key, value in self.get_metrics_2d().items():
            metrics_2d[f'{key}_2d'] = value

        metrics_3d_of_different_type = self.metrics_3d.get_metrics()
        return metrics_2d, metrics_3d_of_different_type

def train(model: nn.Module, 
          dataset: Dataset, 
          optimizer: torch.optim.Optimizer,
          num_epoch: int, 
          device: torch.device,
          ema: EMA = None,
          batch_size = 1,
          enable_progress_bar = False,
          log_metric = False) -> Dict[str, float]:
    
    model.train()
    optimizer.zero_grad()
    loss_history = []

    alpha = dataset.get_alpha()
    logger.info('Alpha = {:.4f}'.format(alpha))
    loss_fn = focal_loss(alpha = alpha, gamma = 2.0)
    
    train_dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = os.cpu_count() // 2)
    total_num_steps = len(train_dataloader)
    for epoch in range(num_epoch):
        if enable_progress_bar:
            progress_bar = get_progress_bar('Train', len(train_dataloader), epoch, num_epoch)
        else:
            progress_bar = None
            
        nodule_metrics = NoduleMetrics(dataset)
        running_loss = 0.0
        for step, (patchs, labels, thresholds, indices) in enumerate(train_dataloader):
            patchs, labels = patchs.to(device), labels.to(device)
            preds = model(patchs)
            
            # Calculate loss
            loss = loss_fn(labels, preds)
            
            # Update weights
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Update EMA
            if ema is not None:
                ema.update()
            
            nodule_metrics.update(preds, labels, thresholds, indices)
            metrics_2d = nodule_metrics.get_metrics_2d()
            # Log loss and metrics
            loss_history.append(loss.item())
            running_loss = running_loss + loss.item()
            
            avg_loss = running_loss / (step + 1)
            if progress_bar is not None:
                progress_bar.set_postfix(loss = avg_loss, **metrics_2d)
                progress_bar.update()
        if progress_bar is not None:
            progress_bar.close()
            
    final_avg_loss = running_loss / total_num_steps
    train_metrics = {'loss': final_avg_loss}
    
    metrics_2d, metrics_3d_of_different_type = nodule_metrics.get_metrics_overall()
    train_metrics.update(metrics_2d)
    train_metrics.update(metrics_3d_of_different_type['all'])
    
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
    nodule_metrics = NoduleMetrics(dataset)
    for step, (patchs, labels, thresholds, indices) in enumerate(val_dataloader):
        patchs, labels = patchs.to(device), labels.to(device)
        preds = model(patchs)
        
        nodule_metrics.update(preds, labels, thresholds, indices)
        metrics_2d = nodule_metrics.get_metrics_2d()
        
        if progress_bar is not None:
            progress_bar.set_postfix(**metrics_2d)
            progress_bar.update()
            
    if progress_bar is not None:
        progress_bar.close()
        
        
    metrics_2d, metrics_3d_of_different_type = nodule_metrics.get_metrics_overall()
    val_metrics = dict()
    val_metrics.update(metrics_2d)
    val_metrics.update(metrics_3d_of_different_type['all'])
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
    model.eval()
    
    val_dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers = os.cpu_count() // 2)
    
    if enable_progress_bar:
        progress_bar = get_progress_bar('Validation', len(val_dataloader), 0, 1)
    else:
        progress_bar = None
        
    nodule_metrics = NoduleMetrics(dataset)
    for step, (patchs, labels, thresholds, indices) in enumerate(val_dataloader):
        patchs, labels = patchs.to(device), labels.to(device)
        preds = model(patchs)
        
        nodule_metrics.update(preds, labels, thresholds, indices)
        metrics_2d = nodule_metrics.get_metrics_2d()
        
        if progress_bar is not None:
            progress_bar.set_postfix(**metrics_2d)
            progress_bar.update()
            
    if progress_bar is not None:
        progress_bar.close()
        
        
    metrics_2d, metrics_3d_of_different_type = nodule_metrics.get_metrics_overall()
    val_metrics = metrics_3d_of_different_type
    return val_metrics