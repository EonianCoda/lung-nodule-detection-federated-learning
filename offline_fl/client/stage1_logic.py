import os
import logging
from typing import Dict
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch

from fl_modules.inference.predictor import PredictorStage1 
from ..optimizer.losses import dice_log_loss
from fl_modules.utilities import get_progress_bar
from fl_modules.model.ema import EMA

logger = logging.getLogger(__name__)
PRINT_EVERY = 1000

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
    
    loss_weights = [2 / 3, 1 / 3]
    loss_fn = dice_log_loss()
    
    running_loss512, running_loss256, running_total_loss = 0.0, 0.0, 0.0
    
    for epoch in range(num_epoch):
        dataset.shuffle_data()
        train_dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers = os.cpu_count() // 2)
        total_num_steps = len(train_dataloader)
        
        if enable_progress_bar:
            progress_bar = get_progress_bar('Train', len(train_dataloader), epoch, num_epoch)
        else:
            progress_bar = None
        running_loss512, running_loss256, running_total_loss = 0.0, 0.0, 0.0
        for step, (inputs, target512, target256) in enumerate(train_dataloader):
            inputs, target512, target256 = inputs.to(device), target512.to(device), target256.to(device)
            outputs512, outputs256 = model(inputs)
            
            # Calculate loss            
            loss512 = loss_fn(target512, outputs512)
            loss256 = loss_fn(target256, outputs256)
            loss = loss512 * loss_weights[0] + loss256 * loss_weights[1]
            
            # Update weights
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Update EMA
            if ema is not None:
                ema.update()
            # Log loss
            running_loss512 = running_loss512 + loss512.item()
            running_loss256 = running_loss256 + loss256.item()
            running_total_loss = running_total_loss + loss.item()
            
            avg_loss512 = running_loss512 / (step + 1)
            avg_loss256 = running_loss256 / (step + 1)
            avg_total_loss = running_total_loss / (step + 1)
            
            if progress_bar is not None:
                progress_bar.set_postfix(loss = avg_total_loss,
                                        loss512 = avg_loss512,
                                        loss256 = avg_loss256)
                progress_bar.update()
        if progress_bar is not None:
            progress_bar.close()
                
    final_avg_loss512 = running_loss512 / total_num_steps
    final_avg_loss256 = running_loss256 / total_num_steps
    final_avg_total_loss = running_total_loss / total_num_steps
    
    train_metrics = {'loss512': final_avg_loss512,
                    'loss256': final_avg_loss256,
                    'loss': final_avg_total_loss}
    
    if log_metric:
        for metric, value in train_metrics.items():
            logger.info("Train metric '{}' = {:.4f}".format(metric, value))
    return train_metrics
    
def validation(model: nn.Module, 
               dataset: Dataset,
               iou_threshold: float = 0.01,
               device: torch.device = torch.device('cpu'),
               batch_size = 64,
               enable_progress_bar = False,
               log_metric = False) -> Dict[str, float]:
    model.eval()
    series_paths = dataset.series_paths
    gt_mask_maps_paths = dataset.gt_mask_maps_paths
    predictor = PredictorStage1(model, 
                                use_lobe = True, 
                                iou_threshold = iou_threshold, 
                                device = device, 
                                log_metrics = log_metric)
    val_metrics = predictor.get_recall_precision_of_nodules_in_series(series_paths, gt_mask_maps_paths)
    val_metrics = val_metrics['all']
    
    if log_metric:
        for metric, value in val_metrics.items():
            logger.info("Val metric '{}' = {:.4f}".format(metric, value))
        
    return val_metrics

def test(model: nn.Module, 
         dataset: Dataset,
         iou_threshold: float = 0.01,
         nodule_3d_minimum_thickness: int = 3,
         nodule_3d_minimum_size: int = 5,
         device: torch.device = torch.device('cpu'),
         batch_size = 64,
         enable_progress_bar = False,
         log_metric = False) -> Dict[str, float]:
    model.eval()
    series_paths = dataset.series_paths
    gt_mask_maps_paths = dataset.gt_mask_maps_paths
    predictor = PredictorStage1(model, 
                                use_lobe = True, 
                                iou_threshold = iou_threshold,
                                nodule_3d_minimum_thickness = nodule_3d_minimum_thickness,
                                nodule_3d_minimum_size = nodule_3d_minimum_size,
                                device = device, 
                                log_metrics = log_metric)
    test_metrics = predictor.get_recall_precision_of_nodules_in_series(series_paths, gt_mask_maps_paths)
    
    return test_metrics