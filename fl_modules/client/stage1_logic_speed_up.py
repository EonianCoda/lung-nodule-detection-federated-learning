import os
import logging
from typing import Dict, List

from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch

from fl_modules.inference.predictor import PredictorStage1 
from fl_modules.utilities import get_progress_bar
from fl_modules.model.ema import EMA
from .average_meter import AverageMeter
from ..optimizer.losses import dice_log_loss

logger = logging.getLogger(__name__)
PRINT_EVERY = 1000

def train_one_step(images: Dict[str, torch.Tensor], 
                   model: nn.Module, 
                   device: torch.device,
                   loss_fn: nn.Module) -> List[torch.Tensor]:
        
        for key in images.keys():
            images[key] = images[key].to(device, non_blocking = True)
        
        image, target512, target256 = images['image'], images['target512'], images['target256']
        outputs512, outputs256 = model(image)
        loss512 = loss_fn(target512, outputs512)
        loss256 = loss_fn(target256, outputs256)
        return loss512, loss256
    
def train(model: nn.Module, 
          dataloader: DataLoader,
          optimizer: torch.optim.Optimizer,
          device: torch.device,
          ema: EMA = None,
          enable_progress_bar = False,
          mixed_precision = True,
          log_metric = False,
          **kwargs) -> Dict[str, float]:
    
    model.train()
    optimizer.zero_grad()
    
    loss_weights = [2 / 3, 1 / 3]
    loss_fn = dice_log_loss()
    
    # Initialize scaler for mixed precision training
    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
    
    progress_bar = get_progress_bar('Train', len(dataloader)) if enable_progress_bar else None
    running_loss512, running_loss256, running_total_loss = AverageMeter(), AverageMeter(), AverageMeter()
    dataloader.dataset.shuffle()
    for step, images in enumerate(dataloader):
        # Update weights
        if mixed_precision:
            with torch.cuda.amp.autocast():
                loss512, loss256 = train_one_step(images, model, device, loss_fn)
                total_loss = loss_weights[0] * loss512 + loss_weights[1] * loss256
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss512, loss256 = train_one_step(images, model, device, loss_fn)
            total_loss = loss_weights[0] * loss512 + loss_weights[1] * loss256
            total_loss.backward()
            optimizer.step()
            
        optimizer.zero_grad()
        
        # Update EMA
        if ema is not None:
            ema.update()
        
        # Log loss
        n = images['image'].size(0)
        running_loss512.update(loss512.item(), n)
        running_loss256.update(loss256.item(), n)
        running_total_loss.update(total_loss.item(), n)
        
        # Print loss
        if progress_bar is not None:
            progress_bar.set_postfix(loss = running_total_loss.avg,
                                    loss512 = running_loss512.avg,
                                    loss256 = running_loss256.avg)
            progress_bar.update()
            
    if progress_bar is not None:
        progress_bar.close()
                
    train_metrics = {'loss': running_total_loss.avg,
                    'loss512': running_loss512.avg,
                    'loss256': running_loss256.avg}
    
    if log_metric:
        for metric, value in train_metrics.items():
            logger.info("Train metric '{}' = {:.4f}".format(metric, value))
    return train_metrics
    
def validation(model: nn.Module, 
               dataloader: DataLoader,
               iou_threshold: float = 0.01,
               device: torch.device = torch.device('cpu'),
               mixed_precision = False,
               log_metric = False,
               **kwargs) -> Dict[str, float]:
    model.eval()
    dataset = dataloader.dataset
    series_paths = dataset.series_paths
    gt_mask_maps_paths = dataset.gt_mask_maps_paths
    predictor = PredictorStage1(model, 
                                use_lobe = True, 
                                iou_threshold = iou_threshold, 
                                mixed_precision = mixed_precision,
                                device = device, 
                                log_metrics = log_metric)
    val_metrics = predictor.get_recall_precision_of_nodules_in_series(series_paths, gt_mask_maps_paths)
    val_metrics = val_metrics['all']
    
    if log_metric:
        for metric, value in val_metrics.items():
            logger.info("Val metric '{}' = {:.4f}".format(metric, value))
        
    return val_metrics

def test(model: nn.Module, 
        dataloader: DataLoader,
         iou_threshold: float = 0.01,
         nodule_3d_minimum_thickness: int = 3,
         nodule_3d_minimum_size: int = 5,
         device: torch.device = torch.device('cpu'),
         mixed_precision = False,
         log_metric = False,
         **kwargs) -> Dict[str, float]:
    model.eval()
    dataset = dataloader.dataset
    series_paths = dataset.series_paths
    gt_mask_maps_paths = dataset.gt_mask_maps_paths
    predictor = PredictorStage1(model, 
                                use_lobe = True, 
                                iou_threshold = iou_threshold,
                                nodule_3d_minimum_thickness = nodule_3d_minimum_thickness,
                                nodule_3d_minimum_size = nodule_3d_minimum_size,
                                device = device, 
                                mixed_precision = mixed_precision,
                                log_metrics = log_metric)
    test_metrics = predictor.get_recall_precision_of_nodules_in_series(series_paths, gt_mask_maps_paths)
    
    return test_metrics