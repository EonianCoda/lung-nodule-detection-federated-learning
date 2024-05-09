import os
import logging
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fl_modules.utilities.utils import get_progress_bar
from .average_meter import AverageMeter
from .utils import get_memory_format

logger = logging.getLogger(__name__)

def train_one_step_wrapper(memory_format,  lambda_cls: float, lambda_shape: float, lambda_offset: float, lambda_iou: float):
    def train_one_step(model: nn.modules, sample: Dict[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image = sample['image'].to(device, non_blocking=True, memory_format=memory_format) # z, y, x
        labels = sample['annot'].to(device, non_blocking=True) # z, y, x, d, h, w, type[-1, 0]
        # Compute loss
        cls_pos_loss, cls_neg_loss, shape_loss, offset_loss, iou_loss = model([image, labels])
        cls_pos_loss, cls_neg_loss, shape_loss, offset_loss, iou_loss = cls_pos_loss.mean(), cls_neg_loss.mean(), shape_loss.mean(), offset_loss.mean(), iou_loss.mean()
        loss = lambda_cls * (cls_pos_loss + cls_neg_loss) + lambda_shape * shape_loss + lambda_offset * offset_loss + lambda_iou * iou_loss
        del image, labels
        return loss, cls_pos_loss, cls_neg_loss, shape_loss, offset_loss, iou_loss
    return train_one_step

def train(mixed_precision: bool,
          memory_format: str,
          iters_to_accumulate: int,
          lambda_cls: float,
          lambda_shape: float,
          lambda_offset: float,
          lambda_iou: float,
          model: nn.modules,
          optimizer: torch.optim.Optimizer,
          dataloader: DataLoader,
          device: torch.device,
          ema = None,
          enable_progress_bar = False,
          log_metric = False,
          **kwargs) -> Dict[str, float]:
    
    # print(device)
    model.to(device)
    model.train()
    avg_cls_pos_loss = AverageMeter()
    avg_cls_neg_loss = AverageMeter()
    avg_cls_loss = AverageMeter()
    avg_shape_loss = AverageMeter()
    avg_offset_loss = AverageMeter()
    avg_iou_loss = AverageMeter()
    avg_loss = AverageMeter()
    
    # mixed precision training
    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
        
    total_num_steps = len(dataloader)
    
    memory_format = get_memory_format(memory_format)
    if memory_format == torch.channels_last_3d:
        logger.info('Use memory format: channels_last_3d to train')
    train_one_step = train_one_step_wrapper(memory_format, lambda_cls, lambda_shape, lambda_offset, lambda_iou)
        
    optimizer.zero_grad()
    
    if enable_progress_bar:
        progress_bar = get_progress_bar('Train', (total_num_steps - 1) // iters_to_accumulate + 1)
    for iter_i, sample in enumerate(dataloader):
        if mixed_precision:
            with torch.cuda.amp.autocast():
                loss, cls_pos_loss, cls_neg_loss, shape_loss, offset_loss, iou_loss = train_one_step(model, sample, device)
            loss = loss / iters_to_accumulate
            scaler.scale(loss).backward()
        else:
            loss, cls_pos_loss, cls_neg_loss, shape_loss, offset_loss, iou_loss = train_one_step(model, sample, device)
            loss = loss / iters_to_accumulate
            loss.backward()
        
        # Update history
        avg_cls_pos_loss.update(cls_pos_loss.item())
        avg_cls_neg_loss.update(cls_neg_loss.item())
        avg_cls_loss.update(cls_pos_loss.item() + cls_neg_loss.item())
        avg_shape_loss.update(shape_loss.item())
        avg_offset_loss.update(offset_loss.item())
        avg_iou_loss.update(iou_loss.item())
        avg_loss.update(loss.item() * iters_to_accumulate)
        
        # Update model
        if (iter_i + 1) % iters_to_accumulate == 0 or iter_i == total_num_steps - 1:
            if mixed_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            # Update EMA
            if ema is not None:
                ema.update()
            
            if enable_progress_bar:    
                progress_bar.set_postfix(loss = avg_loss.avg,
                                        pos_cls = avg_cls_pos_loss.avg,
                                        neg_cls = avg_cls_neg_loss.avg,
                                        cls_loss = avg_cls_loss.avg,
                                        shape_loss = avg_shape_loss.avg,
                                        offset_loss = avg_offset_loss.avg,
                                        iou_loss = avg_iou_loss.avg)
                progress_bar.update()
    if enable_progress_bar:
        progress_bar.close()

    metrics = {'loss': avg_loss.avg,
                'cls_loss': avg_cls_loss.avg,
                'cls_pos_loss': avg_cls_pos_loss.avg,
                'cls_neg_loss': avg_cls_neg_loss.avg,
                'shape_loss': avg_shape_loss.avg,
                'offset_loss': avg_offset_loss.avg,
                'iou_loss': avg_iou_loss.avg}
    if log_metric:
        for metric, value in metrics.items():
            logger.info("Train metric '{}' = {:.4f}".format(metric, value))
    return metrics