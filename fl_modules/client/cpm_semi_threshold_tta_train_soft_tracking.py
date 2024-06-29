import os
import json
import math
import logging
import numpy as np
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .average_meter import AverageMeter
from fl_modules.utilities.utils import get_progress_bar
from .utils import get_memory_format
from fl_modules.dataset.transform.label import CoordToAnnot
from fl_modules.dataset.utils import compute_bbox3d_iou

logger = logging.getLogger(__name__)

TTA_BATCH_SIZE = 4

def unsupervised_train_one_step_wrapper(memory_format, loss_fn):
    def train_one_step(kwargs, model: nn.modules, sample: Dict[str, torch.Tensor], background_mask, soft_prob, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image = sample['image'].to(device, non_blocking=True, memory_format=memory_format) # z, y, x
        labels = sample['annot'].to(device, non_blocking=True) # z, y, x, d, h, w, type[-1, 0]
        # Compute loss
        outputs = model(image)
        cls_pos_loss, cls_neg_loss, cls_soft_loss, shape_loss, offset_loss, iou_loss = loss_fn(outputs, labels, background_mask, soft_prob, device = device)
        cls_pos_loss, cls_neg_loss, cls_soft_loss, shape_loss, offset_loss, iou_loss = cls_pos_loss.mean(), cls_neg_loss.mean(), cls_soft_loss.mean(), shape_loss.mean(), offset_loss.mean(), iou_loss.mean()
        
        lambda_cls= kwargs['lambda_pseu_cls']
        lambda_shape = kwargs['lambda_pseu_shape']
        lambda_offset = kwargs['lambda_pseu_offset']
        lambda_iou = kwargs['lambda_pseu_iou']
        
        loss = lambda_cls * (cls_pos_loss + cls_neg_loss) + lambda_shape * shape_loss + lambda_offset * offset_loss + lambda_iou * iou_loss
        # loss = lambda_cls * (cls_pos_loss + cls_neg_loss + cls_soft_loss) + lambda_shape * shape_loss + lambda_offset * offset_loss + lambda_iou * iou_loss
        return loss, cls_pos_loss, cls_neg_loss, cls_soft_loss, shape_loss, offset_loss, iou_loss, outputs
    return train_one_step

def train_one_step_wrapper(memory_format, loss_fn):
    def train_one_step(kwargs, model: nn.modules, sample: Dict[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image = sample['image'].to(device, non_blocking=True, memory_format=memory_format) # z, y, x
        labels = sample['annot'].to(device, non_blocking=True) # z, y, x, d, h, w, type[-1, 0]
        # Compute loss
        outputs = model(image)
        cls_pos_loss, cls_neg_loss, shape_loss, offset_loss, iou_loss = loss_fn(outputs, labels, device = device)
        cls_pos_loss, cls_neg_loss, shape_loss, offset_loss, iou_loss = cls_pos_loss.mean(), cls_neg_loss.mean(), shape_loss.mean(), offset_loss.mean(), iou_loss.mean()
        
        lambda_cls = kwargs['lambda_cls']
        lambda_shape = kwargs['lambda_shape']
        lambda_offset = kwargs['lambda_offset']
        lambda_iou = kwargs['lambda_iou']
        
        loss = lambda_cls * (cls_pos_loss + cls_neg_loss) + lambda_shape * shape_loss + lambda_offset * offset_loss + lambda_iou * iou_loss
        return loss, cls_pos_loss, cls_neg_loss, shape_loss, offset_loss, iou_loss, outputs
    return train_one_step

@torch.no_grad()
def sharpen_prob(cls_prob, t=0.7):
    cls_prob_s = cls_prob ** (1 / t)
    return (cls_prob_s / (cls_prob_s + (1 - cls_prob) ** (1 / t)))

def burn_in_train(args,
                model: nn.modules,
                detection_loss,
                optimizer: torch.optim.Optimizer,
                dataloader: DataLoader,
                device: torch.device,
                ema = None,) -> Dict[str, float]:
    model.train()
    avg_cls_pos_loss = AverageMeter()
    avg_cls_neg_loss = AverageMeter()
    avg_cls_loss = AverageMeter()
    avg_shape_loss = AverageMeter()
    avg_offset_loss = AverageMeter()
    avg_iou_loss = AverageMeter()
    avg_loss = AverageMeter()
    
    iters_to_accumulate = args.iters_to_accumulate
    # mixed precision training
    mixed_precision = args.mixed_precision
    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
        
    total_num_steps = len(dataloader)
    
    memory_format = get_memory_format(getattr(args, 'memory_format', None))
    if memory_format == torch.channels_last_3d:
        logger.info('Use memory format: channels_last_3d to train')
    train_one_step = train_one_step_wrapper(memory_format, detection_loss)
        
    optimizer.zero_grad()
    progress_bar = get_progress_bar('Train', (total_num_steps - 1) // iters_to_accumulate + 1)
    for iter_i, sample in enumerate(dataloader):
        if mixed_precision:
            with torch.cuda.amp.autocast():
                loss, cls_pos_loss, cls_neg_loss, shape_loss, offset_loss, iou_loss, outputs = train_one_step(args, model, sample, device)
            loss = loss / iters_to_accumulate
            scaler.scale(loss).backward()
        else:
            loss, cls_pos_loss, cls_neg_loss, shape_loss, offset_loss, iou_loss, outputs = train_one_step(args, model, sample, device)
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
            
            progress_bar.set_postfix(loss = avg_loss.avg,
                                    pos_cls = avg_cls_pos_loss.avg,
                                    neg_cls = avg_cls_neg_loss.avg,
                                    cls_loss = avg_cls_loss.avg,
                                    shape_loss = avg_shape_loss.avg,
                                    offset_loss = avg_offset_loss.avg,
                                    iou_loss = avg_iou_loss.avg)
            progress_bar.update()
    
    progress_bar.close()

    metrics = {'loss': avg_loss.avg,
                'cls_loss': avg_cls_loss.avg,
                'cls_pos_loss': avg_cls_pos_loss.avg,
                'cls_neg_loss': avg_cls_neg_loss.avg,
                'shape_loss': avg_shape_loss.avg,
                'offset_loss': avg_offset_loss.avg,
                'iou_loss': avg_iou_loss.avg}
    return metrics

def train(model_t: nn.modules,
          model_s: nn.modules,
          detection_loss,
          unsupervised_detection_loss,
          optimizer: torch.optim.Optimizer,
          dataloader_u: DataLoader,
          dataloader_l: DataLoader,
          detection_postprocess,
          device: torch.device,
          enable_progress_bar = False,
          log_metric = False,
          **kwargs) -> Dict[str, float]:
    mixed_precision = kwargs['mixed_precision']
    memory_format = kwargs['memory_format']
    iters_to_accumulate = kwargs['iters_to_accumulate']
    
    # Semi-supervised training args
    ema_buffer = kwargs.get('ema_buffer', False)
    sharpen_cls = kwargs.get('sharpen_cls', -1) 
    select_bg_crop = kwargs.get('select_bg_crop', 0)
    
    pseudo_update_ema_alpha = kwargs['pseudo_update_ema_alpha']
    pseudo_update_iou_threshold = kwargs['pseudo_update_iou_threshold']
    val_mixed_precision = kwargs['val_mixed_precision']
    pseudo_crop_threshold = kwargs['pseudo_crop_threshold']
    pseudo_nms_topk = kwargs['pseudo_nms_topk']
    pseudo_label_threshold = kwargs['pseudo_label_threshold']
    pseudo_background_threshold = kwargs['pseudo_background_threshold']

    semi_ema_alpha = kwargs['semi_ema_alpha']
    
    if ema_buffer:
        logger.info('Use EMA buffer')
    if sharpen_cls > 0:
        logger.info('Use sharpen cls = {:.3f}'.format(sharpen_cls))
    if select_bg_crop > 0:
        logger.info('Random select bg crop = 1 / {} fg crop'.format(select_bg_crop))
        
    # Initialize average meters
    avg_cls_loss = AverageMeter()
    avg_cls_pos_loss = AverageMeter()
    avg_cls_neg_loss = AverageMeter()
    avg_shape_loss = AverageMeter()
    avg_offset_loss = AverageMeter()
    avg_iou_loss = AverageMeter()
    avg_loss = AverageMeter()
    
    avg_pseu_cls_loss = AverageMeter()
    avg_pseu_cls_pos_loss = AverageMeter()
    avg_pseu_cls_neg_loss = AverageMeter()
    avg_pseu_cls_soft_loss = AverageMeter()
    avg_pseu_shape_loss = AverageMeter()
    avg_pseu_offset_loss = AverageMeter()
    avg_pseu_iou_loss = AverageMeter()
    avg_pseu_loss = AverageMeter()
    
    # For analysis
    avg_num_neg_patches_pseu = AverageMeter()
    avg_iou_pseu = AverageMeter()
    avg_tp_pseu = AverageMeter()
    avg_fp_pseu = AverageMeter()
    avg_fn_pseu = AverageMeter()
    avg_soft_target_pseu = AverageMeter()
    avg_fn_probs = AverageMeter()
    
    num_pseudo_nodules = 0
    # Mixed precision training
    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
    
    # Get memory format
    memory_format = get_memory_format(memory_format)
    if memory_format == torch.channels_last_3d:
        logger.info('Use memory format: channels_last_3d to train')
    
    # Training loop
    iter_l = iter(dataloader_l)
    
    train_one_step = train_one_step_wrapper(memory_format, detection_loss)
    unsupervised_train_one_step = unsupervised_train_one_step_wrapper(memory_format, unsupervised_detection_loss)
    
    coord_to_annot = CoordToAnnot()
    
    if not ema_buffer:
        model_t.train()
    else:
        model_t.eval()
    model_s.train()
    optimizer.zero_grad()
    num_iters = len(dataloader_u)
    
    if enable_progress_bar:
        progress_bar = get_progress_bar('Train', num_iters)
    
    for iter_i, sample_u in enumerate(dataloader_u):
        ### Unlabeled data
        weak_u_sample = sample_u['weak']
        strong_u_sample = sample_u['strong']
        
        # TTA generate pseudo label
        weak_transform_weights = weak_u_sample['transform_weights'] # (N, num_aug)
        weak_transform_weights = torch.from_numpy(weak_transform_weights).to(device, non_blocking=True)
        weak_feat_transforms = weak_u_sample['feat_transform'] # (N, num_aug)
        weak_annots = weak_u_sample['annot'] # list of list of dict, first list is scan, second is n_samples
        num_aug = weak_u_sample['image'].size(1)
        
        weak_images = weak_u_sample['image']
        weak_lobes = weak_u_sample['lobe'].to(device, non_blocking=True, memory_format=memory_format)
        weak_series_names = weak_u_sample['series_name']
        cls_prob = []
        outputs_t = []
        for i in range(int(math.ceil(weak_images.size(0) / TTA_BATCH_SIZE))):
            end = (i + 1) * TTA_BATCH_SIZE
            if end > weak_images.size(0):
                end = weak_images.size(0)
            input = weak_images[i * TTA_BATCH_SIZE:end] # (bs, num_aug, 1, crop_z, crop_y, crop_x)
            
            with torch.no_grad():
                if val_mixed_precision:
                    with torch.cuda.amp.autocast():
                        input = input.view(-1, 1, *input.size()[3:]).to(device, non_blocking=True, memory_format=memory_format) # (bs * num_aug, 1, crop_z, crop_y, crop_x)
                        outputs_t_b = model_t(input)
                else:
                    input = input.view(-1, 1, *input.size()[3:]).to(device, non_blocking=True, memory_format=memory_format) # (bs * num_aug, 1, crop_z, crop_y, crop_x)
                    outputs_t_b = model_t(input)

            # Ensemble the augmentations
            Cls_output = outputs_t_b['Cls'] # (bs * num_aug, 1, 24, 24, 24)
            Shape_output = outputs_t_b['Shape'] # (bs * num_aug, 3, 24, 24, 24)
            Offset_output = outputs_t_b['Offset'] # (bs * num_aug, 3, 24, 24, 24)
            
            _, _, d, h, w = Cls_output.size()
            Cls_output = Cls_output.view(-1, num_aug, 1, d, h, w)
            Shape_output = Shape_output.view(-1, num_aug, 3, d, h, w)
            Offset_output = Offset_output.view(-1, num_aug, 3, d, h, w)
            
            feat_transforms = weak_feat_transforms[i * TTA_BATCH_SIZE:end] # (bs, num_aug)
            for b_i in range(len(feat_transforms)):
                for aug_i in range(num_aug):
                    if len(feat_transforms[b_i][aug_i]) > 0:
                        for trans in reversed(feat_transforms[b_i][aug_i]):
                            Cls_output[b_i, aug_i, ...] = trans.backward(Cls_output[b_i, aug_i, ...])
                            Shape_output[b_i, aug_i, ...] = trans.backward(Shape_output[b_i, aug_i, ...])
            transform_weight = weak_transform_weights[i * TTA_BATCH_SIZE:end] # (bs, num_aug)
            transform_weight = transform_weight.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5) # (bs, num_aug, 1, 1, 1, 1)
            Cls_output = (Cls_output * transform_weight).sum(1) # (bs, 1, 24, 24, 24)
            Cls_output = Cls_output.sigmoid()
            
            Shape_output = (Shape_output * transform_weight).sum(1) # (bs, 3, 24, 24, 24)
            Offset_output = Offset_output[:, 0, ...] # (bs, 3, 24, 24, 24)
            lobe = weak_lobes[i * TTA_BATCH_SIZE:end]
            # if sharpen_cls > 0:
            #     assert sharpen_cls < 1
            #     Cls_output = sharpen_prob(Cls_output, t=sharpen_cls)
            outputs_t_b = {'Cls': Cls_output, 'Shape': Shape_output, 'Offset': Offset_output}
            
            outputs_t_b = detection_postprocess(outputs_t_b, device=device, is_logits=False, lobe_mask = lobe, threshold = pseudo_crop_threshold, nms_topk=pseudo_nms_topk) #1, prob, ctr_z, ctr_y, ctr_x, d, h, w
            outputs_t.append(outputs_t_b.data.cpu().numpy())
            cls_prob.append(Cls_output)
            del input, Shape_output, Offset_output, outputs_t_b
        del weak_lobes
        
        # Add label
        # Remove the padding, -1 means invalid
        outputs_t = np.concatenate(outputs_t, axis=0) # (bs, topk, 8)
        
        bs = outputs_t.shape[0]
        # Calculate and transform background mask
        cls_prob = torch.cat(cls_prob, dim=0) # shape: (bs, 1, d, h, w)
        strong_feat_transforms = strong_u_sample['feat_transform'] # shape = (bs,)
        for batch_i in range(bs):
            for transform in strong_feat_transforms[batch_i]:
                cls_prob[batch_i] = transform.forward(cls_prob[batch_i])
                
        # EMA update pseudo label
        sample_annots = []
        crop_bb_mins = []
        history_ctrs = []
        history_rads = []
        history_probs = []
        weak_spacings = weak_u_sample['spacing'] # shape = (bs,)
        for annot in weak_u_sample['annot']:
            for a in annot:
                crop_bb_mins.append(a['crop_bb_min'])
                history_ctrs.append(a['ctr'])
                history_rads.append(a['rad'])
                history_probs.append(a['prob'])
        
        d, h, w = weak_images.shape[-3:]
        crop_bboxes = np.array([[[0, 0, 0], [d, h, w]]], dtype=np.float32)
        for batch_i in range(bs):
            # Get current pseudo label
            outputs_t_b = outputs_t[batch_i]
            valid_mask = (outputs_t_b[:, -1] != -1.0)
            outputs_t_b = outputs_t_b[valid_mask]
            
            new_probs_b = outputs_t_b[:, 1]
            new_ctrs_b = outputs_t_b[:, 2:5]
            new_rads_b = outputs_t_b[:, 5:8]
            
            # Get history pseudo label
            history_ctrs_b = history_ctrs[batch_i]
            history_rads_b = history_rads[batch_i]
            
            if len(outputs_t_b) == 0 and len(history_ctrs_b) == 0:
                continue
            elif len(outputs_t_b) == 0: # No any pseudo label in this batch
                history_bboxes = np.stack([history_ctrs_b - history_rads_b / 2, history_ctrs_b + history_rads_b / 2], axis=1)
                history_valid_ious = compute_bbox3d_iou(history_bboxes, crop_bboxes)
                history_valid_mask = (history_valid_ious.max(axis=1) >= 0.3)
                if np.count_nonzero(history_valid_mask) != 0:
                    history_probs[batch_i][history_valid_mask] *= pseudo_update_ema_alpha
                continue
            elif len(history_ctrs_b) == 0: # No any pseudo label in history
                history_ctrs[batch_i] = new_ctrs_b
                history_rads[batch_i] = new_rads_b
                history_probs[batch_i] = new_probs_b * pseudo_update_ema_alpha
                continue
                
            # Compute iou between history pseudo label and new pseudo label
            new_bboxes = np.stack([new_ctrs_b - new_rads_b / 2, new_ctrs_b + new_rads_b / 2], axis=1)
            history_bboxes = np.stack([history_ctrs_b - history_rads_b / 2, history_ctrs_b + history_rads_b / 2], axis=1)
            
            history_valid_ious = compute_bbox3d_iou(history_bboxes, crop_bboxes)
            history_valid_mask = (history_valid_ious.max(axis=1) >= 0.5)
            
            ious = compute_bbox3d_iou(history_bboxes, new_bboxes)
            # According to the iou, update the pseudo label in dataset
            history_matched_ious = ious.max(axis=1)
            matched_indices = ious.argmax(axis=1)
            for i, (matched_iou, matched_idx) in enumerate(zip(history_matched_ious, matched_indices)):
                if matched_iou > pseudo_update_iou_threshold:
                    # Update pseudo label
                    history_probs[batch_i][i] = history_probs[batch_i][i] * pseudo_update_ema_alpha + outputs_t_b[matched_idx, 1] * (1 - pseudo_update_ema_alpha)
                    history_ctrs[batch_i][i] = history_ctrs[batch_i][i] * pseudo_update_ema_alpha + outputs_t_b[matched_idx, 2:5] * (1 - pseudo_update_ema_alpha)
                    history_rads[batch_i][i] = history_rads[batch_i][i] * pseudo_update_ema_alpha + outputs_t_b[matched_idx, 5:8] * (1 - pseudo_update_ema_alpha)
                elif history_valid_mask[i] == True: # penalize the pseudo label because of the low iou
                    history_probs[batch_i][i] *= pseudo_update_ema_alpha
            
            # Add new pseudo label
            new_matched_ious = ious.max(axis=0)
            new_probs_b = []
            new_ctrs_b = []
            new_rads_b = []
            for i, matched_iou in enumerate(new_matched_ious):        
                if matched_iou < pseudo_update_iou_threshold:
                    new_probs_b.append(outputs_t_b[i, 1])
                    new_ctrs_b.append(outputs_t_b[i, 2:5])
                    new_rads_b.append(outputs_t_b[i, 5:8])
            
            if len(new_probs_b) > 0:
                history_ctrs[batch_i] = np.concatenate([history_ctrs[batch_i], new_ctrs_b], axis=0)
                history_rads[batch_i] = np.concatenate([history_rads[batch_i], new_rads_b], axis=0)
                # history_probs[batch_i] = np.concatenate([history_probs[batch_i], new_probs_b], axis=0) * args.pseudo_update_ema_alpha
                new_probs_b = np.array(new_probs_b, dtype=np.float32) * 0.9 # penalize the new pseudo label
                history_probs[batch_i] = np.concatenate([history_probs[batch_i], new_probs_b], axis=0)
        
        # Generate pseudo label
        strong_ctr_transforms = strong_u_sample['ctr_transform'] # shape = (bs,)
        weak_spacings = weak_u_sample['spacing'] # shape = (bs,)
        transformed_annots = []

        for batch_i in range(bs):
            # According prob to generate pseudo label
            probs_b = history_probs[batch_i]
            valid_mask = (probs_b > pseudo_label_threshold)
            
            if np.count_nonzero(valid_mask) == 0:
                transformed_annots.append(np.zeros((0, 10), dtype='float32'))
                continue
            ctrs_b = history_ctrs[batch_i][valid_mask].copy()
            rads_b = history_rads[batch_i][valid_mask].copy()
            spacing = weak_spacings[batch_i]
            
            # Transform pseudo label
            for transform in strong_ctr_transforms[batch_i]:
                ctrs_b = transform.forward_ctr(ctrs_b)
                rads_b = transform.forward_rad(rads_b)
                spacing = transform.forward_spacing(spacing)
            
            sample_dict = {'ctr': ctrs_b, 
                            'rad': rads_b, 
                            'cls': np.zeros((len(ctrs_b), ), dtype=np.int32),
                            'spacing': spacing}
            sample_annots.append(sample_dict)
            
            # Transform to annot
            transformed_annots.append(coord_to_annot(sample_dict)['annot'])
            
        # Update pseudo label of dataset
        for batch_i in range(bs):
            series_name = weak_series_names[batch_i]
            crop_bb_min = crop_bb_mins[batch_i]
            history_ctrs_b = history_ctrs[batch_i]
            history_rads_b = history_rads[batch_i]
            history_probs_b = history_probs[batch_i]
            
            if len(history_ctrs_b) != 0:
                history_ctrs_b = history_ctrs_b + crop_bb_min
                dataloader_u.dataset.update_pseudo_label(series_name, history_ctrs_b, history_rads_b, history_probs_b)
            
        # Pad the pseudo label
        valid_mask = np.array([len(annot) > 0 for annot in transformed_annots], dtype=np.int32)
        valid_mask = (valid_mask == 1)
        
        avg_num_neg_patches_pseu.update(np.count_nonzero(valid_mask == 0))
        
        max_num_annots = max(annot.shape[0] for annot in transformed_annots)
        if max_num_annots > 0:
            transformed_annots_padded = np.ones((len(transformed_annots), max_num_annots, 10), dtype='float32') * -1
            for idx, annot in enumerate(transformed_annots):
                if annot.shape[0] > 0:
                    transformed_annots_padded[idx, :annot.shape[0], :] = annot
        else:
            transformed_annots_padded = np.ones((len(transformed_annots), 1, 10), dtype='float32') * -1

        ## For analysis
        # Compute iou between pseudo label and original label
        all_iou_pseu = []
        tp, fp, fn = 0, 0, 0
        all_fn_probs = []
        for i, (annot, pseudo_annot, is_valid) in enumerate(zip(strong_u_sample['gt_annot'].numpy(), transformed_annots_padded, valid_mask)):
            annot = annot[annot[:, -1] != -1] # (ctr_z, ctr_y, ctr_x, d, h, w, space_z, space_y, space_x)
            if not is_valid:
                fn += len(annot)
                for a in annot:
                    ctr_z, ctr_y, ctr_x = a[:3]
                    ctr_z = min(max(int(ctr_z // 4), 0), cls_prob.shape[2] - 1)
                    ctr_y = min(max(int(ctr_y // 4), 0), cls_prob.shape[3] - 1)
                    ctr_x = min(max(int(ctr_x // 4), 0), cls_prob.shape[4] - 1)
                    ctr_prob = cls_prob[i, 0, ctr_z, ctr_y, ctr_x].item()
                    all_fn_probs.append(ctr_prob)
                continue 
            
            pseudo_annot = pseudo_annot[pseudo_annot[:, -1] != -1]
            if len(annot) == 0:
                fp += len(pseudo_annot)
                # Cheating, set FP to 0
                # transformed_annots_padded[i, ...] = -1
                continue
            elif len(pseudo_annot) == 0:
                fn += len(annot)
                for a in annot:
                    ctr_z, ctr_y, ctr_x = a[:3]
                    ctr_z = min(max(int(ctr_z // 4), 0), cls_prob.shape[2] - 1)
                    ctr_y = min(max(int(ctr_y // 4), 0), cls_prob.shape[3] - 1)
                    ctr_x = min(max(int(ctr_x // 4), 0), cls_prob.shape[4] - 1)
                    ctr_prob = cls_prob[i, 0, ctr_z, ctr_y, ctr_x].item()
                    all_fn_probs.append(ctr_prob)
                continue
            
            bboxes = np.stack([annot[:, :3] - annot[:, 3:6] / 2, annot[:, :3] + annot[:, 3:6] / 2], axis=1)
            pseudo_bboxes = np.stack([pseudo_annot[:, :3] - pseudo_annot[:, 3:6] / 2, pseudo_annot[:, :3] + pseudo_annot[:, 3:6] / 2], axis=1)
            ious = compute_bbox3d_iou(pseudo_bboxes, bboxes)
            
            iou_pseu = ious.max(axis=1)
            iou = ious.max(axis=0)
            
            all_iou_pseu.extend(iou_pseu[iou_pseu > 1e-3].tolist())
            tp += np.count_nonzero(iou > 1e-3)
            fn += np.count_nonzero(iou <= 1e-3)
            fp += np.count_nonzero(iou_pseu < 1e-3)
        
            for a in annot[iou <= 1e-3]:
                ctr_z, ctr_y, ctr_x = a[:3]
                ctr_z = min(max(int(ctr_z // 4), 0), cls_prob.shape[2] - 1)
                ctr_y = min(max(int(ctr_y // 4), 0), cls_prob.shape[3] - 1)
                ctr_x = min(max(int(ctr_x // 4), 0), cls_prob.shape[4] - 1)
                ctr_prob = cls_prob[i, 0, ctr_z, ctr_y, ctr_x].item()
                all_fn_probs.append(ctr_prob)
                
            # Cheating, set TP to real coordinate
            # matched_indices = ious.argmax(axis=1)
            # for j in np.where(iou_pseu > 1e-3)[0]:
            #     transformed_annots_padded[i, j, ...] = annot[matched_indices[j]].copy()
                
            # Cheating, set FP to 0
            # for j in np.where(iou_pseu < 1e-3)[0]:
            #     transformed_annots_padded[i, j, ...] = -1
            
        if len(all_iou_pseu) > 0:
            avg_iou_pseu.update(np.mean(all_iou_pseu))
        avg_tp_pseu.update(tp)
        avg_fp_pseu.update(fp)
        avg_fn_pseu.update(fn)
        if len(all_fn_probs) > 0:
            avg_fn_probs.update(np.mean(all_fn_probs), len(all_fn_probs))
        
        num_fg_crop = np.count_nonzero(valid_mask)
        # Random select some bg crop
        if num_fg_crop > 0 and select_bg_crop > 0:
            num_bg_crop = min(max(1, int(num_fg_crop / select_bg_crop)), len(valid_mask) - num_fg_crop)
            selected_bg_idx = np.random.choice(np.where(valid_mask == 0)[0], num_bg_crop, replace=False)
            valid_mask[selected_bg_idx] = 1
        
            transformed_annots_padded = transformed_annots_padded[valid_mask]
            strong_u_sample['image'] = strong_u_sample['image'][valid_mask]
            cls_prob = cls_prob[valid_mask]
            strong_u_sample['gt_annot'] = strong_u_sample['gt_annot'][valid_mask]
        strong_u_sample['annot'] = torch.from_numpy(transformed_annots_padded)
        # Cheating, set pseudo label to real label
        # strong_u_sample['annot'] = strong_u_sample['gt_annot']
        avg_soft_target_pseu.update(torch.sum(torch.logical_and(cls_prob > pseudo_background_threshold, cls_prob < pseudo_label_threshold)).item())
        background_mask = (cls_prob < pseudo_background_threshold) # shape: (bs, 1, d, h, w)
        background_mask = background_mask.view(background_mask.shape[0], -1) # shape: (bs, num_points)
        
        if mixed_precision:
            with torch.cuda.amp.autocast():
                loss_pseu, cls_pos_pseu_loss, cls_neg_pseu_loss, cls_soft_pseu_loss, shape_pseu_loss, offset_pseu_loss, iou_pseu_loss, outputs_pseu = unsupervised_train_one_step(kwargs, model_s, strong_u_sample, background_mask, cls_prob, device)
        else:
            loss_pseu, cls_pos_pseu_loss, cls_neg_pseu_loss, cls_soft_pseu_loss, shape_pseu_loss, offset_pseu_loss, iou_pseu_loss, outputs_pseu = unsupervised_train_one_step(kwargs, model_s, strong_u_sample, background_mask, cls_prob, device)
        
        avg_pseu_cls_loss.update(cls_pos_pseu_loss.item() + cls_neg_pseu_loss.item() + cls_soft_pseu_loss.item())
        avg_pseu_cls_pos_loss.update(cls_pos_pseu_loss.item())
        avg_pseu_cls_neg_loss.update(cls_neg_pseu_loss.item())
        avg_pseu_cls_soft_loss.update(cls_soft_pseu_loss.item())
        avg_pseu_shape_loss.update(shape_pseu_loss.item())
        avg_pseu_offset_loss.update(offset_pseu_loss.item())
        avg_pseu_iou_loss.update(iou_pseu_loss.item())
        avg_pseu_loss.update(loss_pseu.item())
        num_pseudo_nodules += len(strong_u_sample['annot'][strong_u_sample['annot'][..., -1] != -1])
        
        del outputs_pseu, background_mask, cls_prob
        ### Labeled data
        try:
            labeled_sample = next(iter_l)
        except StopIteration:
            iter_l = iter(dataloader_l)
            labeled_sample = next(iter_l)
        
        if mixed_precision:
            with torch.cuda.amp.autocast():
                loss, cls_pos_loss, cls_neg_loss, shape_loss, offset_loss, iou_loss, outputs = train_one_step(kwargs, model_s, labeled_sample, device)
        else:
            loss, cls_pos_loss, cls_neg_loss, shape_loss, offset_loss, iou_loss, outputs = train_one_step(kwargs, model_s, labeled_sample, device)
        
        avg_cls_loss.update(cls_pos_loss.item() + cls_neg_loss.item())
        avg_cls_pos_loss.update(cls_pos_loss.item())
        avg_cls_neg_loss.update(cls_neg_loss.item())
        avg_shape_loss.update(shape_loss.item())
        avg_offset_loss.update(offset_loss.item())
        avg_iou_loss.update(iou_loss.item())
        avg_loss.update(loss.item())
        
        # Update model
        total_loss = loss + loss_pseu * kwargs['lambda_pseu']
        # Compute gradient
        total_loss = total_loss / iters_to_accumulate
        if mixed_precision:
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        
        if (iter_i + 1) % iters_to_accumulate == 0 or iter_i == num_iters - 1:
            if mixed_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if enable_progress_bar:
                progress_bar.set_postfix(cls_pos_l = avg_cls_pos_loss.avg,
                                        cls_neg_l = avg_cls_neg_loss.avg,
                                        cls_pos_u = avg_pseu_cls_pos_loss.avg,
                                        cls_neg_u = avg_pseu_cls_neg_loss.avg,
                                        cls_soft_u = avg_pseu_cls_soft_loss.avg,
                                        avg_iou_pseu = avg_iou_pseu.avg,
                                        num_u = num_pseudo_nodules,
                                        num_neg = avg_num_neg_patches_pseu.sum,
                                        tp = avg_tp_pseu.sum,
                                        fp = avg_fp_pseu.sum,
                                        fn = avg_fn_pseu.sum,
                                        fn_p = avg_fn_probs.avg)
                progress_bar.update()
            
            with torch.no_grad():
                # Update teacher model by exponential moving average
                for param, teacher_param in zip(model_s.parameters(), model_t.parameters()):
                    if param.requires_grad:
                        teacher_param.data.mul_(semi_ema_alpha).add_(param.data, alpha = 1 - semi_ema_alpha)
                        
                if ema_buffer:
                    for (name_s, buffer_s), (name_t, buffer_t) in zip(model_s.named_buffers(), model_t.named_buffers()):
                        if 'num_batches_tracked' in name_s:
                            continue
                        buffer_t.data.mul_(semi_ema_alpha).add_(buffer_s.data, alpha = 1 - semi_ema_alpha)
        del labeled_sample, outputs, loss, loss_pseu
        torch.cuda.empty_cache()
    
    if enable_progress_bar:
        progress_bar.close()    
    
    recall = avg_tp_pseu.sum / max(avg_tp_pseu.sum + avg_fn_pseu.sum, 1e-3)
    precision = avg_tp_pseu.sum / max(avg_tp_pseu.sum + avg_fp_pseu.sum, 1e-3)
    f1 = 2 * recall * precision / max(recall + precision, 1e-3)
    metrics = {'loss': avg_loss.avg,
                'cls_loss': avg_cls_loss.avg,
                'cls_pos_loss': avg_cls_pos_loss.avg,
                'cls_neg_loss': avg_cls_neg_loss.avg,
                'shape_loss': avg_shape_loss.avg,
                'offset_loss': avg_offset_loss.avg,
                'iou_loss': avg_iou_loss.avg,
                'loss_pseu': avg_pseu_loss.avg,
                'cls_loss_pseu': avg_pseu_cls_loss.avg,
                'cls_pos_loss_pseu': avg_pseu_cls_pos_loss.avg,
                'cls_neg_loss_pseu': avg_pseu_cls_neg_loss.avg,
                'cls_soft_loss_pseu': avg_pseu_cls_soft_loss.avg,
                'shape_loss_pseu': avg_pseu_shape_loss.avg,
                'offset_loss_pseu': avg_pseu_offset_loss.avg,
                'avg_iou_pseu': avg_iou_pseu.avg,
                'avg_soft_target_pseu': avg_soft_target_pseu.sum,
                'iou_loss_pseu': avg_pseu_iou_loss.avg,
                'num_pseudo_nodules':  num_pseudo_nodules,
                'num_neg_patches_pseu': avg_num_neg_patches_pseu.sum,
                'pseu_recall': recall,
                'pseu_precision': precision,
                'pseudo_f1': f1,
                'pseu_tp': avg_tp_pseu.sum,
                'pseu_fp': avg_fp_pseu.sum,
                'pseu_fn': avg_fn_pseu.sum,
                'pseu_fn_probs': avg_fn_probs.avg}
    
    if log_metric:
        for metric, value in metrics.items():
            logger.info("Train metric '{}' = {:.4f}".format(metric, value))
    
    return metrics