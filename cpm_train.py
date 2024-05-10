# %% -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import torch
import os
import logging
import numpy as np
from typing import Tuple, Any
### data ###
from fl_modules.dataset.utils import get_image_padding_value
from fl_modules.dataset.collate import train_collate_fn, infer_collate_fn
from fl_modules.dataset.split_combine import SplitComb
from torch.utils.data import DataLoader
import fl_modules.dataset.transform as transform
import torchvision
from torch.utils.tensorboard import SummaryWriter
### logic ###
from fl_modules.client.cpm_train import train
from fl_modules.client.cpm_val import val
from fl_modules.client.utils import write_metrics, save_states, load_states, get_memory_format
### optimzer ###
from fl_modules.optimizer.adamW import AdamW
from fl_modules.optimizer.scheduler import GradualWarmupScheduler
from fl_modules.optimizer.ema import EMA
### postprocessing ###
from fl_modules.utilities.logs import setup_logging
from fl_modules.utilities import init_seed, get_local_time_str_in_taiwan, write_yaml, load_yaml, build_class
from early_stopping_save import EarlyStoppingSave
from config import SAVE_ROOT, DEFAULT_OVERLAP_RATIO, IMAGE_SPACING, NODULE_TYPE_DIAMETERS

logger = logging.getLogger(__name__)

early_stopping = None

def get_args():
    parser = argparse.ArgumentParser()
    # Training settings
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--mixed_precision', action='store_true', default=False, help='use mixed precision in training')
    parser.add_argument('--val_mixed_precision', action='store_true', default=False, help='use mixed precision in validation')
    parser.add_argument('--batch_size', type=int, default=5, help='input batch size for training (default: 6)')
    parser.add_argument('--val_batch_size', type=int, default=2, help='input batch size for validation (default: 2)')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train (default: 3000)')
    parser.add_argument('--crop_size', nargs='+', type=int, default=[96, 96, 96], help='crop size')
    parser.add_argument('--overlap_ratio', type=float, default=DEFAULT_OVERLAP_RATIO, help='overlap ratio')
    parser.add_argument('--early_end_epoch', type=int, default=-1, help='end epoch')
    # Resume
    parser.add_argument('--resume_folder', type=str, default='', help='resume folder')
    parser.add_argument('--pretrained_model_path', type=str, default='')
    # Data
    parser.add_argument('--data_set_class', type=str, default='fl_modules.dataset.cpm_dataset', help='data set class')
    parser.add_argument('--train_set', type=str, help='train_list')
    parser.add_argument('--val_set', type=str, help='val_list')
    parser.add_argument('--test_set', type=str, help='test_list')
    parser.add_argument('--min_d', type=int, default=0, help="min depth of nodule, if some nodule's depth < min_d, it will be` ignored")
    parser.add_argument('--min_size', type=int, default=27, help="min size of nodule, if some nodule's size < min_size, it will be ignored")
    parser.add_argument('--post_proces_min_size', type=int, default=27, help="min size of nodule, if some nodule's size < min_size, it will be ignored")
    
    parser.add_argument('--data_norm_method', type=str, default='none', help='normalize method, mean_std or scale or none')
    parser.add_argument('--memory_format', type=str, default='channels_first', help='memory format of model, channels_first or channels_last, channels_last is faster on linux') # channels_last is faster on linux
    parser.add_argument('--crop_partial', action='store_true', default=False, help='crop partial nodule')
    parser.add_argument('--crop_tp_iou', type=float, default=0.3, help='iou threshold for crop tp use if crop_partial is True')
    parser.add_argument('--use_bg', action='store_true', default=False, help='use background(healthy lung) in training')
    parser.add_argument('--pad_water', action='store_true', default=False, help='pad water or not')
    # Data Augmentation
    parser.add_argument('--tp_ratio', type=float, default=0.6, help='positive ratio in instance crop')
    parser.add_argument('--rand_rot', nargs='+', type=int, default=[30, 0, 0], help='random rotate')
    parser.add_argument('--use_rand_spacing', action='store_true', default=False, help='use random spacing')
    parser.add_argument('--rand_spacing', nargs='+', type=float, default=[0.9, 1.1], help='random spacing range, [min, max]')
    # Learning rate
    parser.add_argument('--lr', type=float, default=2e-3, help='the learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='warmup epochs')
    parser.add_argument('--warmup_gamma', type=float, default=0.01, help='warmup gamma')
    parser.add_argument('--decay_cycle', type=int, default=1, help='decay cycle, 1 means no cycle')
    parser.add_argument('--decay_gamma', type=float, default=0.1, help='decay gamma')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='the weight decay')
    parser.add_argument('--start_val_epoch', type=int, default=150, help='start to validate from this epoch')
    # EMA
    parser.add_argument('--not_apply_ema', action='store_true', default=False, help='apply ema')
    parser.add_argument('--ema_momentum', type=float, default=0.998, help='ema decay')
    parser.add_argument('--ema_warmup_epochs', type=int, default=-1, help='warmup epochs for ema')
    # Loss hyper-parameters
    parser.add_argument('--loss_class', type=str, default='fl_modules.model.cpm_net.loss', help='loss class')
    
    parser.add_argument('--pos_target_topk', type=int, default=7, help='topk grids assigned as positives')
    parser.add_argument('--pos_ignore_ratio', type=int, default=5)
    parser.add_argument('--num_samples', type=int, default=8, help='number of samples for each instance')
    parser.add_argument('--iters_to_accumulate', type=int, default=2, help='number of batches to wait before updating the weights')
    parser.add_argument('--cls_num_neg', type=int, default=-1, help='number of negatives (-1 means all)')
    parser.add_argument('--cls_neg_pos_ratio', type=int, default=100, help='ratio of negatives to positives in positive samples')
    parser.add_argument('--cls_num_hard', type=int, default=200, help='number of hard negatives in negtative samples(-1 means all)')
    parser.add_argument('--cls_fn_weight', type=float, default=4.0, help='weights of cls_fn')
    parser.add_argument('--cls_fn_threshold', type=float, default=0.8, help='threshold of cls_fn')
    
    parser.add_argument('--cls_focal_alpha', type=float, default=0.75, help='alpha of focal loss')
    parser.add_argument('--cls_focal_gamma', type=float, default=2.0, help='gamma of focal loss')
    
    parser.add_argument('--cls_hard_fp_thrs1', type=float, default=0.5, help='threshold of cls_hard_fp1')
    parser.add_argument('--cls_hard_fp_thrs2', type=float, default=0.7, help='threshold of cls_hard_fp2')
    parser.add_argument('--cls_hard_fp_w1', type=float, default=1.5, help='weights of cls_hard_fp1')
    parser.add_argument('--cls_hard_fp_w2', type=float, default=2.0, help='weights of cls_hard_fp2')
    
    parser.add_argument('--lambda_cls', type=float, default=4.0, help='weights of cls loss')
    parser.add_argument('--lambda_offset', type=float, default=1.0,help='weights of offset')
    parser.add_argument('--lambda_shape', type=float, default=1.0, help='weights of reg')
    parser.add_argument('--lambda_iou', type=float, default=4.0, help='weights of iou loss')
    # Val hyper-parameters
    parser.add_argument('--det_post_process_class', type=str, default='fl_modules.model.cpm_net.detection_post_process')
    parser.add_argument('--det_topk', type=int, default=60, help='topk detections')
    parser.add_argument('--det_nms_threshold', type=float, default=0.05, help='detection nms threshold')
    parser.add_argument('--det_nms_topk', type=int, default=20, help='detection nms topk')
    parser.add_argument('--val_iou_threshold', type=float, default=0.2, help='iou threshold for validation')
    parser.add_argument('--val_fixed_prob_threshold', type=float, default=0.65, help='fixed probability threshold for validation')
    parser.add_argument('--val_det_threshold', type=float, default=0.4, help='detection threshold')
    parser.add_argument('--froc_det_thresholds', nargs='+', type=float, default=[0.4, 0.5, 0.7], help='froc det thresholds')
    # Val technical settings
    parser.add_argument('--apply_lobe', action='store_true', default=False, help='apply lobe or not')
    # Test hyper-parameters
    parser.add_argument('--test_iou_threshold', type=float, default=0.1, help='iou threshold for test')
    parser.add_argument('--test_det_threshold', type=float, default=0.2, help='detection threshold for test')
    parser.add_argument('--test_froc_det_thresholds', nargs='+', type=float, default=[0.2, 0.5, 0.7], help='froc det thresholds')
    # Model
    parser.add_argument('--model_class', type=str, default='fl_modules.model.cpm_net.cpm_net', help='model class')
    parser.add_argument('--norm_type', type=str, default='batchnorm', help='norm type of backbone')
    parser.add_argument('--head_norm', type=str, default='batchnorm', help='norm type of head')
    parser.add_argument('--act_type', type=str, default='ReLU', help='act type of network')
    parser.add_argument('--first_stride', nargs='+', type=int, default=[2, 2, 2], help='stride of the first layer')
    parser.add_argument('--n_blocks', nargs='+', type=int, default=[2, 3, 3, 3], help='number of blocks in each layer')
    parser.add_argument('--n_filters', nargs='+', type=int, default=[64, 96, 128, 160], help='number of filters in each layer')
    parser.add_argument('--stem_filters', type=int, default=32, help='number of filters in stem layer')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
    parser.add_argument('--no_se', action='store_true', default=False, help='not use se block')
    parser.add_argument('--aspp', action='store_true', default=False, help='use aspp')
    parser.add_argument('--dw_type', default='conv', help='downsample type, conv or maxpool')
    parser.add_argument('--up_type', default='deconv', help='upsample type, deconv or interpolate')
    parser.add_argument('--out_stride', type=int, default=4, help='output stride')
    # other
    parser.add_argument('--nodule_size_mode', type=str, default='seg_size', help='nodule size mode, seg_size or dhw')
    parser.add_argument('--max_workers', type=int, default=4, help='max number of workers, num_workers = min(batch_size, max_workers)')
    parser.add_argument('--best_metrics', nargs='+', type=str, default=['froc_2_recall', 'froc_mean_recall'], help='metric for validation')
    parser.add_argument('--val_interval', type=int, default=1, help='validate interval')
    parser.add_argument('--exp_name', type=str, default='', metavar='str', help='experiment name')
    parser.add_argument('--save_model_interval', type=int, default=10, help='how many epochs to wait before saving model')
    args = parser.parse_args()
    if args.val_det_threshold != args.froc_det_thresholds[0]:
        raise ValueError(f'val_det_threshold = {args.val_det_threshold} should be equal to froc_det_thresholds[0] = {args.froc_det_thresholds[0]}')
    return args

def freeze_model(model):
    for name, param in model.named_parameters():
        if 'in_conv' in name or 'in_dw' in name or 'block1' in name and 'norm' in name:
            param.requires_grad = False
    return model

def unfreeze_model(model):
    for name, param in model.named_parameters():
        param.requires_grad = True
    return model

def add_weight_decay(net, weight_decay):
    """no weight decay on bias and normalization layer
    """
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue  # skip frozen weights
        # skip bias and bn layer
        if ".norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{"params": no_decay, "weight_decay": 0.0},
            {"params": decay, "weight_decay": weight_decay}]

def prepare_training(args, device, num_training_steps) -> Tuple[int, Any, AdamW, GradualWarmupScheduler, Any]:
    logger.info('Build model "{}"'.format(args.model_class))
    CpmNet = build_class('{}.CpmNet'.format(args.model_class))
    
    logger.info('Build loss "{}"'.format(args.loss_class))
    DetectionLoss = build_class('{}.DetectionLoss'.format(args.loss_class))
    
    logger.info('Build post process "{}"'.format(args.det_post_process_class))
    DetectionPostprocess = build_class('{}.DetectionPostprocess'.format(args.det_post_process_class))               
    # build model
    detection_loss = DetectionLoss(crop_size = args.crop_size,
                                   pos_target_topk = args.pos_target_topk, 
                                   pos_ignore_ratio = args.pos_ignore_ratio,
                                   cls_num_neg=args.cls_num_neg,
                                   cls_num_hard=args.cls_num_hard,
                                   cls_fn_weight = args.cls_fn_weight,
                                   cls_fn_threshold = args.cls_fn_threshold,
                                   cls_neg_pos_ratio = args.cls_neg_pos_ratio,
                                   cls_hard_fp_thrs1 = args.cls_hard_fp_thrs1,
                                   cls_hard_fp_thrs2 = args.cls_hard_fp_thrs2,
                                   cls_hard_fp_w1 = args.cls_hard_fp_w1,
                                   cls_hard_fp_w2 = args.cls_hard_fp_w2,
                                   cls_focal_alpha = args.cls_focal_alpha,
                                   cls_focal_gamma = args.cls_focal_gamma)
                                        
    model = CpmNet(norm_type = args.norm_type,
                     head_norm = args.head_norm, 
                     act_type = args.act_type, 
                     se = not args.no_se, 
                     aspp = args.aspp,
                     first_stride=args.first_stride,
                     n_blocks=args.n_blocks,
                     n_filters=args.n_filters,
                     stem_filters=args.stem_filters,
                     dropout=args.dropout,
                     dw_type = args.dw_type,
                     up_type = args.up_type,
                     detection_loss = detection_loss,
                     out_stride = args.out_stride,
                     device = device)
    
    val_detection_postprocess = DetectionPostprocess(topk = args.det_topk, 
                                                    threshold = args.val_det_threshold, 
                                                    nms_threshold = args.det_nms_threshold,
                                                    nms_topk = args.det_nms_topk,
                                                    crop_size = args.crop_size,
                                                    min_size = args.post_proces_min_size)
    
    test_detection_postprocess = DetectionPostprocess(topk = args.det_topk,
                                                    threshold = args.test_det_threshold,
                                                    nms_threshold = args.det_nms_threshold,
                                                    nms_topk = args.det_nms_topk,
                                                    crop_size = args.crop_size,
                                                    min_size = args.post_proces_min_size)

    start_epoch = 0
    model = model.to(device=device, memory_format=get_memory_format(getattr(args, 'memory_format', 'channels_first')))
        
    # build optimizer and scheduler
    params = add_weight_decay(model, args.weight_decay)
    optimizer = AdamW(params=params, lr=args.lr, weight_decay=args.weight_decay)
    
    T_max = args.epochs // args.decay_cycle
    scheduler_reduce = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=args.lr * args.decay_gamma)
    scheduler_warm = GradualWarmupScheduler(optimizer, gamma=args.warmup_gamma, warmup_epochs=args.warmup_epochs, after_scheduler=scheduler_reduce)
    logger.info('Warmup learning rate from {:.1e} to {:.1e} for {} epochs and then reduce learning rate from {:.1e} to {:.1e} by cosine annealing with {} cycles'.format(args.lr * args.warmup_gamma, args.lr, args.warmup_epochs, args.lr, args.lr * args.decay_gamma, args.decay_cycle))

    # Build EMA
    if args.not_apply_ema:
        ema = None
    else:
        if args.ema_warmup_epochs == -1:
            ema_warmup_steps = int((args.start_val_epoch + 20) * num_training_steps)
        else:
            ema_warmup_steps = int(args.ema_warmup_epochs * num_training_steps) if args.ema_warmup_epochs > 0 else 0
        logger.info('Apply EMA with decay: {:.4f}, warmup steps: {}'.format(args.ema_momentum, ema_warmup_steps))
        ema = EMA(model, momentum = args.ema_momentum, warmup_steps = ema_warmup_steps)

    if args.resume_folder != '':
        logger.info('Resume experiment "{}"'.format(os.path.dirname(args.resume_folder)))
        model_folder = os.path.join(args.resume_folder, 'model')
        
        # Get the latest model
        model_names = os.listdir(model_folder)
        model_epochs = [int(name.split('.')[0].split('_')[-1]) for name in model_names]
        start_epoch = model_epochs[np.argmax(model_epochs)]
        ckpt_path = os.path.join(model_folder, f'epoch_{start_epoch}.pth')
        logger.info('Load checkpoint from "{}"'.format(ckpt_path))

        load_states(ckpt_path, device, model, optimizer, scheduler_warm, ema)
    
        # Resume best metric
        global early_stopping
        early_stopping = EarlyStoppingSave.load(save_dir=os.path.join(args.resume_folder, 'best'), target_metrics=args.best_metrics, model=model)
        start_epoch = start_epoch + 1
    elif args.pretrained_model_path != '':
        logger.info('Load model from "{}"'.format(args.pretrained_model_path))
        load_states(args.pretrained_model_path, device, model)

    if args.resume_folder == '' and ema is not None:
        ema.register()
    
    return start_epoch, model, optimizer, scheduler_warm, ema, val_detection_postprocess, test_detection_postprocess

def build_train_augmentation(args, crop_size: Tuple[int, int, int], pad_value: float, blank_side: int):
    rot_zy = (crop_size[0] == crop_size[1] == crop_size[2])
    rot_zx = (crop_size[0] == crop_size[1] == crop_size[2])
        
    transform_list_train = [transform.RandomFlip(p=0.5, flip_depth=True, flip_height=True, flip_width=True)]
    transform_list_train.append(transform.RandomRotate90(p=0.5, rot_xy=True, rot_xz=rot_zx, rot_yz=rot_zy))
    transform_list_train.append(transform.CoordToAnnot())
                            
    logger.info('Augmentation: random flip: True, random roation90: {}'.format([True, rot_zy, rot_zx]))
    train_transform = torchvision.transforms.Compose(transform_list_train)
    return train_transform

def get_train_dataloder(args, blank_side=0) -> DataLoader:
    crop_size = args.crop_size
    overlap_size = (np.array(crop_size) * args.overlap_ratio).astype(np.int32).tolist()
    rand_trans = [int(s * 2/3) for s in overlap_size]
    pad_value = get_image_padding_value(args.data_norm_method, use_water=args.pad_water)
    
    logger.info('Crop size: {}, overlap size: {}, rand_trans: {}, pad value: {}, tp_ratio: {:.3f}'.format(crop_size, overlap_size, rand_trans, pad_value, args.tp_ratio))
    
    from fl_modules.dataset.crop import InstanceCrop
    crop_fn_train = InstanceCrop(crop_size=crop_size, overlap_ratio=args.overlap_ratio, tp_ratio=args.tp_ratio, rand_trans=rand_trans, rand_rot=args.rand_rot,
                                sample_num=args.num_samples, blank_side=blank_side, instance_crop=True)
    train_transform = build_train_augmentation(args, crop_size, pad_value, blank_side)
    mmap_mode = None
    logger.info('Use itk rotate {}'.format(args.rand_rot))
    
    train_set_class = build_class('{}.TrainDataset'.format(args.data_set_class))
    train_dataset = train_set_class(series_list_path = args.train_set, crop_fn = crop_fn_train, image_spacing=IMAGE_SPACING, transform_post = train_transform, 
                                 min_d=args.min_d, min_size = args.min_size, use_bg = args.use_bg, norm_method=args.data_norm_method, mmap_mode=mmap_mode)
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size, 
                              shuffle=True,
                              collate_fn=train_collate_fn,
                              num_workers=min(args.batch_size, args.max_workers),
                              pin_memory=True,
                              drop_last=True, 
                              persistent_workers=True)
    logger.info("There are {} training samples and {} batches in '{}'".format(len(train_loader.dataset), len(train_loader), args.train_set))
    return train_loader

def get_val_test_dataloder(args) -> Tuple[DataLoader, DataLoader]:
    crop_size = args.crop_size
    overlap_size = (np.array(crop_size) * args.overlap_ratio).astype(np.int32).tolist()
    pad_value = get_image_padding_value(args.data_norm_method, use_water=False) # do not pad water when validating
    
    split_comber = SplitComb(crop_size=crop_size, overlap_size=overlap_size, pad_value=pad_value, do_padding=False)
    
    det_set_class = build_class('{}.DetDataset'.format(args.data_set_class))
    val_dataset = det_set_class(series_list_path = args.val_set, SplitComb=split_comber, image_spacing=IMAGE_SPACING, norm_method=args.data_norm_method, apply_lobe=args.apply_lobe, out_stride=args.out_stride)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=min(args.val_batch_size, args.max_workers), pin_memory=True, drop_last=False, collate_fn=infer_collate_fn)

    test_dataset = det_set_class(series_list_path = args.test_set, SplitComb=split_comber, image_spacing=IMAGE_SPACING, norm_method=args.data_norm_method, apply_lobe=args.apply_lobe, out_stride=args.out_stride)
    test_loader = DataLoader(test_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=min(args.val_batch_size, args.max_workers), pin_memory=True, drop_last=False, collate_fn=infer_collate_fn)
    
    logger.info("There are {} validation samples and {} batches in '{}'".format(len(val_loader.dataset), len(val_loader), args.val_set))
    logger.info("There are {} test samples and {} batches in '{}'".format(len(test_loader.dataset), len(test_loader), args.test_set))
    return val_loader, test_loader

def get_train_infer_dataloader(args) -> DataLoader:
    crop_size = args.crop_size
    overlap_size = (np.array(crop_size) * args.overlap_ratio).astype(np.int32).tolist()
    pad_value = get_image_padding_value(args.data_norm_method)
    
    split_comber = SplitComb(crop_size=crop_size, overlap_size=overlap_size, pad_value=pad_value, do_padding=False)
    
    det_set_class = build_class('{}.DetDataset'.format(args.data_set_class))
    train_infer_dataset = det_set_class(series_list_path = args.train_set, SplitComb=split_comber, image_spacing=IMAGE_SPACING, norm_method=args.data_norm_method, apply_lobe=args.apply_lobe, out_stride=args.out_stride)
    train_infer_loader = DataLoader(train_infer_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=min(args.val_batch_size, args.max_workers), pin_memory=True, drop_last=False, collate_fn=infer_collate_fn)
    return train_infer_loader

if __name__ == '__main__':
    args = get_args()
    
    if args.resume_folder != '': # resume training
        exp_folder = args.resume_folder
        setting_yaml_path = os.path.join(exp_folder, 'setting.yaml')
        setting = load_yaml(setting_yaml_path)
        for key, value in setting.items():
            if key != 'resume_folder':
                setattr(args, key, value)
    else:     
        timestamp = get_local_time_str_in_taiwan()
        exp_folder = os.path.join(SAVE_ROOT, f'{timestamp}_{args.exp_name}')
    setup_logging(level='info', log_file=os.path.join(exp_folder, 'log.txt'))
    init_seed(args.seed)
    write_yaml(os.path.join(exp_folder, 'setting.yaml'), vars(args))
    
    if args.pad_water:
        logger.warning('Pad water in training but not in validation and test')
    
    # Prepare training
    model_save_dir = os.path.join(exp_folder, 'model')
    writer = SummaryWriter(log_dir = os.path.join(exp_folder, 'tensorboard'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = get_train_dataloder(args)
    start_epoch, model, optimizer, scheduler_warm, ema, val_det_postprocess, test_det_postprocess = prepare_training(args, device, len(train_loader) // args.iters_to_accumulate)
    val_loader, test_loader = get_val_test_dataloder(args)
    
    if early_stopping is None:
        early_stopping = EarlyStoppingSave(target_metrics=args.best_metrics, save_dir=os.path.join(exp_folder, 'best'), model=model)
        
    if args.early_end_epoch > 0:
        end_epoch = args.early_end_epoch
    else:
        end_epoch = args.epochs
        
    for epoch in range(start_epoch, end_epoch + 1):
        train_metrics = train(mixed_precision=args.mixed_precision,
                            memory_format=args.memory_format,
                            iters_to_accumulate=args.iters_to_accumulate,
                            lambda_cls=args.lambda_cls,
                            lambda_iou=args.lambda_iou,
                            lambda_offset=args.lambda_offset,
                            lambda_shape=args.lambda_shape,
                            enable_progress_bar=True,
                            log_metric=False,
                            model = model,
                            optimizer = optimizer,
                            dataloader = train_loader,
                            ema = ema, 
                            device = device)
        scheduler_warm.step()
        write_metrics(train_metrics, epoch, 'train', writer)
        for key, value in train_metrics.items():
            logger.info('==> Epoch: {} {}: {:.4f}'.format(epoch, key, value))
        # add learning rate to tensorboard
        logger.info('==> Epoch: {} lr: {:.6f}'.format(epoch, scheduler_warm.get_lr()[0]))
        write_metrics({'lr': scheduler_warm.get_lr()[0]}, epoch, 'train', writer)
        
        # Remove the checkpoint of epoch % save_model_interval != 0
        for i in range(epoch):
            ckpt_path = os.path.join(model_save_dir, 'epoch_{}.pth'.format(i))
            if ((i % args.save_model_interval != 0 or i == 0 or i < args.start_val_epoch) and os.path.exists(ckpt_path)):
                os.remove(ckpt_path)
        save_states(os.path.join(model_save_dir, f'epoch_{epoch}.pth'), model, optimizer, scheduler_warm, ema)
        
        if (epoch >= args.start_val_epoch and epoch % args.val_interval == 0) or epoch == args.epochs:
            # Use Shadow model to validate and save model
            if ema is not None:
                ema.apply_shadow()
            val_metrics = val(mixed_precision=args.val_mixed_precision,
                            memory_format=args.memory_format,
                            patch_label_type='none',
                            froc_det_thresholds=args.froc_det_thresholds,
                            iou_threshold=args.val_iou_threshold,
                            model = model,
                            detection_postprocess=val_det_postprocess,
                            dataloader = val_loader, 
                            device = device,
                            image_spacing = IMAGE_SPACING,
                            series_list_path=args.val_set,
                            exp_folder=os.path.join(exp_folder, 'val'),
                            epoch = epoch,
                            nodule_type_diameters=NODULE_TYPE_DIAMETERS,
                            min_d=args.min_d,
                            apply_lobe=args.apply_lobe,
                            min_size=args.min_size,
                            enable_progress_bar=True,
                            nodule_size_mode=args.nodule_size_mode)
            
            early_stopping.step(val_metrics, epoch)
            write_metrics(val_metrics, epoch, 'val', writer)

            # Restore model
            if ema is not None:
                ema.restore()
    # Test
    logger.info('Test the best model')
    test_save_dir = os.path.join(exp_folder, 'test')
    os.makedirs(test_save_dir, exist_ok=True)
    for (target_metric, model_path), best_epoch in zip(early_stopping.get_best_model_paths().items(), early_stopping.best_epoch):
        logger.info('Load best model from "{}"'.format(model_path))
        load_states(model_path, device, model)
        test_metrics = val(mixed_precision=args.val_mixed_precision,
                            memory_format=args.memory_format,
                            patch_label_type='none',
                            froc_det_thresholds=args.test_froc_det_thresholds,
                            iou_threshold=args.test_iou_threshold,
                            model = model,
                            detection_postprocess=test_det_postprocess,
                            dataloader = val_loader, 
                            device = device,
                            image_spacing = IMAGE_SPACING,
                            series_list_path=args.test_set,
                            exp_folder=exp_folder,
                            apply_lobe=args.apply_lobe,
                            epoch = 'test_best_{}'.format(target_metric),
                            nodule_type_diameters=NODULE_TYPE_DIAMETERS,
                            min_d=args.min_d,
                            min_size=args.min_size,
                            enable_progress_bar=True,
                            nodule_size_mode=args.nodule_size_mode)
        
        write_metrics(test_metrics, epoch, 'test/best_{}'.format(target_metric), writer)
        with open(os.path.join(test_save_dir, 'test_best_{}.txt'.format(target_metric)), 'w') as f:
            f.write('Best epoch: {}\n'.format(best_epoch))
            f.write('-' * 30 + '\n')
            max_length = max([len(key) for key in test_metrics.keys()])
            for key, value in test_metrics.items():
                f.write('{}: {:.4f}\n'.format(key.ljust(max_length), value))
    # Infer train set
    logger.info('Infer train set')
    train_infer_loader = get_train_infer_dataloader(args)
    infer_save_dir = os.path.join(exp_folder, 'infer')
    os.makedirs(infer_save_dir, exist_ok=True)
    for (target_metric, model_path), best_epoch in zip(early_stopping.get_best_model_paths().items(), early_stopping.best_epoch):
        load_states(model_path, device, model)
        logger.info('Load best model from "{}"'.format(model_path))
        train_infer_metrics = val(mixed_precision=args.val_mixed_precision,
                                memory_format=args.memory_format,
                                patch_label_type='none',
                                froc_det_thresholds=args.test_froc_det_thresholds,
                                iou_threshold=args.test_iou_threshold,
                                model = model,
                                detection_postprocess=test_det_postprocess,
                                dataloader = val_loader, 
                                device = device,
                                image_spacing = IMAGE_SPACING,
                                series_list_path=args.test_set,
                                exp_folder=exp_folder,
                                epoch = 'infer_best_{}'.format(target_metric),
                                nodule_type_diameters=NODULE_TYPE_DIAMETERS,
                                min_d=args.min_d,
                                min_size=args.min_size,
                                nodule_size_mode=args.nodule_size_mode)
        write_metrics(train_infer_metrics, epoch, 'infer_train/best_{}'.format(target_metric), writer)
        with open(os.path.join(infer_save_dir, 'infer_train_best_{}.txt'.format(target_metric)), 'w') as f:
            f.write('Best epoch: {}\n'.format(best_epoch))
            f.write('-' * 30 + '\n')
            max_length = max([len(key) for key in train_infer_metrics.keys()])
            for key, value in train_infer_metrics.items():
                f.write('{}: {:.4f}\n'.format(key.ljust(max_length), value))
    writer.close()