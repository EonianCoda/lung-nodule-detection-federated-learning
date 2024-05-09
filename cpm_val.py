# %% -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import torch
import os
import logging
### data ###
from fl_modules.dataset.utils import get_image_padding_value
from fl_modules.dataset.split_combine import SplitComb
from torch.utils.data import DataLoader
# import fl_modules.dataset.transform as transform

from fl_modules.client.utils import load_model, load_teacher_model, get_memory_format, load_states


from fl_modules.utilities.logs import setup_logging
from fl_modules.utilities.utils import init_seed, write_yaml, build_class
from config import SAVE_ROOT, DEFAULT_OVERLAP_RATIO, IMAGE_SPACING, NODULE_TYPE_DIAMETERS, DEFAULT_CROP_SIZE

logger = logging.getLogger(__name__)
def get_args():
    parser = argparse.ArgumentParser()
    # training settings
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--val_mixed_precision', action='store_true', default=False, help='use mixed precision')
    parser.add_argument('--batch_size', type=int, default=2, help='input batch size for training (default: 2)')
    parser.add_argument('--crop_size', nargs='+', type=int, default=DEFAULT_CROP_SIZE, help='crop size')
    parser.add_argument('--overlap_ratio', nargs='+', type=float, default=[DEFAULT_OVERLAP_RATIO, DEFAULT_OVERLAP_RATIO, DEFAULT_OVERLAP_RATIO], help='overlap ratio')
    parser.add_argument('--model_path', type=str, default='')
    # data
    parser.add_argument('--val_set', type=str, default='./data/all_client_test.txt', help='val_list')
    parser.add_argument('--min_d', type=int, default=0, help="min depth of nodule, if some nodule's depth < min_d, it will be` ignored")
    parser.add_argument('--min_size', type=int, default=27, help="min size of nodule, if some nodule's size < min_size, it will be ignored")
    parser.add_argument('--post_process_min_size', type=int, default=27, help="min size of nodule, if some nodule's size < min_size, it will be ignored")
    
    parser.add_argument('--patch_label_type', type=str, default='none', help='whether to use some missing labels compared to the original label, none or benign or all or tp')
    
    parser.add_argument('--data_norm_method', type=str, default='none', help='normalize method, mean_std or scale or none')
    parser.add_argument('--memory_format', type=str, default='channels_first')
    parser.add_argument('--pad_water', action='store_true', default=False, help='pad water or not')
    # Val hyper-parameters
    parser.add_argument('--val_iou_threshold', type=float, default=0.1, help='iou threshold for validation')
    parser.add_argument('--val_fixed_prob_threshold', type=float, default=0.65, help='fixed probability threshold for validation')
    # detection-hyper-parameters
    parser.add_argument('--det_post_process_class', type=str, default='networks.detection_post_process')
    parser.add_argument('--det_topk', type=int, default=60, help='topk detections')
    parser.add_argument('--det_nms_threshold', type=float, default=0.05, help='detection nms threshold')
    parser.add_argument('--det_nms_topk', type=int, default=20, help='detection nms topk')
    parser.add_argument('--det_scan_nms_keep_top_k', type=int, default=40, help='scan nms keep top k')
    parser.add_argument('--do_padding', action='store_true', default=False, help='do padding or not')
    parser.add_argument('--det_threshold', type=float, default=0.2, help='detection threshold')
    parser.add_argument('--froc_det_thresholds', nargs='+', type=float, default=[0.2, 0.5, 0.7], help='froc det thresholds')
    # Det technical settings
    parser.add_argument('--apply_lobe', action='store_true', default=False, help='apply lobe or not')
    parser.add_argument('--apply_aug', action='store_true', default=False, help='apply test time augmentation or not')
    # other
    parser.add_argument('--load_teacher_model', action='store_true', default=False, help='load teacher model or not')
    parser.add_argument('--load_ema', action='store_true', default=False, help='load ema model or not')
    parser.add_argument('--nodule_size_mode', type=str, default='seg_size', help='nodule size mode, seg_size or dhw')
    parser.add_argument('--max_workers', type=int, default=4, help='max number of workers, num_workers = min(batch_size, max_workers)')
    args = parser.parse_args()
    if args.det_threshold != args.froc_det_thresholds[0]:
        raise ValueError(f'det_threshold = {args.det_threshold} should be equal to froc_det_thresholds[0] = {args.froc_det_thresholds[0]}')
    return args

def prepare_validation(args, device):
    DetectionPostprocess = build_class('{}.DetectionPostprocess'.format(args.det_post_process_class))                
    detection_postprocess = DetectionPostprocess(topk=args.det_topk, 
                                                 threshold=args.det_threshold, 
                                                 nms_threshold=args.det_nms_threshold,
                                                 nms_topk=args.det_nms_topk,
                                                 crop_size=args.crop_size,
                                                 min_size=args.post_process_min_size)
    # load model
    logger.info('Load model from "{}"'.format(args.model_path))
    if args.load_teacher_model:
        logger.info('Load teacher model')
        model = load_teacher_model(args.model_path)
    else:
        model = load_model(args.model_path)
        
    if args.load_ema:
        from fl_modules.optimizer.ema import EMA
        ema = EMA(model)
        ema.register()
        
        load_states(args.model_path, device, model, ema = ema)
        ema.apply_shadow() # apply shadow to model
        
    memory_format = get_memory_format(getattr(args, 'memory_format', None))
    model = model.to(device = device, memory_format=memory_format)
    return model, detection_postprocess

def val_data_prepare(args, model_out_stride=4):
    crop_size = args.crop_size
    overlap_size = [int(crop_size[i] * args.overlap_ratio[i]) for i in range(len(crop_size))]
    pad_value = get_image_padding_value(args.data_norm_method, use_water=args.pad_water)
    
    logger.info('Crop size: {}, overlap size: {}'.format(crop_size, overlap_size))
    if args.do_padding:
        logger.info('Do padding: True, pad value: {}'.format(pad_value))
    else:
        logger.info('Do padding: False')
    logger.info('Apply lobe: True' if args.apply_lobe else 'Apply lobe: False')
    
    split_comber = SplitComb(crop_size=crop_size, overlap_size=overlap_size, pad_value=pad_value, do_padding=args.do_padding)
    
    if args.apply_aug:
        from fl_modules.dataset.cpm_dataset_val_aug import DetDataset
        from fl_modules.dataset.collate import infer_aug_collate_fn as infer_collate_fn
        logger.info('Test time augmentation: True')
        args.batch_size = 1 # batch size should be 1 when apply test time augmentation
    else:
        from fl_modules.dataset.cpm_dataset import DetDataset
        from fl_modules.dataset.collate import infer_collate_fn
        logger.info('Test time augmentation: False')

    val_dataset = DetDataset(series_list_path = args.val_set, SplitComb=split_comber, image_spacing=IMAGE_SPACING, norm_method=args.data_norm_method, apply_lobe=args.apply_lobe, out_stride=model_out_stride)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=min(args.batch_size, args.max_workers) , collate_fn=infer_collate_fn, pin_memory=True)
    
    logger.info("There are {} samples in the val set".format(len(val_loader.dataset)))
    return val_loader

if __name__ == '__main__':
    args = get_args()
    base_exp_folder = os.path.join(os.path.dirname(args.model_path), 'val_results')
    setup_logging(log_file=os.path.join(base_exp_folder, 'val.log'))
    
    # Get all models in the folder
    if '*' not in args.model_path: # validation all models in the folder
        model_paths = [args.model_path]
    else:
        model_folder = os.path.dirname(args.model_path)
        model_paths = [os.path.join(model_folder, f) for f in os.listdir(model_folder) if f.endswith('.pth')]
        
    # Get val function
    if args.apply_aug:
        from fl_modules.client.cpm_val_aug import val
    else:
        from fl_modules.client.cpm_val import val
        
    if len(args.overlap_ratio) == 1:
        args.overlap_ratio = [args.overlap_ratio[0]] * 3
    if len(args.crop_size) == 1:
        args.crop_size = [args.crop_size[0]] * 3
        
    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        exp_folder = os.path.join(base_exp_folder, model_name)
        args.model_path = model_path
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, detection_postprocess = prepare_validation(args, device)
        init_seed(args.seed)
        
        val_loader = val_data_prepare(args, getattr(model, 'out_stride', 4))
        
        write_yaml(os.path.join(exp_folder, 'val_config.yaml'), args)
        logger.info('Save validation results to "{}"'.format(exp_folder))
        logger.info('Val set: "{}"'.format(args.val_set))
        
        save_folder_name = '{}_{}'.format(os.path.basename(model_path).split('.')[0], os.path.basename(args.val_set).split('.')[0])
        if args.do_padding:
            save_folder_name += '_pad'
        if args.apply_lobe:
            save_folder_name += '_lobe'
        if args.apply_aug:
            save_folder_name += '_aug'
        
        if args.crop_size[0] != DEFAULT_CROP_SIZE[0] or args.crop_size[1] != DEFAULT_CROP_SIZE[1] or args.crop_size[2] != DEFAULT_CROP_SIZE[2]:
            if args.crop_size[0] == args.crop_size[1] and args.crop_size[0] == args.crop_size[2]:
                save_folder_name += '_crop{}'.format(args.crop_size[0])
            else:
                save_folder_name += '_crop{}_{}_{}'.format(args.crop_size[0], args.crop_size[1], args.crop_size[2])
                
        if args.overlap_ratio[0] != DEFAULT_OVERLAP_RATIO or args.overlap_ratio[1] != DEFAULT_OVERLAP_RATIO or args.overlap_ratio[2] != DEFAULT_OVERLAP_RATIO:
            if args.overlap_ratio[0] == args.overlap_ratio[1] and args.overlap_ratio[0] == args.overlap_ratio[2]:
                save_folder_name += '_ol{:.2f}'.format(args.overlap_ratio[0])
            else:
                save_folder_name += '_ol{:.2f}_{:.2f}_{:.2f}'.format(args.overlap_ratio[0], args.overlap_ratio[1], args.overlap_ratio[2])
        
        if args.patch_label_type != 'none':
            save_folder_name += '_PT{}'.format(args.patch_label_type)
        
        metrics = val(mixed_precision=args.val_mixed_precision,
                        memory_format=args.memory_format,
                        patch_label_type=args.patch_label_type,
                        froc_det_thresholds=args.froc_det_thresholds,
                        model = model,
                        detection_postprocess=detection_postprocess,
                        dataloader = val_loader, 
                        device = device,
                        image_spacing = IMAGE_SPACING,
                        series_list_path=args.val_set,
                        exp_folder=exp_folder,
                        epoch=save_folder_name,
                        nodule_type_diameters=NODULE_TYPE_DIAMETERS,
                        min_d=args.min_d,
                        min_size=args.min_size,
                        nodule_size_mode=args.nodule_size_mode)
        
        save_txt_path = os.path.join(exp_folder, 'val_metrics_{}.txt'.format(save_folder_name))
        max_length = max([len(key) for key in metrics.keys()])
        with open(save_txt_path, 'w') as f:
            for k, v in metrics.items():
                if int(v) == v:
                    f.write('{}: {}\n'.format(k.ljust(max_length), int(v)))
                else:
                    f.write('{}: {:.4f}\n'.format(k.ljust(max_length), v))