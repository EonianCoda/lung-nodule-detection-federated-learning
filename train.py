import os
import math
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR

from fl_modules.model.ema import EMA
from fl_modules.utilities import get_local_time_in_taiwan, init_seed

def get_optimizer(model: nn.Module, 
                  optimizer_type: str,
                  learning_rate: float, 
                  weight_decay: float = 0.0):
    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(grouped_parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(grouped_parameters, lr=learning_rate, weight_decay=weight_decay, momentum=0.9, nesterov=True)
    return optimizer

def write_metrics(metrics: dict, 
                epoch: int,
                prefix: str,
                writer):
    for metric, value in metrics.items():
        writer.add_scalar(f'{prefix}/{metric}', value, global_step = epoch)
    writer.flush()

def save_states(model: nn.Module, 
                optimizer: optim.Optimizer,
                save_path: str = './model.pth',
                scheduler: optim.lr_scheduler = None,
                ema: EMA = None):
    save_dict = {'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'model_structure': model}
    if ema is not None:
        save_dict['ema'] = ema.state_dict()
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()
    torch.save(save_dict, save_path)

def generate_exp_name(args):
    # Experiment Name
    cur_time = get_local_time_in_taiwan()
    timestamp = "[%d-%02d-%02d-%02d%02d]" % (cur_time.year, 
                                            cur_time.month, 
                                            cur_time.day, 
                                            cur_time.hour, 
                                            cur_time.minute)
    if args.exp_name == '':
        exp_name = timestamp
    else:
        exp_name = f'{timestamp}_{args.exp_name}'
    
    # Auto Description
    exp_name += '{}_{}LR{:.0e}WD{:.0e}'.format(args.model, args.optimizer, args.lr, args.weight_decay)
    if not args.no_ema:
        exp_name += '_ema'
    if args.apply_scheduler:
        exp_name += '_cosine_scheduler'
        
    exp_name += '_epoch{}'.format(args.num_epoch)
    return exp_name

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def build_scheduler(optimizer, args):
    num_steps = args.eval_steps * args.num_epoch
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps = num_steps)
    return scheduler
        
def build_train(args):
    # Set seed
    init_seed(args.seed)
    
    # Experiment Name
    exp_name = generate_exp_name(args)
    exp_root = os.path.join(args.save_folder, exp_name)
    
    # Build model and optimizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ## Resume training
    if args.resume_model_path != '': 
        checkpoint = torch.load(args.resume_model_path, map_location = device)
        if 'model_structure' in checkpoint:
            model = checkpoint['model_structure']
        
        optimizer = get_optimizer(model, args.optimizer, args.lr, args.weight_decay)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'ema' in checkpoint:
            ema = EMA(model, decay = args.ema_decay)
            ema.load_state_dict(checkpoint['ema'])
            ema.restore() # Restore model weights from shadow weights to original weights
        else:
            ema = None
            
        if 'scheduler' in checkpoint:
            scheduler = build_scheduler(optimizer, args)
            scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            scheduler = None
        
        # Get start epoch and end epoch from model name, e.g. 1.pth
        model_name = os.path.basename(args.resume_model_path)
        start_epoch = int(model_name.split('.')[0])
        end_epoch = start_epoch + args.num_epoch
        
        exp_name = os.path.basename(os.path.dirname(args.resume_model_path))
        exp_root = os.path.dirname(os.path.dirname(args.resume_model_path))
    else: 
        # Build new model
        if args.model == 'resnet9':
            from fl_modules.model.resnet9 import ResNet9
            model = ResNet9()
        elif args.model == 'wide_resnet':
            from fl_modules.model.wide_resnet import WideResNet
            model = WideResNet()
        model = model.to(device)
        
        optimizer = get_optimizer(model, args.optimizer, args.lr, args.weight_decay)
        start_epoch = 0
        end_epoch = args.num_epoch
        # Register EMA
        if not args.no_ema:
            ema = EMA(model, decay = args.ema_decay)  
            ema.register()
        else:
            ema = None
            
        if args.apply_scheduler:
            scheduler = build_scheduler(optimizer, args)
        else:
            scheduler = None
            
    return model, optimizer, scheduler, ema, start_epoch, end_epoch, device, exp_root, exp_name