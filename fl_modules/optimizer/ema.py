import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class EMA:
    def __init__(self, model: nn.Module, momentum: float = 0.999, warmup_steps: int = 0, apply_buffer: bool = True):
        self.model = model
        self.momentum = momentum
        self.warmup_steps = warmup_steps
        self.apply_buffer = apply_buffer
        self.last_step = 0
        self.shadow = {}
        self.backup = {}
        if self.apply_buffer:
            self.buffer_shadow = {}
            self.buffer_backup = {}

    def _get_momentum(self):
        if self.warmup_steps <= 0 or self.last_step > self.warmup_steps:
            return self.momentum
        else:
            return min(max((self.last_step / self.warmup_steps) * self.momentum, 0), self.momentum)

    def register(self):
        """Register model parameters for EMA.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
        if self.apply_buffer:
            for name, buffer in self.model.named_buffers():
                if 'num_batches_tracked' not in name:
                    self.buffer_shadow[name] = buffer.data.clone()
        
    def to(self, device):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.shadow[name].to(device)
        
        if self.apply_buffer:
            for name in self.buffer_shadow.keys():
                self.buffer_shadow[name] = self.buffer_shadow[name].to(device)

    def update(self):
        self.last_step += 1
        momentum = self._get_momentum()
        if momentum == 0:
            return
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].data.mul_(momentum).add_(param.data, alpha=1 - momentum)

        if self.apply_buffer:
            for name, buffer in self.model.named_buffers():
                if 'num_batches_tracked' not in name:
                    self.buffer_shadow[name].data.mul_(momentum).add_(buffer.data, alpha=1 - momentum)

    def apply_shadow(self, need_backup: bool = True):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if need_backup:
                    self.backup[name] = param.data
                param.data = self.shadow[name]
                
        if self.apply_buffer:
            for name, buffer in self.model.named_buffers():
                if 'num_batches_tracked' not in name:
                    if need_backup:
                        self.buffer_backup[name] = buffer.data
                    buffer.data = self.buffer_shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
        
        if self.apply_buffer:
            for name, buffer in self.model.named_buffers():
                if 'num_batches_tracked' not in name:
                    buffer.data = self.buffer_backup[name]
            self.buffer_backup = {}

    def state_dict(self) -> dict:
        if self.apply_buffer:
            return {
                'apply_buffer': self.apply_buffer,
                'shadow': self.shadow,
                'buffer_shadow': self.buffer_shadow,
                'backup': self.backup,
                'buffer_backup': self.buffer_backup,
                'last_step': self.last_step,
                'warmup_steps': self.warmup_steps,
                'momentum': self.momentum
            }
        else:
            return {
                'apply_buffer': self.apply_buffer,
                'shadow': self.shadow,
                'backup': self.backup,
                'last_step': self.last_step,
                'warmup_steps': self.warmup_steps,
                'momentum': self.momentum
            }

    def load_state_dict(self, state_dict: dict):
        self.shadow = state_dict['shadow']
        self.backup = state_dict['backup']
        self.last_step = state_dict['last_step']
        self.warmup_steps = state_dict['warmup_steps']
        self.momentum = state_dict['momentum']
        self.apply_buffer = state_dict['apply_buffer']
        
        if self.apply_buffer:
            self.buffer_shadow = state_dict['buffer_shadow']
            self.buffer_backup = state_dict['buffer_backup']