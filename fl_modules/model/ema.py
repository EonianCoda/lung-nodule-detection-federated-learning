import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.99):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        """Register model parameters for EMA.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def to(self, device):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.shadow[name].to(device)

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self) -> dict:
        return {
            'shadow': self.shadow,
            'backup': self.backup
        }

    def load_state_dict(self, state_dict: dict):
        self.shadow = state_dict['shadow']
        self.backup = state_dict['backup']