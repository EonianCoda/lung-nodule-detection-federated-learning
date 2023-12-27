import torch.nn as nn
import torch
from typing import List, Tuple

class SSLModel(nn.Module):
    def __init__(self, model, feat_keys: List[str]):
        super().__init__()
        self.model = model
        
        self._features = []
        feat_i = 0
        for name, module in self.model.named_modules():
            if name in feat_keys:
                module.register_forward_hook(self.save_feature(feat_i))
                self._features.append(None)
                feat_i += 1
        
    def save_feature(self, feat_id: int):
        def hook(module, input, output):
            self._features[feat_id] = output
        return hook

    def forward(self, x) -> Tuple[List[torch.Tensor], torch.Tensor]:
        outputs = self.model(x)
        return self._features, outputs