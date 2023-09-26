import torch

from typing import List, Dict, Any, Tuple, Optional
from fl_modules.client.client import Client

class Aggregation:
    def __init__(self, 
                model: Dict[str, Any],
                optimizer: Dict[str, Any] = None):
        if model.get('model', None) is None:
            raise ValueError('model cannot be None')
        self.model = model['model']
        self.model_keep_local_state = model.get('keep_local_state', [])
        
        if optimizer is not None and optimizer.get('optimizer', None) is not None:
            self.optimizer = optimizer['optimizer']
            self.optimizer_keep_local_state = optimizer['keep_local_state']
        else:
            self.optimizer = None
            self.optimizer_keep_local_state = []
    
        self.cpu_device = torch.device('cpu')
        
    def __call__(self, clients: Dict[str, Client], client_weights: Dict[str, float], round_number: int) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Return:
            A tuple of (aggregated_model_state_dict, customized_model_state_dict, aggregated_optimizer_state_dict, customized_optimizer_state_dict)
        """
        raise NotImplementedError
    
    def is_customized_model(self) -> bool:
        return len(self.model_keep_local_state) > 0
    
    def is_customized_optimizer(self) -> bool:
        return len(self.optimizer_keep_local_state) > 0