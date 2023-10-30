import torch

from typing import List, Dict, Any, Tuple, Optional
from fl_modules.client.client import Client

from .aggregation import Aggregation


class FedAvg(Aggregation):
    def __init__(self, 
                model: Dict[str, Any],
                optimizer: Dict[str, Any] = None):
        super().__init__(model, optimizer)        
        
    def __call__(self, clients: Dict[str, Client], client_weights: Dict[str, float], round_number: int) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        # Initialize aggregated model and optimizer
        aggregated_model_state_dict = {key: torch.zeros_like(value) for key, value in self.model.state_dict().items()}
        customized_model_state_dict = dict()
        aggregated_optimizer_state_dict = dict()
        customized_optimizer_state_dict = dict()
        
        for client_name, client in clients.items():
            # Aggregate model
            client.load_state_dict(self.model, 'model', round_number, self.cpu_device)
            local_model_state_dict = self.model.state_dict()
            for key, value in local_model_state_dict.items():
                # customized model state
                if key in self.model_keep_local_state:
                    # Lazy initialization
                    if client_name not in customized_model_state_dict:
                        customized_model_state_dict[client_name] = dict()
                    customized_model_state_dict[client_name][key] = value
                # aggregated model state
                elif value is not None:
                    if value.dtype == torch.long:
                        aggregated_model_state_dict[key] += torch.round(client_weights[client_name] * value).long()
                    else:
                        aggregated_model_state_dict[key] += client_weights[client_name] * value
                
            # Aggregate optimizer
            if self.optimizer is not None:
                client.load_state_dict(self.optimizer, 'optimizer', round_number, self.cpu_device)
                optimizer_state_dict = self.optimizer.state_dict()['state']
                for key, states in optimizer_state_dict.items():
                    for state_key, value in states.items(): # state_key, e.g., 'momentum_buffer' for SGD, 'exp_avg' and 'exp_avg_sq' for Adam
                        # customized optimizer state
                        if state_key in self.optimizer_keep_local_state:
                            # Lazy initialization
                            if client_name not in customized_optimizer_state_dict:
                                customized_optimizer_state_dict[client_name] = dict()
                            if key not in customized_optimizer_state_dict[client_name]:
                                customized_optimizer_state_dict[client_name][key] = dict()
                            # assign value
                            customized_optimizer_state_dict[client_name][key][state_key] = value
                        # aggregated optimizer state
                        elif value is not None:
                            # Lazy initialization
                            if key not in aggregated_optimizer_state_dict:
                                aggregated_optimizer_state_dict[key] = dict()
                            if state_key not in aggregated_optimizer_state_dict[key]:
                                aggregated_optimizer_state_dict[key][state_key] = torch.zeros_like(value)
                            # aggregation
                            aggregated_optimizer_state_dict[key][state_key] += client_weights[client_name] * value
                
                for key in optimizer_state_dict.keys():
                    if len(aggregated_optimizer_state_dict[key]) == 0:
                        del aggregated_optimizer_state_dict[key]
                        
        return aggregated_model_state_dict, customized_model_state_dict, aggregated_optimizer_state_dict, customized_optimizer_state_dict