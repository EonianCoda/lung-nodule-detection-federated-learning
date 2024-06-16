import os
from os.path import join
import copy
import json
import logging
from typing import List, Dict, Any
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
import torch
import shutil
from fl_modules.client.ssl_client import Client
from fl_modules.utilities import build_instance, build_class, build_config

logger = logging.getLogger(__name__)

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

class Server:
    def __init__(self, 
                 config: Dict[str, Any],
                 clients_config: Dict[str, Any],
                 exp_folder: str,
                 pretrained_model_path: str = None,
                 resume: bool = False) -> None:
        self.config = config
        self.clients_config = clients_config
        self.server_config = config['server']
        self.save_local_state = self.config['common']['save_local_state']
        # Resume Options
        self.resume = resume
        self.pretrained_model_path = pretrained_model_path
        
        # Prepare folder
        self.exp_folder = exp_folder
        self.server_folder = join(self.exp_folder, 'server')
        self.working_folder = join(self.server_folder, 'working')
        os.makedirs(self.server_folder, exist_ok = True)
        os.makedirs(self.working_folder, exist_ok = True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.total_rounds = self.config['server']['total_rounds']
        self.start_val_round = self.config['server']['start_val_round']
        self.val_interval = self.config['server']['val_interval']
        self.epoch_per_round = self.config['server']['epoch_per_round']
        
    def start(self):
        self._init_training()
        # Generate pseudo labels for clients
        for client_name, client in self._clients.items():
            client._init_train_dataloader()
            dataloader_u = client.train_config.get('dataloader_u', None)
            if dataloader_u is None:
                raise ValueError(f"Client '{client_name}' should provide 'dataloader_u' in train_config")
            
            self.load_working_state(0, client)
            if len(dataloader_u.dataset.labels) == 0:
                client.gen_pseudo_labels(self.model_t, self.pseudo_label_det_postprocess, epoch = 0)
            else:
                logger.info(f"Client '{client_name}' already has provided pseudo labels")
        # Training and validation
        for round_number in range(self.start_round, self.total_rounds):
            logger.info(f"Round {round_number}/{self.total_rounds - 1}")
            self.one_round(round_number)
            
        logger.info('Best model metric: {:.4f} at round {}'.format(self.best_model_metric, self.best_model_round))
        # Testing with best model
        logger.info("Testing with best model")
        self.testing_and_save_metrics()
        self.writer.close()
        
        if not self.save_local_state:
            for client_name, client in self._clients.items():
                if os.path.exists(join(client.client_folder, 'model')):
                    shutil.rmtree(join(client.client_folder, 'model'))
                if os.path.exists(join(client.client_folder, 'optimizer')):
                    shutil.rmtree(join(client.client_folder, 'optimizer'))
                if os.path.exists(join(client.client_folder, 'scheduler')):
                    shutil.rmtree(join(client.client_folder, 'scheduler'))
                if os.path.exists(join(self.server_folder, 'model')):
                    shutil.rmtree(join(self.server_folder, 'model'))
                
        logger.info("End of federated learning")
    
    def one_round(self, round_number: int):
        """One round of federated learning
        
        Args:
            round_number (int): Current round
        """
        client_train_metrics = dict()
        client_val_local_metrics = dict()
        client_val_global_metrics = dict()
        
        for client_name, client in self._clients.items():
            self.load_working_state(round_number, client)
            # For Fedrox or FedProxAdam, we need to update global weights before training
            if hasattr(self.optimizer, 'update_global_weights'):
                self.optimizer.update_global_weights()
            
            # Warmup pseudo ema alpha
            train_dataset_u = client.train_config.get('dataloader_u').dataset
            original_psuedo_update_ema_alpha = getattr(train_dataset_u, 'pseudo_update_ema_alpha')
            pseudo_tracking_warmup_epochs = getattr(train_dataset_u, 'pseudo_tracking_warmup_epochs', 0)
            if round_number <= pseudo_tracking_warmup_epochs and pseudo_tracking_warmup_epochs > 0:
                pseudo_update_ema_alpha = 1 - (1 - original_psuedo_update_ema_alpha) * (round_number / pseudo_tracking_warmup_epochs)
                logger.info('Tracking warmup epoch: {} pseudo update ema alpha: {:.4f}'.format(round_number, pseudo_update_ema_alpha))
            else:
                pseudo_update_ema_alpha = original_psuedo_update_ema_alpha
            train_dataset_u.pseudo_update_ema_alpha = pseudo_update_ema_alpha
            
            # Training
            train_metrics = client.train(round_number = round_number, num_epoch = self.epoch_per_round, model_s = self.model_s, 
                                         model_t = self.model_t, loss_fn = self.loss, semi_loss_fn = self.semi_loss, 
                                         optimizer = self.optimizer, detection_postprocess = self.train_det_postprocess)
            self.scheduler.step()
            client_train_metrics[client_name] = train_metrics
            for metric_name, metric_value in train_metrics.items():
                logger.info(f"Client '{client.name}' train metric '{metric_name}' = {metric_value:.4f}")
            
            # Print lr
            logger.info(f'Client {client.name} lr: {self.scheduler.get_lr()[0]}')
            client_train_metrics[client_name]['lr'] = self.scheduler.get_lr()[0]
                
            # Save client model, optimizer
            client.save_model_state(self.model_s, round_number)
            client.save_model_state(self.model_t, round_number, save_teacher = True)
            client.save_scheduler_state(self.scheduler, round_number)
            if self.optimizer_aggregaion_strategy != 'reset':
                client.save_optimizer_state(self.optimizer, round_number)
            
            # Validation
            if round_number >= self.start_val_round and round_number % self.val_interval == 0:
                # For scaffold, we need to update control variate before validation
                if hasattr(self.optimizer, 'update_control_variate'):
                    self.optimizer.update_control_variate()
                    
                val_metrics = client.val(round_number, model = self.model_t, is_global = False, detection_postprocess = self.val_det_postprocess) # use teacher model to validate
                client_val_local_metrics[client_name] = val_metrics
                for metric_name, metric_value in val_metrics.items():
                    logger.info(f"Client '{client.name}' val metric '{metric_name}' = {metric_value:.4f}")
                
        self.write_tensorboard(client_train_metrics, round_number, 'train')
        if round_number >= self.start_val_round and round_number % self.val_interval == 0:
            self.write_tensorboard(client_val_local_metrics, round_number, 'val_local')
        
        # Aggregate
        self.apply_aggregation(round_number)
        
        # Use aggregated model to validate
        if round_number >= self.start_val_round and round_number % self.val_interval == 0:
            logger.info(f"Use aggregated model to validate")
            self.load_working_state(round_number, list(self._clients.values())[0])
            for client_name, client in self._clients.items():
                val_metrics = client.val(round_number, model = self.model_s, is_global = True, detection_postprocess = self.val_det_postprocess)
                client_val_global_metrics[client_name] = val_metrics
                for metric_name, metric_value in val_metrics.items():
                    logger.info(f"Client '{client.name}' val metric '{metric_name}' = {metric_value:.4f}")
            self.write_tensorboard(client_val_global_metrics, round_number, 'val_global')
        
        # Save global model and optimizer
        self.save_global_state(join(self.server_folder, 'model', f'{round_number}.pth'))
        
        # Save best model and update best model metric
        avg_metrics = self.calculate_average_metrics(client_val_global_metrics)
        self.save_metrics(avg_metrics, 'val_global', round_number)
        for key, value in avg_metrics.items():
            logger.info(f"Server average metric '{key}' = {value:.4f}")
        
        if self.best_model_metric <= avg_metrics[self.best_model_metric_name]:
            logger.info(f"Best model metric '{self.best_model_metric_name}' updated from {self.best_model_metric:.4f} to {avg_metrics[self.best_model_metric_name]:.4f} at round {round_number}")
            self.best_model_metric = avg_metrics[self.best_model_metric_name]
            self.best_model_round = round_number
            with open(join(self.exp_folder, 'best_model.txt'), 'w') as f:
                f.write(f'best_model_round: {round_number}\n')
                f.write(f'best_model_metric_name: {self.best_model_metric_name}\n')
                f.write(f'best_model_metric: {self.best_model_metric:.4f}\n')
                f.write('-'*10 + '\n')
                for key, value in avg_metrics.items():
                    f.write(f'{key}: {value:.4f}\n')
            
            self.save_global_state(join(self.exp_folder, 'best_model.pth'))
   
    def apply_aggregation(self, round_number: int):
        logger.info(f"Aggregate model and optimizer")
        aggregated_model_s_state_dict, customized_model_s_state_dict, aggregated_model_t_state_dict, customized_model_t_state_dict, aggregated_optimizer_state_dict, customized_optimizer_state_dict = self.aggregate_fn(self._clients, self.client_weights, round_number)
        
        if not self.is_customized_model:
            torch.save(aggregated_model_s_state_dict, self.global_model_s)
            torch.save(aggregated_model_t_state_dict, self.global_model_t)
        else:
            for client_name in self._clients.keys():
                aggregated_model_s_state_dict.update(customized_model_s_state_dict[client_name])
                torch.save(aggregated_model_s_state_dict, join(self.working_folder, f'{client_name}_model_s.pt'))
                aggregated_model_t_state_dict.update(customized_model_t_state_dict[client_name])
                torch.save(aggregated_model_t_state_dict, join(self.working_folder, f'{client_name}_model_t.pt'))
        # Optimizer only keep on client
        if self.optimizer_aggregaion_strategy != 'continue_global':
            return
        
        optimizer_state_dict = dict()
        optimizer_state_dict['param_groups'] = self.optimizer.state_dict()['param_groups']
        if not self.is_customized_optimizer:
            optimizer_state_dict['state'] = aggregated_optimizer_state_dict
            torch.save(optimizer_state_dict, self.global_optimizer)
        else:
            for client_name in self._clients.keys():
                for param_key in aggregated_optimizer_state_dict.keys():
                    custom = customized_optimizer_state_dict[client_name].get(param_key, dict())
                    for custom_state_key in custom.keys():
                        aggregated_optimizer_state_dict[param_key][custom_state_key] = custom[custom_state_key]
                optimizer_state_dict['state'] = aggregated_optimizer_state_dict
                torch.save(optimizer_state_dict, join(self.working_folder, f'{client_name}_optimizer.pt'))
            
    def load_working_state(self, round_number: int, client: Client) -> None:
        """Load working state from file
        
        Before training, we need to load working state from file, including: model, optimizer
        """
        # Load model state
        if round_number == 0 or not self.is_customized_model:
            self.model_s.load_state_dict(torch.load(self.global_model_s, map_location = self.device))
            self.model_t.load_state_dict(torch.load(self.global_model_t, map_location = self.device))
        else: # Load customized model state
            self.model_s.load_state_dict(torch.load(join(self.working_folder, f'{client.name}_model_s.pt'), map_location = self.device))
            self.model_t.load_state_dict(torch.load(join(self.working_folder, f'{client.name}_model_t.pt'), map_location = self.device))
            
        # Load optimizer state based on optimizer_aggregaion_strategy
        if self.optimizer_aggregaion_strategy == 'continue_global': # Load optimizer state from server
            if round_number == 0 or not self.is_customized_optimizer: # aggregated optimizer
                optimizer_path = self.global_optimizer
            else: # customized optimizer
                optimizer_path = join(self.working_folder, f'{client.name}_optimizer.pt')
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location = self.device))
        elif self.optimizer_aggregaion_strategy == 'continue_local': # Load optimizer state from client
            if round_number == 0:
                self.optimizer = self.build_optimizer(self.model_s)
            else:
                client.load_optimizer_state(self.optimizer, round_number - 1, self.device)
        elif self.optimizer_aggregaion_strategy == 'reset': # reset optimizer every round_number
            self.optimizer = self.build_optimizer(self.model_s)
        else:
            raise ValueError(f"Unknown optimizer aggregation strategy '{self.optimizer_aggregaion_strategy}'")
        
        # Load scheduler state
        if round_number == 0:
            self.scheduler.load_state_dict(torch.load(self.global_scheduler, map_location = self.device))
        else:
            client.load_scheduler_state(self.scheduler, round_number - 1, self.device)
            
    def save_global_state(self, save_path: str) -> None:
        """Save global state to file
        
        Saving global state includes: model, optimizer(if optimizer_aggregaion_strategy == 'continue_global')
        """
        global_state_dict = {'model_s_state_dict': self.model_s.state_dict(),
                             'model_structure': self.model_s,
                             'model_t_state_dict': self.model_t.state_dict()}
        if self.optimizer_aggregaion_strategy == 'continue_global':
            global_state_dict['optimizer_state_dict'] = self.optimizer.state_dict()
            
        os.makedirs(os.path.dirname(save_path), exist_ok = True)
        torch.save(global_state_dict, save_path)
        
    def save_metrics(self, metrics: Dict[str, float], task: str, round_number: int):
        """Save metrics to json file
        
        Args:
            metrics: Dict[str, float] 
                Metrics
            task: str
                Task name, e.g. 'train', 'val_local', 'val_global'
            round_number: int
                A number to indicate current round
        """
        save_path = join(self.server_folder, 'metrics', f'{task}_{round_number}.json')
        os.makedirs(os.path.dirname(save_path), exist_ok = True)
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent = 4)
    
    def write_tensorboard(self, client_metrics: Dict[str, float], round_number: int, task: str):
        """Write metrics to tensorboard and save to file
        
        Args:
            client_metrics: Dict[str, float]
                Metrics of each client, e.g. {'client1': {'loss': 0.1, 'loss512': 0.2, 'loss256': 0.3}, 'client2': {'loss': 0.1, 'loss512': 0.2, 'loss256': 0.3}}
            round_number: int
                A number to indicate current round
            task: str
                Task name, e.g. 'train', 'val_local', 'val_global'
        """
        # Write client metrics to tensorboard
        for client_name, metric_value in client_metrics.items():
            for metric_name, metric_value in metric_value.items():
                self.writer.add_scalar(f'{client_name}/{task}/{metric_name}', metric_value, round_number)

        # Write average metrics to tensorboard
        is_val = False if task == 'train' else True
        avg_metrics = self.calculate_average_metrics(client_metrics, is_val = is_val)
        for metric_name, metric_value in avg_metrics.items():
            self.writer.add_scalar(f'Server/{task}/{metric_name}', metric_value, round_number)

        self.writer.flush()
        
    def calculate_average_metrics(self, client_metrics: Dict[str, float], is_val: bool = True, weighted: bool = True) -> Dict[str, float]:
        """
        Args:
            client_metrics: Dict[str, float]
                Metrics of each client, e.g. {'client1': {'loss': 0.1, 'loss512': 0.2, 'loss256': 0.3}, 'client2': {'loss': 0.1, 'loss512': 0.2, 'loss256': 0.3}}
            is_val: bool (deprecated, currently always return weighted average metrics)
                If True, calculate recall, precision, f1_score based on weighted sum of tp, fp, fn, tn of different clients
            weighted: bool
                If True, calculate weighted average metrics of different clients based on weights of different clients, otherwise, calculate average metrics of different clients
        Returns:
            avg_metrics: Dict[str, float]
                Average metrics of different clients, e.g. {'loss': 0.1, 'loss512': 0.2, 'loss256': 0.3}
        """
        avg_metrics = defaultdict(float)
        for client_name, metric_value in client_metrics.items():
            for metric_name, metric_value in metric_value.items():
                if weighted:
                    avg_metrics[metric_name] += metric_value * self.client_weights[client_name]
                else:
                    avg_metrics[metric_name] += metric_value                    

        return avg_metrics
        
    def testing_and_save_metrics(self):
        # Load best model
        best_model_state_dict = torch.load(join(self.exp_folder, 'best_model.pth'), map_location = self.device)['model_t_state_dict']
        self.model_t.load_state_dict(best_model_state_dict)
        
        # Testing
        client_test_metrics = dict()
        for client_name, client in self._clients.items():
            test_metrics = client.test(model = self.model_t, detection_postprocess = self.test_det_postprocess)
            client_test_metrics[client.name] = test_metrics
            
        # Calculate average metrics of different nodule types
        sum_test_metrics = dict()
        for client_name, metrics in client_test_metrics.items():
            for nodule_type in metrics.keys():
                if nodule_type not in sum_test_metrics:
                    sum_test_metrics[nodule_type] = defaultdict(float)
                for metric_key in ['tp', 'fp', 'fn', 'tn']:
                    sum_test_metrics[nodule_type][metric_key] += metrics[nodule_type][metric_key]
                    
    def _init_training(self):
        self._init_model()
        self._init_loss()
        self._init_optimizer()
        self._init_scheduler()
        self._init_clients()
        self._init_aggregation()
        self._init_best_model_metric()
        self._init_det_postprocess()
        self.writer = SummaryWriter(log_dir = join(self.exp_folder, 'tensorboard'))
    
    def _init_det_postprocess(self):
        self.train_det_postprocess = build_instance(self.server_config['det_postprocess']['train']['template'], self.server_config['det_postprocess']['train']['params'])
        self.pseudo_label_det_postprocess = build_instance(self.server_config['det_postprocess']['pseudo_label']['template'], self.server_config['det_postprocess']['pseudo_label']['params'])
        self.val_det_postprocess = build_instance(self.server_config['det_postprocess']['val']['template'], self.server_config['det_postprocess']['val']['params'])
        self.test_det_postprocess = build_instance(self.server_config['det_postprocess']['test']['template'], self.server_config['det_postprocess']['test']['params'])
    
    def _init_model(self):
        model_config = self.server_config['model']['params']
        model_config = build_config(model_config)
        model_config['device'] = self.device
        self.model_s = build_instance(self.server_config['model']['template'], model_config)
        self.model_t = build_instance(self.server_config['model']['template'], model_config)
        
        self.global_model_s = join(self.working_folder, 'global_model_s.pt')
        self.global_model_t = join(self.working_folder, 'global_model_t.pt')
        
        # Load pretrained model
        if self.pretrained_model_path is not None:
            logger.info(f"Load pretrained model from '{self.pretrained_model_path}'")
            
            state_dict = torch.load(self.pretrained_model_path, map_location = self.device)
            
            if 'model_state_dict' in state_dict:
                pretrained_model_state_dict = state_dict['model_state_dict']
                logger.info(f"Load model state dict from 'model_state_dict'")
            else:
                pretrained_model_state_dict = state_dict['model_t_state_dict']
                logger.info(f"Load model state dict from 'model_t_state_dict'")
            
            self.model_s.load_state_dict(pretrained_model_state_dict)
            self.model_t.load_state_dict(pretrained_model_state_dict)
            
        # Resume from past experiment
        if self.resume:
            self.start_round = len(os.listdir(join(self.server_folder, 'model')))
            if not os.path.exists(self.global_model_s):
                torch.save(self.model_s.state_dict(), self.global_model_s)
            if not os.path.exists(self.global_model_t):
                torch.save(self.model_t.state_dict(), self.global_model_t)
        else:
            self.start_round = 0
            # Save initial model
            torch.save(self.model_s.state_dict(), self.global_model_s)
            torch.save(self.model_t.state_dict(), self.global_model_t)
        self.model_s.to(self.device)
        self.model_t.to(self.device)
    
    def _init_loss(self):
        loss_config = self.server_config['detection_loss']
        semi_loss_config = self.server_config['unsupervised_detection_loss']
        
        self.loss = build_instance(loss_config['template'], loss_config['params'])
        self.semi_loss = build_instance(semi_loss_config['template'], semi_loss_config['params'])
        
    def _init_optimizer(self):
        logger.info('Initialize optimizer')
        self.optimizer_aggregaion_strategy = self.server_config['aggregation']['optimizer_aggregate_strategy']
        self.optimizer = self.build_optimizer(self.model_s)
        
        # If optimizer_aggregaion_strategy is 'continue_global', we need to maintain a global optimizer to ensure different 
        # clients use the same optimizer state to update global weights at same round
        if self.optimizer_aggregaion_strategy == 'continue_global':
            self.global_optimizer = join(self.working_folder, 'global_optimizer.pt')
            if not self.resume or (self.resume and not os.path.exists(self.global_optimizer)):
                torch.save(self.optimizer.state_dict(), self.global_optimizer) # Save initial optimizer
        else:
            self.global_optimizer = None
    
    def build_optimizer(self, model):
        # Add weight decay
        params = add_weight_decay(model, self.server_config['optimizer']['params']['weight_decay'])
        optimizer_template = build_class(self.server_config['optimizer']['template'])
        kwargs = copy.deepcopy(self.server_config['optimizer']['params'])
        return optimizer_template(params, **kwargs)
    
    def _init_scheduler(self):
        logger.info('Initialize scheduler')
        self.scheduler = self.build_scheduler(self.optimizer)
        self.global_scheduler = join(self.working_folder, 'global_scheduler.pt')
        if not self.resume or (self.resume and not os.path.exists(self.global_scheduler)):
            torch.save(self.scheduler.state_dict(), self.global_scheduler)
    
    def build_scheduler(self, optimizer):
        lr = self.server_config['optimizer']['params']['lr']
        scheduler_template = build_class(self.server_config['scheduler']['template'])
        scheduler = scheduler_template(optimizer, **self.server_config['scheduler']['params'])
        return scheduler
        
    def _init_clients(self):
        logger.info('Initialize clients')
        # Prepare training, validation and testing function
        shared_params = self.server_config['actions']['shared_params']
        
        train_fn = build_class(self.server_config['actions']['train']['template'])
        train_fn_params = self.server_config['actions']['train']['params'] 
        train_fn_params.update(copy.deepcopy(shared_params))
        
        pseudo_label_fn = build_class(self.server_config['actions']['pseudo_label']['template'])
        pseudo_label_fn_params = self.server_config['actions']['pseudo_label']['params']
        pseudo_label_fn_params.update(copy.deepcopy(shared_params))
        
        val_fn = build_class(self.server_config['actions']['val']['template'])
        val_fn_params = self.server_config['actions']['val']['params']
        val_fn_params.update(copy.deepcopy(shared_params))
        
        test_fn = build_class(self.server_config['actions']['test']['template'])
        test_fn_params = self.server_config['actions']['test']['params']
        test_fn_params.update(copy.deepcopy(shared_params))
        
        # Prepare clients
        clients = dict()
        self.num_of_client = len(self.clients_config)
        for client_name in self.clients_config.keys():
            logger.info(f"Initialize client '{client_name}'")
            client = Client(name = client_name, 
                            client_folder = join(self.exp_folder, 'client', client_name),
                            client_config = self.config['client'],
                            dataset_params_config = self.clients_config[client_name]['dataset_params'], 
                            # model = self.model_s,
                            # optimizer = self.optimizer,
                            device = self.device,
                            save_local_state = self.save_local_state)
            client.prepare()
            
            # Build action
            client.build_action(train_fn, train_fn_params, 'train')
            client.build_action(val_fn, val_fn_params, 'val')
            client.build_action(test_fn, test_fn_params, 'test')
            
            # Check if client name already exists
            if client_name in clients:
                raise ValueError(f"Client name '{client_name}' already exists, Please use different name!")
            clients[client_name] = client
        self._clients = clients
        
    def _init_aggregation(self):
        """Initialize aggregation function
        """
        logger.info('Initialize aggregation function')
        model = build_instance(self.server_config['model']['template'], self.server_config['model']['params'])
        # If optimizer_aggregaion_strategy is 'continue_global', we need to aggregate optimizer
        if self.optimizer_aggregaion_strategy == 'continue_global':
            optimizer = self.build_optimizer(model)
        else:
            optimizer = None
            
        aggregate_fn_template = build_class(self.server_config['aggregation']['template'])
        aggregate_fn_params = copy.deepcopy(self.server_config['aggregation']['params'])
        aggregate_fn_params['model']['model'] = model
        aggregate_fn_params['optimizer']['optimizer'] = optimizer
        
        # Initialize aggregation function
        self.aggregate_fn = aggregate_fn_template(**aggregate_fn_params)
        self.is_customized_model = self.aggregate_fn.is_customized_model()
        self.is_customized_optimizer = self.aggregate_fn.is_customized_optimizer()
    
    def _init_best_model_metric(self):
        self.best_model_metric_name = self.server_config['best_model_metric_name']
        self.best_model_metric = 0.0
        self.best_model_round = -1
        if self.resume:
            if not os.path.exists(join(self.exp_folder, 'best_model.txt')):
                logger.warning(f"Cannot find best model file in '{self.exp_folder}'")
            else: # Load best model metric from file
                with open(join(self.exp_folder, 'best_model.txt'), 'r') as f:
                    lines = f.readlines()
                best_model_round = int(lines[0].split(':')[-1].strip())
                best_model_metric_name = lines[1].split(':')[-1].strip()
                best_model_metric = float(lines[2].split(':')[-1].strip())
                if best_model_metric_name != self.best_model_metric_name:
                    logger.warning(f"Best model metric name '{best_model_metric_name}' in file is different from '{self.best_model_metric_name}' in config file")
                    logger.warning(f"Use '{best_model_metric_name}' as best model metric name")
                    self.best_model_metric_name = best_model_metric_name
                self.best_model_metric = best_model_metric
                self.best_model_round = best_model_round
    
    @property
    def client_weights(self) -> Dict[str, float]:
        if getattr(self, '_client_weights', None) is not None:
            return self._client_weights
        
        client_weights = dict()
        for client_name, client in self._clients.items():
            client_weights[client_name] = len(client.train_set)
            
        # Normalize client weights
        count_samples = sum(client_weights.values()) 
        for client_name, weight in client_weights.items():
            client_weights[client_name] = weight / count_samples
        self._client_weights = client_weights
        return self._client_weights