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
from fl_modules.client.client import Client
from fl_modules.utilities.nodule_metrics import NoduleMetrics
from fl_modules.utilities import build_instance, build_class
from fl_modules.utilities.draw_fig import MetricDrawer
from fl_modules.inference.utils import compute_recall, compute_precision, compute_f1_score
from fl_modules.model.ema import EMA


logger = logging.getLogger(__name__)

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
        self.enable_progress_bar = self.config['common']['enable_progress_bar']
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
        
    def start(self):
        self._init_training()
        total_rounds = self.config['server']['total_rounds']
        # Training and validation
        for round_number in range(self.start_round, total_rounds):
            logger.info(f"Round {round_number}/{total_rounds - 1}")
            self.one_round(round_number)
            
        logger.info('Best model metric: {:.4f} at round {}'.format(self.best_model_metric, self.best_model_round))
        # Testing with best model
        logger.info("Testing with best model")
        self.testing_and_save_metrics()
        self.writer.close()
        
        # Draw figure
        drawer = MetricDrawer(join(self.exp_folder, 'tensorboard'))
        figure_save_folder = join(self.exp_folder, 'figure')
        os.makedirs(figure_save_folder, exist_ok = True)
        drawer.save_figure(figure_save_folder)
        
        if not self.save_local_state:
            for client_name, client in self._clients.items():
                if os.path.exists(join(client.client_folder, 'model')):
                    shutil.rmtree(join(client.client_folder, 'model'))
                if os.path.exists(join(client.client_folder, 'optimizer')):
                    shutil.rmtree(join(client.client_folder, 'optimizer'))
                if os.path.exists(join(client.client_folder, 'ema')):
                    shutil.rmtree(join(client.client_folder, 'ema'))
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
            # Training
            train_metrics = client.train(round_number, model = self.model, optimizer = self.optimizer, ema = self.ema)
            client_train_metrics[client_name] = train_metrics
            for metric_name, metric_value in train_metrics.items():
                logger.info(f"Client '{client.name}' train metric '{metric_name}' = {metric_value:.4f}")
                
            # Validation
            if self.apply_ema:
                self.ema.apply_shadow()
            
            # For scaffold, we need to update control variate before validation
            if hasattr(self.optimizer, 'update_control_variate'):
                self.optimizer.update_control_variate()
                
            val_metrics = client.val(round_number, model = self.model, is_global = False)
            client_val_local_metrics[client_name] = val_metrics
            for metric_name, metric_value in val_metrics.items():
                logger.info(f"Client '{client.name}' val metric '{metric_name}' = {metric_value:.4f}")
            
            # Save client model, optimizer and ema state
            client.save_model_state(self.model, round_number)           
            if self.optimizer_aggregaion_strategy != 'reset':
                client.save_optimizer_state(self.optimizer, round_number)
            if self.apply_ema:
                client.save_ema_state(self.ema, round_number)
                self.ema.restore()
                
        self.write_tensorboard(client_train_metrics, round_number, 'train')
        self.write_tensorboard(client_val_local_metrics, round_number, 'val_local')
        
        # Aggregate
        self.apply_aggregation(round_number)
        
        # Use aggregated model to validate
        logger.info(f"Use aggregated model to validate")
        self.load_working_state(round_number, list(self._clients.values())[0])
        for client_name, client in self._clients.items():
            val_metrics = client.val(round_number, model = self.model, is_global = True)
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
        aggregated_model_state_dict, customized_model_state_dict, aggregated_optimizer_state_dict, customized_optimizer_state_dict = self.aggregate_fn(self._clients, self.client_weights, round_number)
        if not self.is_customized_model:
            torch.save(aggregated_model_state_dict, self.global_model)
        else:
            for client_name in self._clients.keys():
                aggregated_model_state_dict.update(customized_model_state_dict[client_name])
                torch.save(aggregated_model_state_dict, join(self.working_folder, f'{client_name}_model.pt'))
        
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
        
        Before training, we need to load working state from file, including: model, optimizer, ema
        """
        # Load model state
        if round_number == 0 or not self.is_customized_model:
            self.model.load_state_dict(torch.load(self.global_model, map_location = self.device))
        else: # Load customized model state
            self.model.load_state_dict(torch.load(join(self.working_folder, f'{client.name}_model.pt'), map_location = self.device))
        
        # Load optimizer state based on optimizer_aggregaion_strategy
        if self.optimizer_aggregaion_strategy == 'continue_global': # Load optimizer state from server
            if round_number == 0 or not self.is_customized_optimizer: # aggregated optimizer
                optimizer_path = self.global_optimizer
            else: # customized optimizer
                optimizer_path = join(self.working_folder, f'{client.name}_optimizer.pt')
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location = self.device))
        elif self.optimizer_aggregaion_strategy == 'continue_local': # Load optimizer state from client
            if round_number == 0:
                self.optimizer = self.build_optimizer(self.model)
            else:
                client.load_optimizer_state(self.optimizer, round_number - 1, self.device)
        elif self.optimizer_aggregaion_strategy == 'reset': # reset optimizer every round_number
            self.optimizer = self.build_optimizer(self.model)
        else:
            raise ValueError(f"Unknown optimizer aggregation strategy '{self.optimizer_aggregaion_strategy}'")
        
        # Load ema state
        if self.apply_ema:
            self.ema.load_state_dict(torch.load(self.global_ema, map_location = self.device))
        
    def save_global_state(self, save_path: str) -> None:
        """Save global state to file
        
        Saving global state includes: model, optimizer(if optimizer_aggregaion_strategy == 'continue_global'), ema(if apply_ema == True)
        """
        global_state_dict = {'model_state_dict': self.model.state_dict(),
                             'model_structure': self.model}
        if self.optimizer_aggregaion_strategy == 'continue_global':
            global_state_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        if self.apply_ema:
            global_state_dict['ema_state_dict'] = self.ema.state_dict()
            
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

        # if is_val:
        #     # Reset recall, precision, f1_score based on weighted sum of tp, fp, fn, tn of different clients
        #     avg_metrics['recall'] = compute_recall(avg_metrics['tp'], avg_metrics['fn'])
        #     avg_metrics['precision'] = compute_precision(avg_metrics['tp'], avg_metrics['fp'])
        #     avg_metrics['f1_score'] = compute_f1_score(avg_metrics['recall'], avg_metrics['precision'])
            
        return avg_metrics
        
    def testing_and_save_metrics(self):
        # Load best model
        best_model_state_dict = torch.load(join(self.exp_folder, 'best_model.pth'), map_location = self.device)['model_state_dict']
        self.model.load_state_dict(best_model_state_dict)
        
        testing_save_path = join(self.exp_folder, 'testing_result.csv')
        # Testing
        client_test_metrics = dict()
        for client_name, client in self._clients.items():
            test_metrics = client.test(model = self.model)
            client_test_metrics[client.name] = test_metrics
            
            # Write metrics to csv file for each client
            metrics = NoduleMetrics(test_metrics)
            series_list_path = os.path.basename(client.test_series_list_path)
            
            params = copy.deepcopy(self.server_config['actions']['test']['params'])
            if params is None:
                params = dict()    
            params['series_list_path'] = series_list_path
            if 'log_metric' in params:
                params.pop('log_metric')
            params['client_name'] = client_name
            metrics.write_metric_csv(params, testing_save_path)
        
        # Calculate average metrics of different nodule types
        sum_test_metrics = dict()
        for client_name, metrics in client_test_metrics.items():
            for nodule_type in metrics.keys():
                if nodule_type not in sum_test_metrics:
                    sum_test_metrics[nodule_type] = defaultdict(float)
                for metric_key in ['tp', 'fp', 'fn', 'tn']:
                    sum_test_metrics[nodule_type][metric_key] += metrics[nodule_type][metric_key]
                    
        # Calculate recall, precision, f1_score
        for metrics in sum_test_metrics.values():
            metrics['recall'] = compute_recall(metrics['tp'], metrics['fn'])
            metrics['precision'] = compute_precision(metrics['tp'], metrics['fp'])
            metrics['f1_score'] = compute_f1_score(metrics['recall'], metrics['precision'])
        
        # Write average metrics to csv file
        metrics = NoduleMetrics(sum_test_metrics)
        params = copy.deepcopy(self.server_config['actions']['test']['params'])
        if params == None:
            params = dict()
        params['series_list_path'] = '+'.join([os.path.basename(client.test_series_list_path) for client in self._clients.values()])
        params['client_name'] = 'server'
        if 'log_metric' in params:
            params.pop('log_metric')
        metrics.write_metric_csv(params, testing_save_path)
        
        self.writer.add_hparams(hparam_dict=params, metric_dict = dict(sum_test_metrics['all']), run_name = 'testing_result')
        
    def _init_training(self):
        self._init_model()
        self._init_optimizer()
        self._init_ema()
        self._init_clients()
        self._init_aggregation()
        self._init_best_model_metric()
        self.writer = SummaryWriter(log_dir = join(self.exp_folder, 'tensorboard'))
    
    def _init_model(self):
        self.model = build_instance(self.server_config['model']['template'], self.server_config['model']['params'])
        self.global_model = join(self.working_folder, 'global_model.pt')
        
        # Load pretrained model
        if self.pretrained_model_path is not None:
            logger.info(f"Load pretrained model from '{self.pretrained_model_path}'")
            pretrained_model_state_dict = torch.load(self.pretrained_model_path, map_location = self.device)['model_state_dict']
            self.model.load_state_dict(pretrained_model_state_dict)
        
        # Resume from past experiment
        if self.resume:
            self.start_round = len(os.listdir(join(self.server_folder, 'model')))
            if not os.path.exists(self.global_model):
                torch.save(self.model.state_dict(), self.global_model)
        else:
            self.start_round = 0
            # Save initial model
            torch.save(self.model.state_dict(), self.global_model)
        self.model.to(self.device)
    
    def _init_optimizer(self):
        self.optimizer_aggregaion_strategy = self.server_config['aggregation']['optimizer_aggregate_strategy']
        self.optimizer = self.build_optimizer(self.model)
        
        # If optimizer_aggregaion_strategy is 'continue_global', we need to maintain a global optimizer to ensure different 
        # clients use the same optimizer state to update global weights at same round
        if self.optimizer_aggregaion_strategy == 'continue_global':
            self.global_optimizer = join(self.working_folder, 'global_optimizer.pt')
            if not self.resume or (self.resume and not os.path.exists(self.global_optimizer)):
                torch.save(self.optimizer.state_dict(), self.global_optimizer) # Save initial optimizer
        else:
            self.global_optimizer = None
    
    def build_optimizer(self, model):
        optimizer_template = build_class(self.server_config['optimizer']['template'])
        params = copy.deepcopy(self.server_config['optimizer']['params'])
        return optimizer_template(model.parameters(), **params)
    
    def _init_ema(self):
        self.apply_ema = self.server_config['ema']['apply']
        if self.apply_ema:
            self.ema = EMA(self.model, **self.server_config['ema']['params'])
            self.ema.register()
            self.global_ema = join(self.working_folder, 'global_ema.pt')
            if not self.resume or (self.resume and not os.path.exists(self.global_ema)):
                torch.save(self.ema.state_dict(), self.global_ema)
        else:
            self.ema = None
    
    def _init_clients(self):
        # Prepare training, validation and testing function
        train_fn = build_class(self.server_config['actions']['train']['template'])
        train_fn_params = self.server_config['actions']['train']['params'] 
        
        val_fn = build_class(self.server_config['actions']['val']['template'])
        val_fn_params = self.server_config['actions']['val']['params']
        
        test_fn = build_class(self.server_config['actions']['test']['template'])
        test_fn_params = self.server_config['actions']['test']['params']
        
        # Prepare clients
        clients = dict()
        self.num_of_client = len(self.clients_config)
        for client_name in self.clients_config.keys():
            # Update client config
            default_dataset_params_config = copy.deepcopy(self.config['client']['dataset']['params'])
            dataset_params_config = dict()
            for identifier in self.clients_config[client_name]['dataset_params'].keys():
                params = copy.deepcopy(default_dataset_params_config)
                params.update(self.clients_config[client_name]['dataset_params'][identifier])
                dataset_params_config[identifier] = params
                
            client = Client(name = client_name, 
                            client_folder = join(self.exp_folder, 'client', client_name),
                            client_config = self.config['client'],
                            dataset_params_config = dataset_params_config, 
                            model = self.model,
                            optimizer = self.optimizer,
                            ema = self.ema,
                            device = self.device,
                            enable_progress_bar = self.enable_progress_bar)
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