import os
from os.path import join
import argparse

from offline_fl.server.server import Server
from offline_fl.utilities import load_yaml, get_local_time_in_taiwan, init_seed, setup_logging, write_yaml

def get_args():
    parser = argparse.ArgumentParser(description = 'Offline Federated Learning')
    parser.add_argument('--exp_name', type = str, default = '', help = 'Experiment name')
    parser.add_argument('--resume_folder', type = str, default = None, help = 'Resume from a folder')
    parser.add_argument('--pretrained_model_path', type = str, default = None, help = 'Path to pretrained model')
    parser.add_argument('--config_path', type = str, default = './config/stage1.yaml', help = 'Path to config file')
    parser.add_argument('--clients_config_path', type = str, default = './config/clients/stage1_clients.yaml', help = 'Path to clients config file')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    exp_name = args.exp_name
    config_path = args.config_path
    clients_config_path = args.clients_config_path
    resume_folder = args.resume_folder
    pretrained_model_path = args.pretrained_model_path
    
    if pretrained_model_path is not None and resume_folder is not None:
        raise ValueError('Cannot specify both pretrained_model_path and resume_folder')
    
    # Resume from a folder
    if resume_folder is not None:
        exp_folder = resume_folder
        exp_name = os.path.basename(exp_folder)
        config_path = join(exp_folder, 'experiment_config.yaml')
        clients_config_path = join(exp_folder, 'clients_config.yaml')
        config = load_yaml(config_path)
        clients_config = load_yaml(clients_config_path)
        resume = True
    else: # Start a new experiment
        config = load_yaml(config_path)
        clients_config = load_yaml(clients_config_path)
        cur_time = get_local_time_in_taiwan()
        timestamp = "[%d-%02d-%02d-%02d%02d]" % (cur_time.year, 
                                                cur_time.month, 
                                                cur_time.day, 
                                                cur_time.hour, 
                                                cur_time.minute)
        if exp_name != '':
            exp_name = timestamp + f'_{exp_name}'
        else:
            exp_name = timestamp
        exp_folder = '{}/{}'.format(config['common']['save_dir'], exp_name)
        resume = False
        
    setup_logging(log_file = join(exp_folder, 'log.txt'))
    init_seed(config['common']['seed'])
    
    # Save config
    write_yaml(join(exp_folder, 'experiment_config.yaml'), config, default_flow_style = None)
    write_yaml(join(exp_folder, 'clients_config.yaml'), load_yaml(clients_config_path), default_flow_style = None)
    # Build server
    server = Server(config = config, 
                    clients_config = clients_config,
                    exp_folder = exp_folder, 
                    resume = resume, 
                    pretrained_model_path = pretrained_model_path)
    
    server.start()