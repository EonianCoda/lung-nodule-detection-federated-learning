import argparse
import os
from os.path import join
import numpy as np
from fl_modules.dataset.utils import prepare_cifar10_datasets
from fl_modules.utilities.utils import write_yaml

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--seed', type = int, default = 1029)
    parser.add_argument('--train_val_test_split', type = list, default = [0.8, 0.1, 0.1], nargs='+')
    parser.add_argument('--supervised_ratio', type = float, default = 0.1)
    parser.add_argument('--is_balance', action='store_false', default=True)
    parser.add_argument('--save_dir', type=str, default='./data/fl_data')
    parser.add_argument('--client_config_save_path', type=str, default='./config/clients/cifar10_client_config.yaml')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    seed = args.seed
    num_clients = args.num_clients
    train_val_test_split = args.train_val_test_split
    supervised_ratio = args.supervised_ratio
    is_balance = args.is_balance
    save_dir = args.save_dir
    
    
    os.makedirs(save_dir, exist_ok=True)
    
    client_train_s, client_train_u, val_set, test_set = prepare_cifar10_datasets(train_val_test_split=train_val_test_split,
                                                                                    s_u_split=[supervised_ratio, 1 - supervised_ratio],
                                                                                    num_clients = num_clients,
                                                                                    is_balance = is_balance,
                                                                                    seed = seed)
    
    val_save_path = join(save_dir, 'val.npz')
    test_save_path = join(save_dir, 'test.npz')
    np.savez(val_save_path, **val_set)
    np.savez(test_save_path, **test_set)
    
    
    clients_config = {}
    template = "fl_modules.dataset.cifar10_dataset.Cifar10Dataset"
    for client_id in client_train_s.keys():
        train_s_save_path = join(save_dir, f'client_{client_id}_train_s.npz')
        train_u_save_path = join(save_dir, f'client_{client_id}_train_u.npz')
        
        np.savez(train_s_save_path, **client_train_s[client_id])
        np.savez(train_u_save_path, **client_train_u[client_id])
        
        config = {'dataset': {'train_s': {'template': template, 'params': {'data': train_s_save_path, 'targets': ['weak'], 'batch_size': 64}},
                                    'train_u': {'template': template, 'params': {'data': train_u_save_path, 'targets': ['weak', 'strong'], 'batch_size': 64 * 7}},
                                    'val': {'template': template, 'params': {'data': val_save_path, 'batch_size': 64 * 2}},
                                    'test': {'template': template, 'params': {'data': test_save_path, 'batch_size': 64 * 2}}}}
        
        clients_config[f'Client_{client_id}'] = config
    write_yaml(args.client_config_save_path, clients_config)