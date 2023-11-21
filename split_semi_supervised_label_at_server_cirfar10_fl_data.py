import argparse
import os
from os.path import join
import shutil
from fl_modules.dataset.utils import prepare_semi_supervised_label_at_server_cifar10_datasets, save_pickle
from fl_modules.utilities.utils import write_yaml

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--seed', type = int, default = 1029)
    parser.add_argument('--train_val_test_split', nargs=3, type=float, default=[0.9, 0.05, 0.05])
    parser.add_argument('--num_labeled', type = int, default = 1000)
    parser.add_argument('--bs', type=int, default = 64)
    parser.add_argument('--iters', type=int, default = 300)
    parser.add_argument('--is_unbalance', action='store_true', default=False)
    parser.add_argument('--save_dir', type=str, default='./data/fl_data')
    
    parser.add_argument('--client_config_save_path', type=str, default='./config/clients/cifar10_client_config.yaml')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    seed = args.seed
    num_clients = args.num_clients
    train_val_test_split = args.train_val_test_split
    is_unbalance = args.is_unbalance
    save_dir = args.save_dir
    iters = args.iters
    batch_size = args.bs
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    server_train_s, client_train_u, val_set, test_set = prepare_semi_supervised_label_at_server_cifar10_datasets(train_val_test_split=train_val_test_split,
                                                                                                                num_labeled = args.num_labeled,
                                                                                                                num_clients = num_clients,
                                                                                                                is_balanced=not is_unbalance,
                                                                                                                seed = seed)
    val_save_path = join(save_dir, 'val.pkl')
    test_save_path = join(save_dir, 'test.pkl')
    
    save_pickle(val_set, val_save_path)
    save_pickle(test_set, test_save_path)
    save_pickle(server_train_s, join(save_dir, 'server_train_s.pkl'))
    clients_config = {}
    template = "fl_modules.dataset.cifar10_dataset.Cifar10Dataset"
    for client_id in client_train_u.keys():
        train_u_save_path = join(save_dir, f'client_{client_id}_train_u.pkl')
        
        save_pickle(client_train_u[client_id], train_u_save_path)
        
        config = {'dataset': {'train_u': {'template': template, 'params': {'data': train_u_save_path, 'targets': ['weak', 'strong'], 'batch_size': batch_size * 7, 'iters': iters}},
                            'val': {'template': template, 'params': {'data': val_save_path, 'batch_size': batch_size * 2}},
                            'test': {'template': template, 'params': {'data': test_save_path, 'batch_size': batch_size * 2}}}}
        
        clients_config[f'Client_{client_id}'] = config
    write_yaml(args.client_config_save_path, clients_config)