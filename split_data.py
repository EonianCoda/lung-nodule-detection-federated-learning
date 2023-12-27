from fl_modules.inference.nodule_counter import NoduleCounter
from fl_modules.utilities import load_yaml
from fl_modules.dataset.utils import load_series_list
from sklearn.cluster import KMeans
from typing import List

import shutil
import numpy as np
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--series_txt_path', type=str)
    parser.add_argument('--n_clusters', type=int, default=10)
    parser.add_argument('--n_clients', type=int, default=3)
    parser.add_argument('--experiment_config_path', type=str, default='./config/stage1_fedavg.yaml')
    parser.add_argument('--save_root', type=str, default='./data/lung_nodule')
    parser.add_argument('--use_pretrained', action='store_true', default=False)
    parser.add_argument('--pretrained_ratio', type=float, default=0.1)
    parser.add_argument('--pretrained_train_val_ratios', nargs='+', type=float, default=[0.95, 0.05])
    parser.add_argument('--train_val_test_ratios', nargs='+', type=float, default=[0.9, 0.05, 0.05])
    parser.add_argument('--use_unlabel', action='store_true', default=False)
    parser.add_argument('--unlabeled_ratio', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=1029)
    args = parser.parse_args()
    return args

def split_data_by_ratio(feats: np.ndarray, 
                        n_clusters: int, 
                        ratios: List[float],
                        seed: int,
                        balanced_numbers: bool = True) -> List[np.ndarray]:
    """
    Args:
        feats: Feature of data, which is used to cluster data via kmeans
        n_clusters: Number of clusters of kmeans
        ratios: List of ratios of each splitted data
        seed: Random seed
        balanced_numbers: If True, add remained data to clients with less data, otherwise add remained data to clients with larger ratio
    Returns: List[np.ndarray]
        List of splitted data indices
    """
    # Use kmeans to group data, and then select data from each group evenly
    kmeans = KMeans(n_clusters = n_clusters, random_state = seed, n_init = 'auto')
    kmeans.fit(feats)

    cluster_labels = kmeans.labels_
    cluster_indices = np.arange(n_clusters)
    
    splitted_samples = [list() for _ in range(len(ratios))]
    
    arg_sorted_ratios = np.argsort(ratios)
    
    for cluster_i in cluster_indices:
        data_idx = np.where(cluster_labels == cluster_i)[0]
        len_data_idx = len(data_idx)
        
        n_samples_per_client = [int(len_data_idx * ratio) for ratio in ratios]
        for split_i in range(len(ratios)):
            splitted_samples[split_i].extend(data_idx[:n_samples_per_client[split_i]].tolist())
            data_idx = data_idx[n_samples_per_client[split_i]:]
        
        remained_num_data = len(data_idx)
        if not balanced_numbers: # add remained data to clients with larger ratio
            for i in range(remained_num_data):
                splitted_samples[arg_sorted_ratios[i]].append(data_idx[i])
        else: # add remained data to clients with less data
            num_splitted_samples = [len(samples) for samples in splitted_samples]
            arg_sorted_num_splitted_samples = np.argsort(num_splitted_samples)
            for i in range(remained_num_data):
                splitted_samples[arg_sorted_num_splitted_samples[i]].append(data_idx[i])
    
    splitted_samples = [np.array(samples) for samples in splitted_samples]
    return splitted_samples

def print_nodules(feat_of_series: np.ndarray , indices: List[int], sorted_nodule_size_ranges, save_file: str = None):
    feat = np.zeros_like(feat_of_series[0])
    for idx in indices:
        feat += feat_of_series[idx]
        
    if save_file is not None:
        f = open(save_file, 'a')
    else:
        f = None
        
    for nodule_type, num in zip(sorted_nodule_size_ranges, feat):
        print('{:19s}: {:5d}'.format(nodule_type, num), file=f)
    print('', file=f)
    
    if f is not None:
        f.close()

if __name__ == '__main__':
    args = get_args()
    series_txt_path = args.series_txt_path
    n_clusters = args.n_clusters
    n_clients = args.n_clients
    experiment_config_path = args.experiment_config_path
    train_val_test_ratios = args.train_val_test_ratios
    use_pretrained = args.use_pretrained
    pretrained_ratio = args.pretrained_ratio
    pretrained_train_val_ratios = args.pretrained_train_val_ratios
    use_unlabel = args.use_unlabel
    unlabeled_ratio = args.unlabeled_ratio
    seed = args.seed
    
    experiment_config = load_yaml(experiment_config_path)
    series_list = load_series_list(series_txt_path)
    nodule_size_ranges = experiment_config['client']['nodule_size_ranges']
    
    nodule_counter = NoduleCounter()
    num_nodule_of_series = nodule_counter.count_and_analyze_nodules_of_multi_series(series_txt_path, nodule_size_ranges, mode='single')
    n_samples = len(num_nodule_of_series)

    # Use number of different nodule tpye of series to build feature of series 
    sorted_nodule_size_ranges = sorted(nodule_size_ranges.keys())
    feat_of_series = []
    for num_nodule in num_nodule_of_series:
        feat_of_series.append(np.array([num_nodule[nodule_type] for nodule_type in sorted_nodule_size_ranges]))
    feat_of_series = np.stack(feat_of_series, axis=0)

    # Split pretrained samples
    if use_pretrained:
        pretrained_samples, non_pretrained_samples = split_data_by_ratio(feat_of_series, n_clusters, [pretrained_ratio, 1 - pretrained_ratio], seed)
        # Split train/val
        feats = feat_of_series[pretrained_samples]
        train_pretrained_samples, val_pretrained_samples = split_data_by_ratio(feats, n_clusters // 2, pretrained_train_val_ratios, seed)
        pretrained_train_samples = pretrained_samples[train_pretrained_samples]
        pretrained_val_samples = pretrained_samples[val_pretrained_samples]
        
    # Split clients samples
    normal_ratios = [1 / n_clients for _ in range(n_clients)]
    if use_pretrained:
        feats = feat_of_series[non_pretrained_samples]
        clients_samples = split_data_by_ratio(feats, n_clusters, normal_ratios, seed)
        for i in range(n_clients):
            clients_samples[i] = non_pretrained_samples[clients_samples[i]]
    else:
        clients_samples = split_data_by_ratio(feat_of_series, n_clusters, normal_ratios, seed)
    
    # Split train/val/test
    train_clients_samples = []
    val_clients_samples = []
    test_clients_samples = []
    for i in range(n_clients):
        feats = feat_of_series[clients_samples[i]]
        train_samples, val_samples, test_samples = split_data_by_ratio(feats, n_clusters // 2, train_val_test_ratios, seed)
        
        train_clients_samples.append(clients_samples[i][train_samples])
        val_clients_samples.append(clients_samples[i][val_samples])
        test_clients_samples.append(clients_samples[i][test_samples])

    # Split unlabeled data
    if use_unlabel:
        unlabeled_train_clients_samples = []
        for i in range(n_clients):
            feats = feat_of_series[train_clients_samples[i]]
            unlabeled_train_samples, labeled_train_samples = split_data_by_ratio(feats, n_clusters // 2, [unlabeled_ratio, 1 - unlabeled_ratio], seed)
            
            unlabeled_train_clients_samples.append(train_clients_samples[i][unlabeled_train_samples])
            train_clients_samples[i] = train_clients_samples[i][labeled_train_samples]

    # Calculate number of different nodule type of each client
    if use_pretrained:
        print('Pretrained, number of samples: {}'.format(len(pretrained_samples)))
        print_nodules(feat_of_series, pretrained_samples, sorted_nodule_size_ranges)
        print('Non-pretrained, number of samples: {}'.format(len(non_pretrained_samples)))
        print_nodules(feat_of_series, non_pretrained_samples, sorted_nodule_size_ranges)
        
        print('Pretrained train, number of samples: {}'.format(len(pretrained_train_samples)))
        print_nodules(feat_of_series, pretrained_train_samples, sorted_nodule_size_ranges)
        print('Pretrained val, number of samples: {}'.format(len(pretrained_val_samples)))
        print_nodules(feat_of_series, pretrained_val_samples, sorted_nodule_size_ranges)
        print('-' * 50)
    for i in range(n_clients):
        print(f'Client{i}, number of samples: {len(clients_samples[i])}')
        if use_unlabel:
            print('Unlabeled train, number of samples: {}'.format(len(unlabeled_train_clients_samples[i])))
            print_nodules(feat_of_series, unlabeled_train_clients_samples[i], sorted_nodule_size_ranges)
        
        print('Labeled train, number of samples: {}'.format(len(train_clients_samples[i])))
        print_nodules(feat_of_series, train_clients_samples[i], sorted_nodule_size_ranges)
        
        print('Val, number of samples: {}'.format(len(val_clients_samples[i])))
        print_nodules(feat_of_series, val_clients_samples[i], sorted_nodule_size_ranges)
        print('Test, number of samples: {}'.format(len(test_clients_samples[i])))
        print_nodules(feat_of_series, test_clients_samples[i], sorted_nodule_size_ranges)
        print('-' * 50)
    # Save clients samples
    if os.path.exists(args.save_root):
        shutil.rmtree(args.save_root)
    header = 'Folder,Filename\n'
    
    # Save pretrained samples
    if use_pretrained:
        save_path = f'{args.save_root}/pretrained.txt'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(header)
            for idx in pretrained_samples:
                f.write(f'{series_list[idx][0]},{series_list[idx][1]}\n')
        
        # Save train/val samples
        modes = ['train', 'val']
        samples_list = [pretrained_train_samples, pretrained_val_samples]
        for mode, samples in zip(modes, samples_list):
            save_path = f'{args.save_root}/pretrained_{mode}.txt'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(header)
                for idx in samples:
                    f.write(f'{series_list[idx][0]},{series_list[idx][1]}\n')
                    
    for i in range(n_clients):
        # Save all samples
        save_path = f'{args.save_root}/client{i}.txt'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(header)
            for idx in clients_samples[i]:
                f.write(f'{series_list[idx][0]},{series_list[idx][1]}\n')
        
        # Save train/val/test samples
        modes = ['train', 'val', 'test']
        samples_list = [train_clients_samples[i], val_clients_samples[i], test_clients_samples[i]]
        if use_unlabel:
            modes.append('unlabeled_train')
            samples_list.append(unlabeled_train_clients_samples[i])
        for mode, samples in zip(modes, samples_list):
            save_path = f'{args.save_root}/client{i}_{mode}.txt'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(header)
                for idx in samples:
                    f.write(f'{series_list[idx][0]},{series_list[idx][1]}\n')