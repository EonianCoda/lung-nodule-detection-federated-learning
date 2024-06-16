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
    parser.add_argument('--series_txt_path', type=str, required=True)
    parser.add_argument('--n_clusters', type=int, default=20)
    parser.add_argument('--experiment_config_path', type=str, default='./config/stage1_fedavg.yaml')
    parser.add_argument('--label_ratio', type=float, default=0.1)
    parser.add_argument('--save_root', type=str, default='./data/new_unlabeled')
    parser.add_argument('--save_name', type=str, required=True)
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
    arg_sorted_ratios = arg_sorted_ratios[::-1] # descending order
    
    for cluster_i in cluster_indices:
        data_idx = np.where(cluster_labels == cluster_i)[0]
        len_data_idx = len(data_idx)
        
        n_samples_per_client = [max(int(len_data_idx * ratio), 1) for ratio in ratios]
        
        for split_i in range(len(ratios)):
            if len(data_idx) == 0:
                break
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

def get_nodule_counts_info(feat_of_series: np.ndarray, indices: List[int], sorted_nodule_size_ranges) -> str:
    info = ''
    feat = np.zeros_like(feat_of_series[0])
    for idx in indices:
        feat += feat_of_series[idx]
    
    all_counts = np.sum(feat)
    
    for nodule_type, num in zip(sorted_nodule_size_ranges, feat):
        info += '{:19s}: {:5d}\n'.format(nodule_type, num)
    info += '{:19s}: {:5d}\n'.format('All', all_counts)
    info += '\n'
    return info

def get_dist_of_split(feat_of_series: np.ndarray, indices: List[int]) -> np.ndarray:
    feat = np.zeros_like(feat_of_series[0])
    for idx in indices:
        feat += feat_of_series[idx]
    
    feat = feat / np.sum(feat)
    return feat
    
if __name__ == '__main__':
    args = get_args()
    series_txt_path = args.series_txt_path
    n_clusters = args.n_clusters
    experiment_config_path = args.experiment_config_path
    save_root = args.save_root
    seed = args.seed
    
    experiment_config = load_yaml(experiment_config_path)
    series_list = load_series_list(series_txt_path)
    nodule_size_ranges = experiment_config['client']['nodule_size_ranges']
    
    nodule_counter = NoduleCounter()
    num_nodule_of_series = nodule_counter.count_and_analyze_nodules_of_multi_series(series_txt_path, nodule_size_ranges, mode='single')

    # Use number of different nodule tpye of series to build feature of series 
    sorted_nodule_size_ranges = sorted(nodule_size_ranges.keys())
    feat_of_series = []
    for num_nodule in num_nodule_of_series:
        feat_of_series.append(np.array([num_nodule[nodule_type] for nodule_type in sorted_nodule_size_ranges]))
    feat_of_series = np.stack(feat_of_series, axis=0)
    # Normalize feature = feat - mean / std
    normalized_feat_of_series = feat_of_series.copy()
    normalized_feat_of_series = (feat_of_series - np.mean(feat_of_series, axis=0)) / np.std(feat_of_series, axis=0)

    # Split unlabeled and labeled data
    unlabel_samples, label_samples = split_data_by_ratio(normalized_feat_of_series, n_clusters, [1 - args.label_ratio, args.label_ratio], seed)

    # Save clients samples
    os.makedirs(save_root, exist_ok=True)
    
    header = 'Folder,Filename\n'
    # Save pretrained train/val samples
    modes = ['unlabeled', 'labeled']
    samples_list = [unlabel_samples, label_samples]
    for mode, samples in zip(modes, samples_list):
        save_path = f'{save_root}/{args.save_name}_{mode}_train.txt'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(header)
            for idx in samples:
                f.write(f'{series_list[idx][0]},{series_list[idx][1]}\n')