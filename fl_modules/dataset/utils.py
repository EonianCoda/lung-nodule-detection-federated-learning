import os
import logging
import torchvision
import numpy as np
import random
import pickle
from PIL import Image
from numpy.typing import NDArray
from typing import Tuple, Dict, List, Union
from collections import defaultdict

SAVE_ROOT = './data'
CIFAR10_PATH = os.path.join(SAVE_ROOT, 'cifar10.pkl')

logger = logging.getLogger(__name__)

def save_pickle(obj, save_path: str):
    with open(save_path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(load_path: str):
    with open(load_path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def load_cifar10() -> Tuple[NDArray[np.uint8], NDArray[np.int32]]:
    """
    Returns: 
        A Tuple of (x, y)
        Below is the shape of each element in the tuple:
        x: (60000, 32, 32, 3)
        y: (60000,)
    """
    if not os.path.exists(CIFAR10_PATH):
        train_set = torchvision.datasets.CIFAR10(root=SAVE_ROOT, train=True, transform=None, download=True)
        test_set = torchvision.datasets.CIFAR10(root=SAVE_ROOT, train=False, transform=None, download=True)
        
        x, y = [], []
        x.append(train_set.data)
        x.append(test_set.data)
        x = np.concatenate(x, axis = 0)
        x_pillow = []
        for i in range(x.shape[0]):
            x_pillow.append(Image.fromarray(x[i]))
        
        y.extend(train_set.targets)
        y.extend(test_set.targets)
        y = np.array(y, dtype=np.int32)
        
        save_pickle({'x': x_pillow, 'y': y}, CIFAR10_PATH)
    
    data = load_pickle(CIFAR10_PATH)
    
    return data['x'], data['y']

def split_data(x: List[Image.Image], 
               y: NDArray[np.int32], 
               num_datas: List[int],
               is_balanced: bool = True,
               seed: int = 0) -> Dict[int, Dict[str, Union[List[Image.Image],np.ndarray]]]:
    """
    Args:
        split_ratio: A list of float numbers that sum to 1.0
        dist: A n x m matrix, where n is the number of categories and m is the number of splits. The sum of each row must be 1.0.
    """
    # Split into one set
    if len(num_datas) == 1:
        splited_dataset = dict()
        splited_dataset[0] = {'x': x[:num_datas[0]], 'y': y[:num_datas[0]]}
        return splited_dataset
    
    categories = defaultdict(list)
    for i, label in enumerate(y):
        categories[label].append(i)
        
    random.seed(seed)
    # Split uniformly
    if is_balanced:
        splited_dataset = dict()
        for split_i in range(len(num_datas)):
            splited_dataset[split_i] = {'x': [], 'y': []}
        num_datas_of_each_class = []
        for label, indices in categories.items():
            num_datas_of_each_class.append([])
            for split_i in range(len(num_datas)):
                num_datas_of_each_class[-1].append(num_datas[split_i] // len(categories))
                
            num_datas_of_each_class[-1][-1] = len(categories[label]) - sum(num_datas_of_each_class[-1][:-1])
            
        for label, indices in categories.items():
            random.shuffle(indices)
            for split_i in range(len(num_datas)):
                start_i = sum(num_datas_of_each_class[label][:split_i])
                end_i = sum(num_datas_of_each_class[label][:split_i + 1])
                splited_dataset[split_i]['x'].extend([x[idx] for idx in indices[start_i: end_i]])
                splited_dataset[split_i]['y'].extend([y[idx] for idx in indices[start_i: end_i]])
        for split_i in range(len(num_datas)):
            splited_dataset[split_i]['y'] = np.array(splited_dataset[split_i]['y'], dtype=np.int32)
    else:
        splited_dataset = dict()
        for split_i in range(len(num_datas)):
            splited_dataset[split_i] = {'x': [], 'y': []}
        ten_types_of_class_imbalanced_dist = [
                [0.50,0.15,0.03,0.03,0.03,0.02,0.03,0.03,0.03,0.15], # type 0
                [0.15,0.50,0.15,0.03,0.03,0.03,0.02,0.03,0.03,0.03], # type 1 
                [0.03,0.15,0.50,0.15,0.03,0.03,0.03,0.02,0.03,0.03], # type 2 
                [0.03,0.03,0.15,0.50,0.15,0.03,0.03,0.03,0.02,0.03], # type 3 
                [0.03,0.03,0.03,0.15,0.50,0.15,0.03,0.03,0.03,0.02], # type 4 
                [0.02,0.03,0.03,0.03,0.15,0.50,0.15,0.03,0.03,0.03], # type 5 
                [0.03,0.02,0.03,0.03,0.03,0.15,0.50,0.15,0.03,0.03], # type 6 
                [0.03,0.03,0.02,0.03,0.03,0.03,0.15,0.50,0.15,0.03], # type 7 
                [0.03,0.03,0.03,0.02,0.03,0.03,0.03,0.15,0.50,0.15], # type 8 
                [0.15,0.03,0.03,0.03,0.02,0.03,0.03,0.03,0.15,0.50], # type 9
        ]
        labels = list(categories.keys())
        offset_per_label = {label: 0 for label in labels}
        
        for client_id, num_u in enumerate(num_datas):
            x_unlabeled = []
            y_unlabeled = []
            dist_type = client_id % len(labels)
            freqs = np.random.choice(labels, num_u, p=ten_types_of_class_imbalanced_dist[dist_type])
            for label, indices in categories.items():
                num_instances = len(freqs[freqs == label])
                start = offset_per_label[label]
                end = offset_per_label[label] + num_instances
                x_unlabeled.extend([x[idx] for idx in categories[label][start:end]])
                y_unlabeled.extend([y[idx] for idx in categories[label][start:end]])
                offset_per_label[label] = end
            splited_dataset[client_id]['x'] = x_unlabeled
            splited_dataset[client_id]['y'] = np.array(y_unlabeled, dtype=np.int32)
    return splited_dataset

def prepare_supervised_cifar10_datasets(train_val_test_split: List[float],
                                        num_clients: int = 10,
                                        is_balanced: bool = True,
                                        seed: int = 0) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Returns:
        A tuple of (client_train_data, val_set, test_set)
        client_train_data:
            A dictionary of client's train set. The key is the client id and the value is a dictionary of x and y.
        val_set:
            A dictionary of validation set. The key is 'x' and 'y'.
        test_set:
            A dictionary of test set. The key is 'x' and 'y'.
    """
    x, y = load_cifar10()
    # Split train/val/test set uniformly
    num_datas = []
    for ratio in train_val_test_split:
        num_datas.append(int(len(x) * ratio))
    num_datas[-1] = len(x) - sum(num_datas[:-1])
    
    data = split_data(x, y, num_datas, is_balanced=True, seed=seed)
    train_data, val_data, test_data = data[0], data[1], data[2]
    print("There are {} train data, {} val data, and {} test data.".format(len(train_data['y']), len(val_data['y']), len(test_data['y'])))
    
    # Split train set into supervised and unsupervised set
    num_datas = [len(train_data['x']) // num_clients] * num_clients
    num_datas[-1] = len(train_data['x']) - sum(num_datas[:-1])
    client_train_data = split_data(train_data['x'], train_data['y'], num_datas, is_balanced=is_balanced, seed=seed)
    for client_id, data in client_train_data.items():
        print('Client {:2d} has {:5d} labeled data for class {}.'.format(client_id, len(data['y']), np.unique(data['y'])))
    return client_train_data, val_data, test_data

def prepare_semi_supervised_cifar10_datasets(train_val_test_split: List[float],
                                            num_labeled: int,
                                            num_clients: int = 10,
                                            is_balanced: bool = True,
                                            seed: int = 0) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict[int, Dict[str, np.ndarray]], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Returns:
        A tuple of (client_train_s, client_train_u, val_set, test_set)
        client_train_s:
            A dictionary of client's train set. The key is the client id and the value is a dictionary of x and y.
        client_train_u:
            A dictionary of client's train set. The key is the client id and the value is a dictionary of x and y.
        val_set:
            A dictionary of validation set. The key is 'x' and 'y'.
        test_set:
            A dictionary of test set. The key is 'x' and 'y'.
    """
    x, y = load_cifar10()
    # Split train/val/test set uniformly
    num_datas = []
    for ratio in train_val_test_split:
        num_datas.append(int(len(x) * ratio))
    num_datas[-1] = len(x) - sum(num_datas[:-1])
    
    data = split_data(x, y, num_datas, is_balanced=True, seed=seed)
    train_data, val_data, test_data = data[0], data[1], data[2]
    print("There are {} train data, {} val data, and {} test data.".format(len(train_data['y']), len(val_data['y']), len(test_data['y'])))
    
    # Split train set into supervised and unsupervised set
    num_datas = [num_labeled * num_clients, (len(x) - num_labeled * num_clients)]
    data = split_data(train_data['x'], train_data['y'], num_datas, is_balanced=True, seed=seed)
    train_s, train_u = data[0], data[1]
    print("There are {} labeled data and {} unlabeled data for training.".format(len(train_s['y']), len(train_u['y'])))
    # Split train set into clients
    # Supervised set
    num_datas = [num_labeled] * num_clients
    client_train_s = split_data(train_s['x'], train_s['y'], num_datas, is_balanced=True, seed=seed)
    # Unsupervised set
    num_datas = [(len(train_u['x']) // num_clients)] * num_clients
    num_datas[-1] = len(train_u['x']) - sum(num_datas[:-1])
    client_train_u = split_data(train_u['x'], train_u['y'], num_datas, is_balanced=is_balanced, seed=seed)
    
    for (client_id, data_s), (_, data_u) in zip(client_train_s.items(), client_train_u.items()):
        print('Client {:2d} has {:5d} labeled data and {:5d} unlabeled data for class {}.'.format(client_id, len(data_s['y']), len(data_u['y']), np.unique(data_u['y'])))
    return client_train_s, client_train_u, val_data, test_data