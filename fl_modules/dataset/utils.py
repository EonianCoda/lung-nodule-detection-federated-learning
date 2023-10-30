import os
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
               split_ratios: List[float],
               dist: List[List[float]] = None,
               seed: int = 0) -> Dict[int, Dict[str, Union[List[Image.Image],np.ndarray]]]:
    """
    Args:
        split_ratio: A list of float numbers that sum to 1.0
        dist: A n x m matrix, where n is the number of categories and m is the number of splits. The sum of each row must be 1.0.
    """
    
    # Check distribution
    if dist != None:
        if len(split_ratios) != len(dist):
            raise ValueError('The length of split_ratio and distribution must be the same.')
        sum_of_each_row = np.sum(dist, axis=1)
        if np.any(np.abs(sum_of_each_row - 1.0) > 1e-6):
            raise ValueError('The sum of each row of distribution must be 1.0.')
    
    # Split into one set
    if len(split_ratios) == 1:
        splited_dataset = dict()
        splited_dataset[0] = {'x': x, 'y': y}
        return splited_dataset
    
    categories = defaultdict(list)
    for i, label in enumerate(y):
        categories[label].append(i)
        
    if dist != None and len(categories) != len(dist[0]):
        raise ValueError('The length of distribution must be the same as the number of categories.')
    
    random.seed(seed)
    # Split uniformly
    if dist == None:
        splited_dataset = dict()
        for split_i in range(len(split_ratios)):
            splited_dataset[split_i] = {'x': [], 'y': []}
        
        for label, indices in categories.items():
            random.shuffle(indices)
            for split_i in range(len(split_ratios)):
                start_i = int(len(indices) * sum(split_ratios[:split_i]))
                if split_i != len(split_ratios) - 1:
                    end_i = int(len(indices) * sum(split_ratios[:split_i+1]))
                else:
                    end_i = len(indices)
                    
                for idx in indices[start_i: end_i]:
                    splited_dataset[split_i]['x'].append(x[idx])
                    splited_dataset[split_i]['y'].append(y[idx])
        for split_i in range(len(split_ratios)):
            splited_dataset[split_i]['y'] = np.array(splited_dataset[split_i]['y'], dtype=np.int32)
    # Split according to distribution
    else:
        splited_dataset = dict()
        for split_i in range(len(split_ratios)):
            splited_dataset[split_i] = {'x': [], 'y': []}
        
        for label_i, (label, indices) in enumerate(categories.items()):
            random.shuffle(indices)
            
            label_dist = dist[label_i]
            for split_i in range(len(split_ratios)):
                start_i = int(len(indices) * sum(label_dist[:split_i]))

                if split_i != len(split_ratios) - 1:
                    end_i = int(len(indices) * sum(label_dist[:split_i+1]))
                else:
                    end_i = len(indices)
                    
                for idx in indices[start_i: end_i]:
                    splited_dataset[split_i]['x'].append(x[idx])
                    splited_dataset[split_i]['y'].append(y[idx])
        for split_i in range(len(split_ratios)):
            splited_dataset[split_i]['y'] = np.array(splited_dataset[split_i]['y'], dtype=np.int32)
    return splited_dataset
    
def prepare_cifar10_datasets(train_val_test_split: List[float] = [0.8, 0.1, 0.1],
                            s_u_split: List[float] = [0.1, 0.9],
                            num_clients: int = 10,
                            is_balance: bool = True,
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
    data = split_data(x, y, train_val_test_split, seed=seed)
    train_data, val_data, test_data = data[0], data[1], data[2]
    
    # Split train set into supervised and unsupervised set
    data = split_data(train_data['x'], train_data['y'], s_u_split, seed=seed)
    train_s, train_u = data[0], data[1]
    
    # Split train set into clients
    if is_balance:
        dist = None
    else:
        if num_clients != 10:
            raise ValueError('num_clients must be 10 when is_balanced is False.')
        dist = [
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
    ratios = [1 / num_clients] * num_clients
    # Supervised set
    client_train_s = split_data(train_s['x'], train_s['y'], ratios, dist=dist, seed=seed)
    # Unsupervised set
    client_train_u = split_data(train_u['x'], train_u['y'], ratios, dist=dist, seed=seed)
    
    return client_train_s, client_train_u, val_data, test_data