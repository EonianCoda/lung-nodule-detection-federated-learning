import datetime
import yaml
import logging
import shutil
import os
from os.path import splitext
from tqdm import tqdm
from importlib import import_module
from typing import Optional, Union, Dict, Any


def get_local_time_in_taiwan() -> datetime.datetime:
    utc_now = datetime.datetime.utcnow()
    taiwan_now = utc_now + datetime.timedelta(hours=8) # Taiwan in UTC+8
    return taiwan_now

def reset_working_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def gen_abs_dir(path:str) -> str:
    """Generate the absolute path for given path
    """
    path = os.path.abspath(path)
    path = os.path.dirname(path)
    return path

def load_yaml(path: str, 
            overwrite_config:Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
    """Load the yaml path
    """
    with open(path, 'r') as f:
        settings = yaml.safe_load(f)
    if overwrite_config != None:
        for key, value in overwrite_config:
            if settings.get(key) != None:
                settings[key] = value
    return settings

def write_yaml(save_path: str,
                config: dict,
                default_flow_style: str = ''):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style = default_flow_style)

def build_class(template: str) -> Any:
    """Build a classs based on give template
    """
    class_name = splitext(template)[1].strip('.')
    module_path = splitext(template)[0]
    module = import_module(module_path)
    cls = getattr(module, class_name)

    return cls

def build_instance(template: str, settings: Optional[Dict[str, Any]] = None) -> Any:
    """BUild a object based on given template and setting
    """
    if settings == None or len(settings) == 0:
        return build_class(template)()
    else:
        return build_class(template)(**settings)

def gen_log_level(level:str):
    level = level.lower()

    if level == 'debug':
        return logging.DEBUG
    elif level == 'info':
        return logging.INFO
    elif level == 'warning':
        return logging.WARNING
    else:
        raise ValueError("Should given valid log level")
    
def get_progress_bar(identifer: str, total_steps: int) -> tqdm:
    """Get the progress bar
    """
    progress_bar = tqdm(total = total_steps, 
                        desc = "{:10s}".format(identifer), 
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    return progress_bar

def init_seed(seed: int):
    import torch
    import numpy as np
    import random
    
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)