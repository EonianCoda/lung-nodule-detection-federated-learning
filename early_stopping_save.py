import os
import logging
from typing import Dict, List
from fl_modules.client.utils import save_states
logger = logging.getLogger(__name__)

class EarlyStoppingSave:
    def __init__(self, target_metrics: List[str], save_dir: str, model, best: List[float] = None, best_epoch: List[int] = None):
        self.target_metrics = target_metrics
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.best = best if best else [0.0 for _ in target_metrics]
        self.best_epoch = best_epoch if best_epoch else [0 for _ in target_metrics]
        self.model = model
        
        self.best_model_save_paths = [os.path.join(self.save_dir, 'best_{}.pth'.format(target_metric)) for target_metric in self.target_metrics]
        self.best_txt_save_paths = [os.path.join(self.save_dir, 'best_{}.txt'.format(target_metric)) for target_metric in self.target_metrics]
    
    def step(self, metrics: Dict[str, float], epoch: int):
        for i, target_metric in enumerate(self.target_metrics):
            if metrics[target_metric] > self.best[i]:
                self.best[i] = metrics[target_metric]
                self.best_epoch[i] = epoch
                self._save_txt(metrics, target_metric)
                self._save_model(target_metric)
                logger.info('====> Best model saved at epoch: {} with {} {:.4f}'.format(epoch, target_metric, self.best[i]))

    def _save_model(self, target_metric: str):
        save_states(self.best_model_save_paths[self.target_metrics.index(target_metric)], self.model)
    
    def _save_txt(self, metrics: Dict[str, float], target_metric: str):
        target_idx = self.target_metrics.index(target_metric)
        with open(self.best_txt_save_paths[target_idx], 'w') as f:
            f.write('Epoch: {}\n'.format(self.best_epoch[target_idx]))
            f.write('Metric: {}\n'.format(target_metric))
            f.write('Value: {:.4f}\n'.format(self.best[target_idx]))
            f.write('-'*20 + '\n')
            
            max_length = max([len(key) for key in metrics.keys()])
            for key, value in metrics.items():
                f.write('{}: {:.4f}\n'.format(key.ljust(max_length), value))

    def get_best_model_paths(self) -> Dict[str, str]:
        rs = dict()
        for target_metric in self.target_metrics:
            rs[target_metric] = self.best_model_save_paths[self.target_metrics.index(target_metric)]
        return rs
    
    @staticmethod
    def load(save_dir: str, target_metrics: List[str], model):
        best = [0.0 for _ in target_metrics]
        best_epoch = [0 for _ in target_metrics]
        
        for target_metric in target_metrics:
            txt_path = os.path.join(save_dir, 'best_{}.txt'.format(target_metric))
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    best_epoch[target_metrics.index(target_metric)] = int(f.readline().split(':')[-1])
                    metric = f.readline().split(':')[-1].strip()
                    if metric != target_metric:
                        raise ValueError('Target metric {} does not match with the metric in the file {}'.format(target_metric, txt_path))
                    best[target_metrics.index(target_metric)] = float(f.readline().split(':')[-1])
                logger.info('====> Best metric {} found at epoch: {} with value: {:.4f}'.format(target_metric, best_epoch[target_metrics.index(target_metric)], best[target_metrics.index(target_metric)]))
        return EarlyStoppingSave(target_metrics, save_dir, model, best, best_epoch)