import os
import numpy as np

from typing import List, Union, Dict

class NoduleMetrics:
    def __init__(self, nodule_metrics: Dict[str, int]):
        self.nodule_metrics = nodule_metrics
        if '>4mm' not in self.nodule_metrics:
            self._add_4mm()
    
    def _add_4mm(self):
        metric_4mm = np.zeros((4,), dtype=np.int32)
        for nodule_type, metric in self.nodule_metrics.items():
            if nodule_type in ['probably_benign', 'probably_suspicious', 'suspicious']:
                metric_4mm += np.array([metric['tp'], metric['fp'], metric['fn'], metric['tn']])
        
        tp, fp, fn, tn = metric_4mm
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        if recall + precision <= 0:
            f1_score = 0
        else:
            f1_score = 2 * ((recall * precision) / (recall + precision))
        if tp + fn + fp + tn <= 0:
            accuracy = 0
        else:
            accuracy = (tp + tn) / max(tp + fn + fp + tn, 1)
        metrics_4mm = {'recall': recall, 
                        'precision': precision, 
                        'f1_score': f1_score, 
                        'accuracy': accuracy,
                        'tp': tp,
                        'fp': fp,
                        'fn': fn,
                        'tn': tn}
        self.nodule_metrics['>4mm'] = metrics_4mm
    
    def write_metric_csv(self, params: Dict[str, Union[str, int, float]], save_path: str):
        if len(self.nodule_metrics) != 0:
            params_header = ','.join(params.keys())
            params_values = ','.join([str(value) for value in params.values()])
            lines = [params_header, params_values]
        else:
            lines = []
        lines.extend(self.generate_metric_csv_lines())
        with open(save_path, 'a') as f:
            for i, line in enumerate(lines):
                f.write(line + '\n')
            f.write('\n')
    def generate_metric_csv_lines(self) -> List[str]:
        sorted_nodule_size_ranges = ['benign', 'probably_benign', 'probably_suspicious', 'suspicious', '>4mm', 'all']
        sorted_metrics = ['recall', 'precision', 'f1_score']
        sorted_counts = ['tp', 'fp', 'fn', 'tn']
        
        header = 'Nodule_type,'
        header += ','.join([m + '(%)' for m in sorted_metrics]) + ','
        header += ','.join(sorted_counts)
        
        lines = [header]
        for nodule_type in sorted_nodule_size_ranges:
            line = nodule_type
            for metric_name in sorted_metrics:
                line += ',{:.1f}'.format(self.nodule_metrics[nodule_type][metric_name] * 100)
            for metric_name in sorted_counts:
                line += ',{}'.format(self.nodule_metrics[nodule_type][metric_name])
            lines.append(line)
        
        return lines