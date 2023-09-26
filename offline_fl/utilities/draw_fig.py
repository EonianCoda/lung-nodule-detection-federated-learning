from os.path import join
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

class MetricDrawer:
    def __init__(self, event_file_folder: str):
        self.event = EventAccumulator(event_file_folder)
        self.event.Reload()

    def save_figure(self, save_folder: str):
        client_names = set()
        val_metrics = set()
        train_metrics = set()
        
        for tag in self.event.Tags()['scalars']:
            client_name = tag.split('/')[0]
            task = tag.split('/')[1]
            metric = tag.split('/')[2]
            
            if client_name != 'Server':
                client_names.add(client_name)
            
            if task == 'val_global':
                if metric not in ['tp', 'fp', 'fn', 'tn']:
                    val_metrics.add(metric)
            elif task == 'train':
                train_metrics.add(metric)

        client_names = list(client_names)
        val_metrics = list(val_metrics)

        # Get train metric values
        client_train_metric_values = dict()
        for metric in train_metrics:
            for client_name in client_names:
                tag = '{}/{}/{}'.format(client_name, 'train', metric)
                values = np.array([v.value for v in self.event.Scalars(tag)])
                if client_train_metric_values.get(client_name) is None:
                    client_train_metric_values[client_name] = dict()
                client_train_metric_values[client_name][metric] = values
        # Get validation metric values
        server_target_task = 'val_global'
        client_target_task = 'val_local'
        client_validation_metric_values = dict()
        server_validation_metric_values = dict()
        for metric in val_metrics:
            # Server
            tag = '{}/{}/{}'.format('Server', server_target_task, metric)
            values = np.array([v.value for v in self.event.Scalars(tag)]) * 100 # * 100 for percentage
            server_validation_metric_values[metric] = values
            # Client
            for client_name in client_names:
                tag = '{}/{}/{}'.format(client_name, client_target_task, metric)
                values = np.array([v.value for v in self.event.Scalars(tag)]) * 100 # * 100 for percentage
                if client_validation_metric_values.get(client_name) is None:
                    client_validation_metric_values[client_name] = dict()
                client_validation_metric_values[client_name][metric] = values

        cmap = plt.get_cmap('tab10')

        # Draw train metrics
        col = 1
        row = len(train_metrics)
        plt.figure(figsize=(row * 8, 6), tight_layout=True)
        for metric_i, (metric, value) in enumerate(client_train_metric_values[client_names[0]].items()):
            ax = plt.subplot(col, row, metric_i + 1)
            ax.set_title('Training Performance - {}'.format(metric.replace('_', '-')))
            ax.set_xlabel('Round')
            ax.set_ylabel(metric)
            
            for client_i, (client_name, values) in enumerate(client_train_metric_values.items()):
                color = cmap(client_i)
                ax.plot(list(range(len(values[metric]))), values[metric], label=client_name, linestyle='dotted', color=color, marker='o', markersize=3)
            
            ax.set_xlim(-1, len(value) + 1)
            
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.legend(loc='upper right')
        plt.savefig(join(save_folder, 'train.png'), dpi=300)
        # Draw validation metrics
        col = 1
        row = len(val_metrics)
        plt.figure(figsize=(row * 8, 6), tight_layout=True)
        for fig_i, (metric, server_values) in enumerate(server_validation_metric_values.items()):
            ax = plt.subplot(col, row, fig_i + 1)
            ax.set_title('Aggregation Performance - {}'.format(metric))
            ax.set_xlabel('Round')
            ax.set_ylabel(metric + '(%)')

            for client_i, (client_name, values) in enumerate(client_validation_metric_values.items()):
                color = cmap(client_i)
                ax.plot(list(range(len(values[metric]))), values[metric], label=client_name, linestyle='dotted', color=color, marker='o', markersize=3)
            
            ax.set_xlim(-1, len(server_values) + 1)
            ax.set_ylim(-5, 105)
            
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.plot(list(range(len(server_values))), server_values, label='Server', color=cmap(len(client_names)), marker='o', markersize=4)
            
            ax.legend(loc='upper right')
        plt.savefig(join(save_folder, 'val.png'), dpi=300)