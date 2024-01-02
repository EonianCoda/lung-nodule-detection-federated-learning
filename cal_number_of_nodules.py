import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from fl_modules.inference.nodule_counter import NoduleCounter
from fl_modules.utilities import load_yaml
from fl_modules.dataset.utils import load_series_list
import argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config_path', type=str, default='./config/data_config.yaml')
    parser.add_argument('--co_config_path', type=str, default='./config/co_config.yaml')
    parser.add_argument('--series_list', type=str, required=True)
    parser.add_argument('--print_each_series', action='store_true', default=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    series_list = args.series_list
    data_config_path = args.data_config_path
    co_config_path = args.co_config_path
    print_each_series = args.print_each_series
    
    co_config = load_yaml(co_config_path)
    data_config = load_yaml(data_config_path)
    
    series_list_path = os.path.join(data_config['root'], series_list)
    series_infos = load_series_list(series_list_path)
    counter = NoduleCounter()

    mode = 'single' if print_each_series else 'sum'

    num_nodules = counter.count_and_analyze_nodules_of_multi_series(series_list_path = series_list_path,
                                                                    nodule_size_ranges=co_config['training']['nodule_size_ranges'],
                                                                    mode=mode)
                                                                    
    
    print("Number of nodules in {}:".format(series_list))
    if mode == 'sum':
        ordered_keys = ['benign', 'probably_benign', 'probably_suspicious', 'suspicious', 'all']
        for k in ordered_keys:
            print("{:19s}: {}".format(k, num_nodules[k]))
    else:
        ordered_keys = ['benign', 'probably_benign', 'probably_suspicious', 'suspicious']
        for series_info, num_nodule in zip(series_infos, num_nodules):
            num_info = ''
            for k in ordered_keys:
                num_info += "{:3d}, ".format(num_nodule[k])
            print("{:19s}: {}".format(series_info[1], num_info))