import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging
import shutil
import argparse

from fl_modules.inference.stage1_post_processor import PostProcessorStage1
from fl_modules.utilities import load_yaml, setup_logging

data_config_path = './config/data_config.yaml'
data_config = load_yaml(data_config_path)
root = data_config['root']

series_txts = ['fl_cmp_valA.txt','fl_cmp_valB.txt','fl_cmp_valC.txt','fl_cmp_trainA.txt','fl_cmp_trainB.txt','fl_cmp_trainC.txt', 'val.txt', 'LDCT_test.txt']

logger = logging.getLogger(__name__)

def do_stage1_post_process(stage1_model_path: str, 
                           series_txts: list,
                           save_pred_mask: bool = False) -> None:
    logger.info("Start to do stage1 post process")
    model_name = os.path.basename(stage1_model_path).split('.')[0]

    def post_process(stage1_post_processor: PostProcessorStage1, path: str):
        with open(path, 'r') as f:
            lines = f.readlines()
            lines = lines[1:] # Remove the line of description
        
        # Process each line
        series_infos = []
        for line in lines:
            line = line.strip()
            series_folder, file_name = line.split(',') # e.g. ('ID-0000001/Std-0001/Ser-001', 'ID-000001_Std-0001_Ser-001') 
            series_infos.append([series_folder, file_name])

        series_paths = []
        gt_mask_maps_paths = []
        saving_paths = []
        pred_mask_saving_paths = []
        # stage1_post_process_path_list = []
        for i, (series_folder, file_name) in enumerate(series_infos):
            saving_path = stage1_post_processor.gen_cache_save_path(series_folder, model_name, file_name)
            pred_mask_saving_path = os.path.join(series_folder, 'stage1_post_process', 'pred_mask.npz')
            os.makedirs(os.path.dirname(pred_mask_saving_path), exist_ok=True)
            # If the result of stage 1 post process exist, then not do post process again.
            if os.path.exists(saving_path):
                continue
            
            series_paths.append(os.path.join(series_folder, 'npy', f'{file_name}.npy'))
            gt_mask_maps_paths.append(os.path.join(series_folder, 'mask', f'{file_name}.npz'))
            saving_paths.append(saving_path)
            
            if save_pred_mask:
                pred_mask_saving_paths.append(pred_mask_saving_path)

        if len(series_paths) != 0:
            results = stage1_post_processor.process_series(series_paths, gt_mask_maps_paths, pred_mask_saving_paths)
            for result, saving_path in zip(results, saving_paths):
                stage1_post_processor.write_result_of_stage1_post_process(saving_path, result)

        # Move to result from 'stage1_post_process/cache/{model_name}' to 'stage1_post_process/'
        for series_folder, file_name in series_infos:
            saving_path = stage1_post_processor.gen_cache_save_path(series_folder, model_name, file_name)
            default_txt_path = os.path.join(series_folder, 'stage1_post_process', f'{file_name}.txt')
            shutil.copy(saving_path, default_txt_path)

    
    for series_txt in series_txts:
        series_list_path = os.path.join(root, series_txt)        
        logger.warning(series_list_path)
        stage1_post_processor = PostProcessorStage1(stage1_model_path, use_lobe= True)
        post_process(stage1_post_processor, series_list_path)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--save_pred_mask', action='store_true', default=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    model_path = args.model_path
    save_pred_mask = args.save_pred_mask
    setup_logging()
    logger.info(f"model path = '{model_path}'")
    logger.info(f"save_pred_mask: {save_pred_mask}")
    do_stage1_post_process(model_path, series_txts, save_pred_mask)