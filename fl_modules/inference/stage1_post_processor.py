import os
from typing import List, Union
from .predictor import PredictorStage1

class PostProcessorStage1(object):
    def __init__(self, stage1_model_path: str, use_lobe = False):
        
        self.use_lobe = use_lobe
        self.predictor_config = {'model': stage1_model_path,
                                 'device': 'cuda:0',
                                'use_lobe': use_lobe}
        self.predictor = None

    def process_series(self, 
                       series_paths: Union[List[str], str], 
                       gt_mask_maps_paths: Union[List[str], str],
                       pred_mask_save_paths: Union[List[str], str] = None) -> List[list]:
        predictor = self._get_predictor()
        results = predictor.get_bboxes_ious_of_nodules_in_series(series_paths, gt_mask_maps_paths, pred_mask_save_paths)
        return results
    
    def _get_predictor(self) -> PredictorStage1:
        if self.predictor == None:
            self.predictor = PredictorStage1(**self.predictor_config)
        return self.predictor
    
    def gen_cache_save_path(self, series_folder: str, model_name: str, file_name: str) -> str:
        """Generate the path of saving in cache folder
        
        Return: str
            A path of given paramenters, e.g. '/ID-0000001/Std-0001/Ser-001/stage1_post_process/cache/model-09/ID-000001_Std-0001_Ser-001.txt'
        """
        lobe_info = 'wLobe' if self.use_lobe else 'woLobe'
        save_path = os.path.join(series_folder, 'stage1_post_process', 'cache', model_name, f'{file_name}_{lobe_info}.txt')
        return save_path

    def write_result_of_stage1_post_process(self, save_path: str, results: list) -> None:
        """ Write the inference result
        """
        # create annotations
        nodules = []
        nodule_sizes = []
        for nodule in results:
            nodule_sizes.append(nodule[0]) # nodule[0] is [gt_nodule_index, gt_nodule_size, pred_nodule_size]
            nodules.append(nodule[1:]) # nodule[1:] is a list of tuple of [z, y_min, x_min, y_max, x_max, iou]
        
        lines = ['gt_nodule_index,gt_nodule_size,pred_nodule_size slice_number,y_min,x_min,y_max,x_max,iou slice_number,y_min,x_min,y_max,x_max,iou...\n'] 
        for nodeul_size, nodule in zip(nodule_sizes, nodules):
            nodeul_info = ','.join([str(value) for value in nodeul_size])
            line = [nodeul_info]
            for nodule_bbox in nodule:
                line.append(','.join([str(value) for value in nodule_bbox]))
            # Separate each bbox2d by space
            line = ' '.join(line) 
            lines.append(line + '\n')
    
        # Remove the new line('\n') in last line
        lines[-1] = lines[-1][:-1]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.writelines(lines)