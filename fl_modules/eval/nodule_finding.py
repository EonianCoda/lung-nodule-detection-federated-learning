import math
import copy
from typing import List

def to_float(x):
    return float(x) if x is not None else None

class NoduleFinding:
    """
    Represents a nodule
    
    Attributes:
        name (str): The name of the series.
        ctr_x (float): The x-coordinate of the nodule's center.
        ctr_y (float): The y-coordinate of the nodule's center.
        ctr_z (float): The z-coordinate of the nodule's center.
        w (float): The width of the nodule.
        h (float): The height of the nodule.
        d (float): The depth of the nodule.
        prob (float): The predicted probability of the nodule. 
        cand_id (int): The candidate ID of the nodule. 
        nodule_type (str): The type of the nodule.
        is_gt (bool): Whether the nodule is a ground truth nodule.        
    """
    def __init__(self, series_name: str, ctr_x: float, ctr_y: float, ctr_z: float, w: float, h: float, d: float, nodule_type: str, 
                 prob=-1, is_gt=False, **kwargs):
        # set the variables and convert them to the correct type
        self.series_name = series_name
        self.ctr_x = to_float(ctr_x)
        self.ctr_y = to_float(ctr_y)
        self.ctr_z = to_float(ctr_z)
        self.w = to_float(w)
        self.h = to_float(h)
        self.d = to_float(d)
        self.prob = to_float(prob)
        self.nodule_type = nodule_type
        
        self.is_gt = is_gt
        # Matching Status
        self.match_iou = -1
        self.match_nodule_finding = None 
        
        for key, value in kwargs.items():
            setattr(self, key, to_float(value))
        
    def set_match(self, match_iou: float, match_nodule_finding, force=False):
        # if the nodule is a ground truth nodule, TH match_iou is the iou between the nearest candidate and the ground truth
        # if the nodule is a candidate, TH match_iou is the iou between the candidate and the ground truth
        match_iou = to_float(match_iou)
        if self.match_nodule_finding is None or match_iou > self.match_iou or force:
            self.match_iou = match_iou
            if match_nodule_finding is not None:
                self.match_nodule_finding = copy.deepcopy(match_nodule_finding)
                
    # def get_box(self) -> List[int]:
    #     """
    #     Returns the bounding box of the nodule in the format [[z1, y1, x1], [z2, y2, x2]]
    #     """
    #     return [[self.ctr_z - self.d/2, self.ctr_y - self.h/2, self.ctr_x - self.w/2],
    #             [self.ctr_z + self.d/2, self.ctr_y + self.h/2, self.ctr_x + self.w/2]]
    
    def get_box(self) -> List[int]:
        """
        Returns the bounding box of the nodule in the format [[z1, y1, x1], [z2, y2, x2]]
        """
        return [[round(self.ctr_z - self.d/2), round(self.ctr_y - self.h/2), round(self.ctr_x - self.w/2)],
                [math.ceil(self.ctr_z + self.d/2), math.ceil(self.ctr_y + self.h/2), math.ceil(self.ctr_x + self.w/2)]]
    
    def __str__(self):
        return f"({self.series_name:16s}, {self.ctr_x}, {self.ctr_y}, {self.ctr_z}, {self.w}, {self.h}, {self.d}, {self.prob}, {self.is_gt})"