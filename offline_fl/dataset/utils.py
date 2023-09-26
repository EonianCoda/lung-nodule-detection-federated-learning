from typing import List, Tuple, Dict
    
def get_nodule_type(nodule_size: int, nodule_size_ranges: Dict[str, Tuple[int, int]]) -> str:
    for nodule_type, size_range in nodule_size_ranges.items():
        lower_bound, upper_bound = size_range
        if upper_bound == -1:
            upper_bound = nodule_size + 1

        if nodule_size > lower_bound and nodule_size <= upper_bound:
            return nodule_type

def load_series_list(series_list_path: str) -> List[Tuple[str, str]]:
    """
    Return:
        series_list: list of tuples (series_folder, file_name)

    """
    with open(series_list_path, 'r') as f:
        lines = f.readlines()
        lines = lines[1:] # Remove the line of description
        
    series_list = []
    for series_info in lines:
        series_info = series_info.strip()
        series_folder, file_name = series_info.split(',')
        series_list.append([series_folder, file_name])
    return series_list

def get_start_and_end_slice(path: str) -> List[int]:
    # Get index of start slice and end slice in lobe
    with open(path, 'r') as f:
        lines = f.readlines()[3:]
        lines = [line.strip() for line in lines]
    first_slice_id, end_slice_id = [int(v) for v in lines[0].split(',')]
    return [first_slice_id, end_slice_id]