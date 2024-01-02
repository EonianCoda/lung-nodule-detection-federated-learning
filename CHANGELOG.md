# Change Log
---
## [TODO]
## [Known Bug]
## Unreleased
### Main Feature
- add semi supervised learning(ssl) training

## [1.1.0] - 2023-12-30
### Fixed
- change calculate lobe mask using tensorflow to torch
### Added
- add `split_data.py` to split lung nodules data into pretrained/ train_labeled/ train_unlabeled/ val/ test
- add `/tools/create_dataset.py` for creating structured dataset
- add `/tools/convert_series_list_cross_device.py` for converting series list cross device
- add `/tools/cal_number_of_nodules.py` for calculating number of nodules
- add `prepare_lobe.py` for generating lobe mask

### Changed
- add `cupy-cuda11x` into `requeriments.txt` for generating lobe
- revise `build_env.sh` for `cupy-cuda11x`
### Refactor
- refactor `/fl_modules/inference/utils.py`, `/fl_modules/inference/predictor.py` and `/fl_modules/inference/utils.py` to speed up inference
    - refactor class `PrefetchSeries` from object to `torch.utils.data.Dataset` to use `torch.utils.data.DataLoader` for parallel data loading  
## [1.0.0] - before 2023-12-26