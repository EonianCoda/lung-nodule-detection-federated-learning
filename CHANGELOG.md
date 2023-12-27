# Change Log
---
## [TODO]
## [Known Bug]
## Unreleased
### Fixed
- change calculate lobe mask using tensorflow to torch
### Added
- add `split_data.py` to split lung nodules data into pretrained/ train_labeled/ train_unlabeled/ val/ test
- add `/tools/create_dataset.py` for creating structured dataset
- add `/tools/convert_series_list_cross_device.py` for converting series list cross device
- add `prepare_lobe.py` for generating lobe mask
- add semi supervised learning(ssl) training
### Changed
- add `cupy-cuda12x` into `requeriments.txt` for generating lobe
- add ``
## [1.0.0] - before 2023-12-26