# Lung Nodule Detection Federated Learning

## Environment
```shell
python >= 3.8
torch >= 2.0.1
```

## Installation
- Via Anaconda
```shell
source build_env.sh
```

# Details
## Training
Usage:
```shell
python main.py --config_path <config_path> --exp_name <exp_name> --clents_config_path <clients_config_path> --resume_folder <resume_folder> --pretrained_model_path <pretrained_model_path>
```
Arguments:
- `--config_path` the path of config file
- `--exp_name` the name of experiment
- `--clents_config_path` the path of clients config file
- `--resume_folder` the path of resuming folder, default is `None`
- `--pretrained_model_path` the path of pretrained model, default is `None`