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
## Prepare data
1. For each client, prepare three txt files, which are train/val/test txt file, which contains the root path of one series of CT images. Each line of the txt file is the path of one CT image.
    - Example: COA_train.txt
    ```
    Folder,Filename
    /LDCT/CHEST001/,CHEST001
    /LDCT/CHEST002/,CHEST002
    ```
    Take first line as an example, the root folder of the CT images is `/LDCT/CHEST001`.There will be two folder in `./LDCT/CHEST001/`, `npy` and `mask`. In `npy` folder, there will be a npy file named `CHEST001.npy`, which contains the CT images, a npz file named `CHEST001_lobe.npz`, which contains the lobe mask of the CT images and a txt file named `lobe_info.txt`, which contains the lobe information of the CT images. In `mask` folder, there will be a npz file named `CHEST001.npz`, which contains the lung mask of the CT images.
    ```
    ├── LDCT
    │   ├── CHEST001
    │   |   ├── npy
    │   |   |   ├── CHEST001.npy
    │   |   |   ├── CHEST001_lobe.npz
    │   |   │   └── lobe_info.txt
    │   |   ├── mask
    │   |   │   └── CHEST001.npz
    │   ├── CHEST002
    │   |   ├── npy
    │   |   |   ├── CHEST002.npy
    │   |   |   ├── CHEST002_lobe.npz
    │   |   │   └── lobe_info.txt
    │   |   ├── mask
    │   |   │   └── CHEST002.npz
    │   ....
    ```
2. Write a yaml file to describe the clients information, e.g. `./config/clients/COA/stage1_clients.yaml`
## Training
Usage:
```shell
conda activate <your_env_name>
python main.py --config_path <config_path> --exp_name <exp_name> --clents_config_path <clients_config_path> --resume_folder <resume_folder> --pretrained_model_path <pretrained_model_path>
```
Arguments:
- `--config_path` the path of config file, e.g. `./config/stage1_fedavg.yaml`
- `--exp_name` the name of experiment, e.g. `exp1`
- `--clents_config_path` the path of clients config file, e.g. `./config/clients/stage1_clients.yaml`
- `--resume_folder` the path of resuming folder, default is `None`
- `--pretrained_model_path` the path of pretrained model, default is `None`