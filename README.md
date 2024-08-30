# Lung Nodule Detection Federated Learning

# 安裝
## 環境需求
- Anaconda
- Python >= 3.9
- Pytorch >= 2.0.1

## 安裝步驟
1. 複製專案
    ```shell
    git clone https://github.com/EonianCoda/lung-nodule-detection-federated-learning.git
    ```
2. 切換分支
    ```shell
    git checkout CPM_Net
    ```
3. 創建環境，如果是在linux環境下，可以直接執行以下指令
    ```shell
    source build_env.sh
    ```
    如果是在windows環境下，則需要手動創建環境
    ```shell
    conda create -n FL_CPM_NET python=3.9 -y
    conda activate FL_CPM_NET
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    conda install -c conda-forge cudatoolkit=11.2 -y
    pip install -r requirements.txt
    ```
## 資料準備 
    1. 下載資料集: 從NAS上下載資料集，有兩個zip檔案需要下載
        1. `BME/FL_dataset/ME_dataset/ME_dataset.zip`
        2. `BME/FL_dataset/LDCT_test/LDCT_test_dataset.zip`
    2. 解壓縮資料集，假設解壓縮到`./data`資料夾
    3. 以ME_dataset為例，解壓縮後的資料夾結構如下:
        ```
        ├── ME_dataset
        │   ├── CHEST1001
        │   |   ├── npy
        │   |   |   ├── CHEST1001.npy # LDCT影像，舉例來說，大小可能是(x, y, z) = (512, 512, 300)
        │   |   |   ├── CHEST1001_lobe.npz # LDCT影像的lobe mask
        │   |   |   ├── series_metadata.txt # LDCT影像的metadata，如重採樣前的pixel spacing與重採樣後的影像大小
        │   |   │   └── lobe_info.txt # LDCT影像的lobe mask的資訊，即有意義的lobe mask的大小
        │   |   ├── mask
        │   |   |   ├── CHEST1001_nodule_count.json # LDCT影像的肺結節標記資訊，下面再詳細說明
        │   |   │   └── CHEST1001.npz # LDCT影像的lung mask，以numpy的npz格式儲存，只有一個key: 'image'
        │   ├── CHEST1002
        │   |   ├── npy
        │   |   |   ├── CHEST1002.npy
        │   |   |   ├── CHEST1002_lobe.npz
        │   |   |   ├── series_metadata.txt
        │   |   │   └── lobe_info.txt
        │   |   ├── mask
        │   |   |   ├── CHEST1002_nodule_count.json
        │   |   │   └── CHEST1002.npz
        ```
    4. 以下以`ME_dataset/CHEST1001/mask/CHEST1001_nodule_count.json`說明肺結節標記資訊的格式
        1. `last_modified_time`: 最後修改時間
        2. `nodule_size`: 肺結節的大小，以pixel為單位
        3. `bboxes`: 肺結節的bounding box，每個bounding box有兩個點，分別是左上角與右下角的座標，座標的順序是(y, x, z)
        4. `nodule_start_slice_ids`: 肺結節的起始slice id，即z軸的座標
        ```json
        {
            "last_modified_time": 1703229794.5700548, 
            "nodule_size": [92, 87], 
            "bboxes": [[[64, 313, 146], [70, 321, 151]], [[100, 313, 81], [107, 323, 85]]], 
            "nodule_start_slice_ids": [146, 81]
        }
        ```
    5. 根據lobe mask進行資料的切割以加速訓練，修改`crop_images.py`內的`root`路徑，指到剛剛解壓縮的資料夾，然後執行以下指令:
        ```shell
        python crop_images.py
        ```
    6. 切割後的資料夾結構如下:
        ```shell
        ├── ME_dataset
        │   ├── CHEST1001
        │   |   ├── npy
        │   |   |   ├── CHEST1001.npy # LDCT影像，舉例來說，大小可能是(x, y, z) = (512, 512, 300)
        │   |   |   ├── CHEST1001_lobe.npz # LDCT影像的lobe mask
        │   |   |   ├── CHEST1001_crop.npy # 切割後的LDCT影像
        │   |   |   ├── CHEST1001_crop_lobe.npz # 切割後的lobe mask
        │   |   |   ├── series_metadata.txt # LDCT影像的metadata，如重採樣前的pixel spacing與重採樣後的影像大小
        │   |   │   └── lobe_info.txt # LDCT影像的lobe mask的資訊，即有意義的lobe mask的大小
        │   |   ├── mask
        │   |   |   ├── CHEST1001_nodule_count.json # LDCT影像的肺結節標記資訊
        │   |   |   ├── CHEST1001_nodule_count_crop.json # 切割後的肺結節標記資訊
        │   |   |   ├── CHEST1001.npz # LDCT影像的lung mask，以numpy的npz格式儲存，只有一個key: 'image'
        │   |   │   └── CHEST1001_crop.npz # 切割後的LDCT影像的lung mask，以numpy的npz格式儲存，只有一個key: 'image'
        ```
    7. 下載各個資料集的txt檔
# 使用
## 監督式學習訓練
進行CPM-Net模型監督式學習的方式是執行`cpm_train.py`，訓練CPM-Net模型，值得注意的是這個模型有許多參數可以調整，必須要在執行前先確認參數是否正確，下面是必須給定的參數
```shell
python cpm_train.py [--exp_nam EXP_NAME] [--train_set TRAIN_SET] [--val_set VAL_SET] [--test_set TEST_SET] [--memory_format MEMORY_FORMAT] [--start_val_epoch START_VAL_EPOCH]
```
1. `--exp_name`: 實驗名稱，例如 `exp1`
2. `--train_set`: 訓練集的路徑，例如 `./data/client0_train.txt`
3. `--val_set`: 驗證集的路徑，例如 `./data/client0_val.txt`
4. `--test_set`: 測試集的路徑，例如 `./data/client0_test.txt`
5. `--memory_format`: 訓練時的memory format，預設為`channels_first`，如果在linux系統上訓練，建議將其設置為`channels_last`，這樣可以加速約40%。
6. `--start_val_epoch`: 開始驗證的epoch數，預設為150

### 範例
1. 使用pretrained data進行訓練:
    ```shell
    python cpm_train.py --train_set ./data/pretrained_train.txt --val_set ./data/pretrained_val.txt --test_set ./data/all_client_test.txt --mixed_precision --val_mixed_precision --exp_name PT_bs5n8TPr06_posIg5_numNeg-1_iouL4_shapeL1_rot30 --start_val_epoch 300 --warmup_epochs 20 --val_interval 10 --epochs 500 --start_val_epoch 300 --model_class fl_modules.model.cpm_net.cpm_net --ema_warmup_epochs 300 --early_end_epoch 400
    ```
2. 使用訓練好的pretrained model進行訓練，假設pretrained model的路徑為`./save/PT.pth`:
    ```shell
    python cpm_train.py --train_set ./data/all_client_train.txt --val_set ./data/all_client_val.txt --test_set ./data/all_client_test.txt --mixed_precision --val_mixed_precision --exp_name FromPT_All__bs5n8TPr06_posIg5_numNeg-1_iouL4_shapeL1_rot30 --start_val_epoch 100 --warmup_epochs 5 --val_interval 2 --epochs 300 --start_val_epoch 120 --model_class fl_modules.model.cpm_net.cpm_net --ema_warmup_epochs 120 --early_end_epoch 250 --pretrained_model_path ./save/PT.pth
    ```

### 其他參數說明
以下列出一些較為重要的參數，若一些可以調整的超參數有預設值，這代表這是目前最佳的超參數，建議不要更動，
1. `--mixed_precision`: 訓練時是否使用混合精度進行訓練，預設為不開啟，但為了將batch_size設置為較大的值，建議將其設置為開啟
2. `--val_mixed_precision`: 驗證時是否使用混合精度進行訓練，預設為不開啟，但為了加速，建議開啟
3. `--batch_size`: 訓練時的batch size，預設為5
4. `--val_batch_size`: 驗證時的batch size，預設為2
5. `--epochs`: 訓練時的epoch數，預設為300
6. `--memory_format`: 訓練時的memory format，預設為`channels_first`，如果在linux系統上訓練，建議將其設置為
`channels_last`，這樣可以加速約40%。
7. `--tp_ratio`: 訓練時一個batch中positive和negative的比例，預設為0.6，即60%的positive和40%的negative
8. `--rand_rot`: 訓練時是否隨機旋轉，預設為`[30,0,0]`，即在xy平面上隨機旋轉30度，不在xz和yz平面上旋轉
9. `--lr`: 訓練時的learning rate，預設為0.002
10. `--start_val_epoch`: 開始驗證的epoch數，預設為150
11. `--not_apply_ema`: 不使用ema，預設為使用ema
12. `--ema_momentum`: ema的momentum，預設為0.998
13. `--ema_warmup_epochs`: ema的warmup epoch數，預設為-1，但通常會設置為與`--start_val_epoch`相同
14. `--pos_target_topk`: 訓練時positive target的數目，預設為7，即每顆肺結節選擇前7個最近的點作為前景
15. `--pos_ignore_ratio`: 訓練時ignore target的比例，預設為5，即每顆肺結節選擇pos_target_topk * pos_ignore_ratio個點作為ignore target，以14點為例，會選擇35個點作為ignore target，即每顆肺結節會有49個點被選擇，其中最靠近中心點的7個點為positive target，接下來的35個點為ignore target。
16. `num_samples`: 每筆資料的sample數，預設為8，一份病患資料包含一張LDCT 3D影像，舉例來說，大小可能是(x, y, z) = (512, 512, 300)，這份影像會先被切割成多個邊長為96的cube，sample為8，代表每筆LDCT 3D影像會選擇8個96x96x96的cube作為訓練資料，其中比例會是`tp_ratio`，positive cube被選擇的機率為`tp_ratio`，negative cube被選擇的機率為1-`tp_ratio`。
17. `iters_to_accumulate`: 等待多少個batch才更新權重，預設為2，即等待2個batch才更新權重
18. `cls_num_neg`: negative的數目，預設為-1，即全部都是negative
19. `cls_neg_pos_ratio`: negative和positive點數量的比例，預設為100，即positive和negative的比例為1:100，舉例來說，假設這個cube中有1顆肺結節，根據`pos_target_topk`為7，則會選擇7個前景點，而7*100=700，則會選擇700個背景點，總共為707個點，其中7個點為前景，700個點為背景。
20. `cls_num_hard`: hard negative的數目，預設為200，即如果此cube是背景cube，則從所有的negative中選擇損失值最大的前200個點作為目標。

## 聯邦學習訓練
進行聯邦學習的步驟如下:
1. 設置好聯邦設定檔，例如`./config/cpm_fedavg.yaml`，以下說明各個欄位的意義:
    ```yaml
    common:
        seed: 0 # 隨機種子
        save_dir: './save' # 儲存模型的資料夾
        save_local_state: False # 是否儲存local state(目前沒有實作)
        
    server:
        total_rounds: 250 # 總訓練round
        start_val_round: 100 # 開始驗證的round數，通過設置此參數可以減少訓練時間，因為不需要每個round都驗證
        val_interval: 1 # 驗證的間隔round數
        val_local: False # 是否驗證local model，在本地訓練並完成聚合前的模型稱為local model
        epoch_per_round: 1 # 每個round的epoch數
        best_model_metric_name: froc_mean_recall # 在聯邦學習中，選擇最佳模型的metric，預設為froc_mean_recall
    
        model: # 模型設定
            template: fl_modules.model.cpm_net.cpm_net.CpmNet
            params:
            n_filters: [64, 96, 128, 160]
            stem_filters: 32
            norm_type: batchnorm
            act_type: ReLU
            out_stride: 4
            detection_loss: # 損失函數設定
                template: fl_modules.model.cpm_net.loss.DetectionLoss
                params:
                crop_size: [96, 96, 96]
                pos_target_topk: 7
                pos_ignore_ratio: 5
                cls_num_neg: -1 # -1 means all negative samples
                cls_num_hard: 200
                cls_fn_weight: 4.0
                cls_fn_threshold: 0.8
                cls_neg_pos_ratio: 100 
                cls_hard_fp_thrs1: 0.5
                cls_hard_fp_thrs2: 0.7
                cls_hard_fp_w1: 1.5
                cls_hard_fp_w2: 2.0
                cls_focal_alpha: 0.75 # focal loss alpha
                cls_focal_gamma: 2.0 # focal loss gamma

        ema: # EMA設定
            apply: False # 是否使用EMA，預設為不使用，因為實驗發現
            params:
                momentum: 0.998
                apply_buffer: True

        optimizer: # 優化器設定
            template: fl_modules.optimizer.adamW.AdamW
            params:
                lr: 0.002
                weight_decay: 0.0001

        scheduler: # 學習率調整策略
            template: fl_modules.optimizer.scheduler.WarmupCosineAnnealingScheduler
            params:
            gamma: 0.01
            warmup_epochs: 10
            T_max: 300
            eta_min: 0.0005

        det_postprocess: # 驗證/測試 (檢測)後處理設定
            val:
                template: fl_modules.model.cpm_net.detection_post_process.DetectionPostprocess
                params:
                    topk: 60
                    threshold: 0.4 # 與actions/val/froc_det_thresholds[0]相同
                    min_size: 27
                    nms_topk: 20
                    crop_size: [96, 96, 96]
            test:
                template: fl_modules.model.cpm_net.detection_post_process.DetectionPostprocess
                params:
                    topk: 60
                    threshold: 0.2 # 與actions/test/froc_det_thresholds[0]相同
                    min_size: 27
                    nms_topk: 20
                    crop_size: [96, 96, 96]  

        actions: # 訓練/驗證/測試設定
            shared_params: # shared params for all actions
                memory_format: channels_first
                mixed_precision: True
                enable_progress_bar: True
                log_metric: False
                nodule_type_diameters: # 肺結節直徑範圍
                    benign: [0, 4]
                    probably_benign: [4, 6]
                    probably_suspicious: [6, 8]
                    suspicious: [8, -1]
                min_d: 0
                min_size: 27

            train:
                template: fl_modules.client.cpm_train.train
                params:
                    iters_to_accumulate: 1
                    lambda_cls: 4.0
                    lambda_iou: 4.0
                    lambda_shape: 1.0
                    lambda_offset: 1.0
                    batch_size: 5

            val:
                template: fl_modules.client.cpm_val.val
                params:
                    iou_threshold: 0.2 # 驗證採用比較高的iou threshold
                    froc_det_thresholds: [0.4, 0.5, 0.7] # 驗證時採用比較高的threshold
                    min_size: 27
                    nodule_size_mode: seg_size
                    apply_lobe: False
                    image_spacing: [1.0, 0.8, 0.8]
                    patch_label_type: none
                    batch_size: 2

            test:
                template: fl_modules.client.cpm_val.val
                params:
                    iou_threshold: 0.1 # 測試採用比較低的iou threshold
                    froc_det_thresholds: [0.2, 0.5, 0.7] # 測試時採用比較低的threshold
                    min_size: 27
                    nodule_size_mode: seg_size
                    apply_lobe: False
                    image_spacing: [1.0, 0.8, 0.8]
                    patch_label_type: none
                    batch_size: 1

        aggregation: # 聚合策略
            optimizer_aggregate_strategy: continue_global # continue_global, continue_local, reset，預設為continue_global，continue_global代表optimizer也會被聚合，continue_local代表optimizer不會被聚合，reset代表每輪訓練後，每個client的optimizer都會被reset
            template: 
                fl_modules.server.aggregation.fedavg.FedAvg
            params:
                model:
                    keep_local_state: []
                optimizer:
                    keep_local_state: [] # If strategy is not continue_global, this field is ignored.

        client:
            nodule_size_ranges: # pixel size, 目前只有FP會使用這個設定，TP會根據GT的直徑來決定
                benign: [0, 52] 
                probably_benign: [52, 176] 
                probably_suspicious: [176, 418]
                suspicious: [418, -1]

            shared_params:
                image_spacing: [1.0, 0.8, 0.8]
                min_size: 27
                norm_method: none
            
            train_dataset: # 訓練集設定
                template: fl_modules.dataset.cpm_dataset.TrainDataset
                params:
                    crop_fn:
                        template: fl_modules.dataset.crop.InstanceCrop
                        params:
                            crop_size: [96, 96, 96]
                            overlap_ratio: 0.25
                            rand_trans: [16, 16, 16]
                            rand_rot: [30, 0, 0]
                            instance_crop: True
                            tp_ratio: 0.6
                            sample_num: 8
                    
            val_dataset: # 驗證集設定
                template: fl_modules.dataset.cpm_dataset.DetDataset
                params:
                    apply_lobe: False # 是否套用lobe mask
                    out_stride: 4
                    SplitComb:
                        template: fl_modules.dataset.split_combine.SplitComb
                        params:
                            crop_size: [96, 96, 96]
                            overlap_size: [24, 24, 24]
                            do_padding: False
                            pad_value: 0
    ```
2. 設定好client資料集設定檔，例如`./config/clients/cpm_clients.yaml`，以下說明欄位意義
    ```yaml
    COA: # client名稱
        dataset_params: # 資料集設定，聯邦學習只需要設定train/val/test三個資料集的路徑
            train:
                series_list_path: './data/client0_train.txt'
            val:
                series_list_path: './data/client0_val.txt'
            test:
                series_list_path: './data/client0_test.txt'

    COB:
        dataset_params:
            train:
                series_list_path: './data/client1_train.txt'
            val:
                series_list_path: './data/client1_val.txt'
            test:
                series_list_path: './data/client1_test.txt'

    COC:
        dataset_params:
            train:
                series_list_path: './data/client2_train.txt'
            val:
                series_list_path: './data/client2_val.txt'
            test:
                series_list_path: './data/client2_test.txt'
    ```
3. 開始訓練，執行`main.py`，假設有預訓練模型`./save/pretrained.pth`，則可以執行以下指令:
    ```shell
    python main.py [--exp_name EXP_NAME] [--config_path CONFIG_PATH] [--clients_config_path CLIENTS_CONFIG_PATH] [--pretrained_model_path PRETRAINED_MODEL_PATH] [--resume_folder RESUME_FOLDER]
    ```
    1. `--exp_name`: 實驗名稱，例如 `exp1`
    2. `--config_path`: config設定檔的路徑，例如 `./config/cpm_fedavg.yaml`
    3. `--clients_config_path`: clients設定檔的路徑，例如 `./config/clients/cpm_clients.yaml`
    4. `--pretrained_model_path`: 預訓練模型的路徑，預設為 `None`
    5. `--resume_folder`: 繼續訓練的資料夾路徑，如果有設定，則會使用該資料夾繼續訓練，預設為 `None`
    範例
    ```shell
    python main.py --exp_name fedavg_baseline --config_path ./config/cpm_fedavg.yaml --clients_config_path ./config/clients/cpm_clients.yaml --pretrained_model_path ./save/pretrained.pth
    ```

## 半監督式學習 + 聯邦學習訓練
1. 設置好聯邦設定檔，例如`./config/cpm_ssl_fedavg_enhance_small.yaml`，以下說明各個欄位的意義，若與聯邦學習的設定檔有不同，則會在下面說明:
    ```yaml
    common:
        seed: 0 # 隨機種子
        save_dir: './save' # 儲存模型的資料夾
        save_local_state: False # 是否儲存local state(目前沒有實作)
    
    server:
        total_rounds: 250
        start_val_round: 60
        val_interval: 1
        val_local: False
        epoch_per_round: 1
        best_model_metric_name: froc_mean_recall
        
        model: 
            template: fl_modules.model.cpm_net.cpm_net.CpmNet
            params:
            n_filters: [64, 96, 128, 160]
            stem_filters: 32
            norm_type: batchnorm
            act_type: ReLU
            out_stride: 4

    detection_loss: # 標記資料的損失函數，與聯邦學習的設定檔相同
        template: fl_modules.model.cpm_net.loss_semi_soft.DetectionLoss
        params:
            crop_size: [96, 96, 96]
            pos_target_topk: 7
            pos_ignore_ratio: 5
            cls_num_neg: -1 # -1 means all negative samples
            cls_num_hard: 200
            cls_fn_weight: 4.0
            cls_fn_threshold: 0.8
            cls_neg_pos_ratio: 100
            cls_hard_fp_thrs1: 0.5
            cls_hard_fp_thrs2: 0.7
            cls_hard_fp_w1: 1.5
            cls_hard_fp_w2: 2.0
            cls_focal_alpha: 0.75
            cls_focal_gamma: 2.0
            
    unsupervised_detection_loss: # 未標記資料的損失函數
        template: fl_modules.model.cpm_net.loss_semi_soft.Unsupervised_DetectionLoss
        params:
            crop_size: [96, 96, 96]
            pos_target_topk: 7
            pos_ignore_ratio: 5
            cls_num_neg: -1 # -1 means all negative samples
            cls_num_hard: 200
            cls_fn_weight: 4.0
            cls_fn_threshold: 0.8
            cls_neg_pos_ratio: 100
            # 沒有cls_hard_fp_thrs1, cls_hard_fp_thrs2, cls_hard_fp_w1, cls_hard_fp_w2 這些參數
            
            # cls_hard_fp_thrs1: 0.5
            # cls_hard_fp_thrs2: 0.7
            # cls_hard_fp_w1: 1.5
            # cls_hard_fp_w2: 2.0
            cls_focal_alpha: 0.75
            cls_focal_gamma: 2.0

    optimizer: 
        template: fl_modules.optimizer.adamW.AdamW 
        params:
            lr: 0.001
            weight_decay: 0.0001

    scheduler: 
        template: fl_modules.optimizer.scheduler.WarmupCosineAnnealingScheduler
        params:
            gamma: 0.01
            warmup_epochs: 10
            T_max: 300
            eta_min: 0.0005

    det_postprocess:
        pseudo_label: # for first round to generate initial pseudo labels
            template: fl_modules.model.cpm_net.detection_post_process.DetectionPostprocess
            params:
                topk: 60
                threshold: 0.4
                min_size: 27
                nms_topk: 20
                crop_size: [160, 160, 160]

        train: # on the fly pseudo label generation
            template: fl_modules.model.cpm_net.detection_post_process.DetectionPostprocess
            params:
                topk: 60
                threshold: 0.2
                min_size: 27
                nms_topk: 20
                crop_size: [96, 96, 96]
        val:
            template: fl_modules.model.cpm_net.detection_post_process.DetectionPostprocess
            params:
                topk: 60
                threshold: 0.4
                min_size: 27
                nms_topk: 20
                crop_size: [96, 96, 96]
        test:
            template: fl_modules.model.cpm_net.detection_post_process.DetectionPostprocess
            params:
                topk: 60
                threshold: 0.2
                min_size: 27
                nms_topk: 20
                crop_size: [96, 96, 96]  

        actions:
            shared_params: # shared params for all actions
                memory_format: channels_first # Linux電腦上訓練時，建議設置為channels_last，可以加速約40%
                mixed_precision: True
                val_mixed_precision: True
                enable_progress_bar: True
                log_metric: False
                nodule_type_diameters:
                    benign: [0, 4]
                    probably_benign: [4, 6]
                    probably_suspicious: [6, 8]
                    suspicious: [8, -1]
                min_d: 0
                min_size: 27

            pseudo_label:
                template: fl_modules.client.pseudo_label.gen_pseu_labels
                params:
                    batch_size: 1
                    nms_keep_top_k: 40

            train:
                template: fl_modules.client.cpm_semi_threshold_tta_train_soft_tracking_enhance_small.train
                params:
                    # Shared args
                    iters_to_accumulate: 1
                    batch_size: 4

                    # Supervised learning args
                    lambda_cls: 4.0
                    lambda_iou: 4.0
                    lambda_offset: 1.0
                    lambda_shape: 1.0

                    # Semi-supervised learning args
                    ema_buffer: True
                    sharpen_cls: 1.0 # 1.0表示不sharpen
                    select_bg_crop: 3

                    lambda_pseu_cls: 4.0
                    lambda_pseu_iou: 1.0
                    lambda_pseu_offset: 1.0
                    lambda_pseu_shape: 0.1
                    lambda_pseu: 1.0

                    semi_ema_alpha: 0.998
                    pseudo_label_threshold: 0.6
                    pseudo_background_threshold: 0.4
                    pseudo_crop_threshold: 0.55
                    pseudo_remove_threshold: 0.5 # 必須與unlabeled_train_dataset的pseudo_remove_threshold數值相同
                    pseudo_update_iou_threshold: 0.05
                    pseudo_update_ema_alpha: 0.95

                    pseudo_nms_topk: 10
                    semi_increase_ratio: 1.2
                    pseudo_tracking_warmup_epochs: 30
                    pseudo_update_interval: -1 ##TODO，未實作

            val:
                template: fl_modules.client.cpm_val.val
                params:
                    iou_threshold: 0.2
                    froc_det_thresholds: [0.4, 0.5, 0.7]
                    min_size: 27
                    nodule_size_mode: seg_size
                    apply_lobe: False
                    image_spacing: [1.0, 0.8, 0.8]
                    patch_label_type: none
                    batch_size: 2

            test:
                template: fl_modules.client.cpm_val.val
                params:
                    iou_threshold: 0.1
                    froc_det_thresholds: [0.2, 0.5, 0.7]
                    min_size: 27
                    nodule_size_mode: seg_size
                    apply_lobe: False
                    image_spacing: [1.0, 0.8, 0.8]
                    patch_label_type: none
                    batch_size: 1

        aggregation:
            optimizer_aggregate_strategy: continue_global # continue_global, continue_local, reset
            template:
            fl_modules.server.aggregation.ssl_fedavg.FedAvg
            params:
                model:
                    keep_local_state: []
                optimizer:
                    keep_local_state: [] # If strategy is not continue_global, this field is ignored.

        client:
            nodule_size_ranges: # pixel size,
                benign: [0, 52] 
                probably_benign: [52, 176] 
                probably_suspicious: [176, 418]
                suspicious: [418, -1]

            shared_params:
                image_spacing: [1.0, 0.8, 0.8]
                min_size: 27
                min_d: 0
                norm_method: none
            
            train_dataset:
                template: fl_modules.dataset.cpm_dataset_semi_tta_tracking_enhance_small.TrainDataset
                params:
                    crop_fn:
                        template: fl_modules.dataset.crop.InstanceCrop
                        params:
                        crop_size: [96, 96, 96]
                        overlap_ratio: 0.25
                        rand_trans: [16, 16, 16]
                        rand_rot: [30, 0, 0]
                        instance_crop: True
                        tp_ratio: 0.75
                        sample_num: 6

            unlabeled_det_dataset: # 未標記預測用的資料集
                template: fl_modules.dataset.cpm_dataset_val_aug.DetDataset
                params:
                    apply_lobe: True
                    out_stride: 4
                    SplitComb:
                        template: fl_modules.dataset.split_combine.SplitComb
                        params:
                        crop_size: [160, 160, 160] # 預測用時並非使用96x96x96的cube，而是使用160x160x160的cube
                        overlap_size: [16, 16, 16] # 指定10%的重疊率
                        do_padding: False
                        pad_value: 0

            unlabeled_train_dataset: # 未標記訓練用的資料集
                template: fl_modules.dataset.cpm_dataset_semi_tta_tracking_enhance_small.UnLabeledDataset
                params:
                    pseudo_remove_threshold: 0.5 # same as pseudo_remove_threshold in train action
                    pseudo_crop_threshold: 0.55
                    use_gt_crop: False
                    pseudo_update_ema_alpha: 0.95 # same as pseudo_update_ema_alpha in train action
                    use_rotate90: True # 是否使用旋轉90度作為偽標籤生成時的augmentation
                    pseudo_label_threshold: 0.6 # 偽標籤選擇為前景的threshold
                    crop_fn:
                        template: fl_modules.dataset.crop_semi_tta_tracking.InstanceCrop
                        params:
                            crop_size: [96, 96, 96]
                            overlap_ratio: 0.25
                            rand_trans: [16, 16, 16]
                            tp_ratio: 0.75
                            sample_num: 6
                    
            val_dataset:
                template: fl_modules.dataset.cpm_dataset.DetDataset
                params:
                    apply_lobe: False
                    out_stride: 4
                    SplitComb:
                        template: fl_modules.dataset.split_combine.SplitComb
                        params:
                        crop_size: [96, 96, 96]
                        overlap_size: [24, 24, 24]
                        do_padding: False
                        pad_value: 0
    ```
2. 設定好client資料集設定檔，例如`./config/clients/cpm_clients_ssl_r02.yaml`，沒有加入預訓練模型產生的偽標籤的設定檔範例是`./config/clients/cpm_clients_ssl_r02_noPseu.yaml`，以下說明欄位意義
    ```yaml
    COA:
        dataset_params:
            train:
                series_list_path: './data/new_unlabeled/client0_r02_labeled_train.txt'
            unlabeled_train: # 未標記訓練集
                series_list_path: './data/new_unlabeled/client0_r02_unlabeled_train.txt'
                pseudo_label_pkl_path: './save/pt_pseu_labels_client0_r02_new.pkl' # 預訓練模型產生的偽標籤，設定這一項可以減少每次訓練時重新生成偽標籤的時間
            unlabeled_det: # 未標記預測集
                series_list_path: './data/new_unlabeled/client0_r02_unlabeled_train.txt'
            val:
                series_list_path: './data/client0_val.txt'
            test:
                series_list_path: './data/client0_test.txt'

    COB:
        dataset_params:
            train:
                series_list_path: './data/new_unlabeled/client1_r02_labeled_train.txt'
            unlabeled_train:
                series_list_path: './data/new_unlabeled/client1_r02_unlabeled_train.txt'
                pseudo_label_pkl_path: './save/pt_pseu_labels_client1_r02_new.pkl'
            unlabeled_det:
                series_list_path: './data/new_unlabeled/client1_r02_unlabeled_train.txt'
            val:
                series_list_path: './data/client1_val.txt'
            test:
                series_list_path: './data/client1_test.txt'

    COC:
        dataset_params:
            train:
                series_list_path: './data/new_unlabeled/client2_r02_labeled_train.txt'
            unlabeled_train:
                series_list_path: './data/new_unlabeled/client2_r02_unlabeled_train.txt'
                pseudo_label_pkl_path: './save/pt_pseu_labels_client2_r02_new.pkl'
            unlabeled_det:
                series_list_path: './data/new_unlabeled/client2_r02_unlabeled_train.txt'
            val:
                series_list_path: './data/client2_val.txt'
            test:
                series_list_path: './data/client2_test.txt'
    ```
3. 開始訓練，執行`main.py`，假設有預訓練模型`./save/pretrained.pth`，則可以執行以下指令:
    ```shell
    python main.py [--exp_name EXP_NAME] [--config_path CONFIG_PATH] [--clients_config_path CLIENTS_CONFIG_PATH] [--pretrained_model_path PRETRAINED_MODEL_PATH] [--resume_folder RESUME_FOLDER] [--ssl]
    ```
    1. `--exp_name`: 實驗名稱，例如 `exp1`
    2. `--config_path`: config設定檔的路徑，例如 `./config/cpm_ssl_fedavg_enhance_small.yaml`
    3. `--clients_config_path`: clients設定檔的路徑，例如 `./config/clients/cpm_clients_ssl_r02.yaml`或`./config/clients/cpm_clients_ssl_r02_noPseu.yaml`
    4. `--pretrained_model_path`: 預訓練模型的路徑，預設為 `None`
    5. `--resume_folder`: 繼續訓練的資料夾路徑，如果有設定，則會使用該資料夾繼續訓練，預設為 `None`
    6. `--ssl`: 是否進行半監督式學習，預設為不進行，因為這裡是半監督式學習的設定檔，所以要加入`--ssl`
    範例:
    ```shell
    python main.py --exp_name fedavg_ssl_baseline_r02_fg06_bg04_lr1e-3_enhanceSmall_fromFSSLR02 --config_path ./config/cpm_ssl_fedavg_enhance_small.yaml --clients_config_path ./config/clients/cpm_clients_ssl_r02.yaml --pretrained_model ./save/pretrained.pth --ssl
    ```

## 驗證
執行`cpm_val.py`進行驗證
```shell
    python cpm_val.py [--val_set VAL_SET] [--model_path MODEL_PATH] [--val_mixed_precision] [--apply_aug] [--apply_lobe] [--crop_size CROP_SIZE] [--overlap_ratio OVERLAP_RATIO] [--patch_label_type PATCH_LABEL_TYPE]
```
1. `--val_set`: 驗證集的路徑，例如`./data/all_client_test.txt`
2. `--model_path`: 模型的路徑，例如`./save/PT.pth`
3. `--val_mixed_precision`: 是否使用mixed precision，預設為不使用，但盡量使用，因為可以加速約40%
4. `--apply_aug`: 是否套用augmentation，預設為不套用，加入這項可以提升驗證的效果約2%，但會增加驗證時間
5. `--apply_lobe`: 是否套用lobe mask，預設為不套用，加入這項可以提升驗證的效果，但會增加驗證時間
6. `--crop_size`: 驗證時的crop size，預設為96，但經過測試，160的效果比較好
7. `--overlap_ratio`: 驗證時的overlap ratio，預設為0.25，但經過測試，160搭上0.1的overlap ratio效果比較好
8. `--patch_label_type`: 驗證時的patch label type，所謂的patch label是指醫師經過二次FP標記後的結果，在論文上的實驗結果都是設定為benign的結果。

範例: 假設有訓練好的模型`./save/PT.pth`，則可以執行以下指令:
```shell
python cpm_val.py --val_set ./data/all_client_test.txt --model_path ./save/PT.pth --val_mixed_precision --apply_aug --apply_lobe --crop_size 160 --overlap_ratio 0.1 --patch_label_type benign
```
# 程式碼架構
以下是專案的資料夾結構，以及各個資料夾的功能說明。
```
├── config
│   └── clients
├── fl_modules
│   ├── client
│   ├── dataset
│   ├── eval
│   ├── model
│   ├── inference
│   ├── optimizer
│   ├── server
│   └── utilities
└── main.py
```
以下是`fl_modules`的資料夾功能說明:
1. `client`: 客戶端(CO)程式碼，包含訓練、驗證的程式碼
2. `dataset`: 資料集程式碼，包含資料集的讀取、前處理的程式碼
3. `eval`: 評估程式碼
4. `model`: 模型程式碼，包含模型的建立、損失函數、後處理的程式碼
5. `inference`: 推論程式碼，包含推論的程式碼
6. `optimizer`: 優化器程式碼，包含優化器的程式碼
7. `server`: 伺服器(AG)程式碼，包含聯邦學習與聚合函數的程式碼
8. `utilities`: 工具程式碼，包含一些工具程式碼，例如logger、progress bar等

## client
### 資料夾結構
```
├── client
│   ├── average_meter.py
│   ├── client.py
│   ├── cpm_semi_threshold_tta_train_soft_tracking_enhance_small.py # 
│   ├── cpm_semi_threshold_tta_train_soft_tracking.py # (已棄用，因為enhance_small的效果比較好)
│   ├── cpm_train.py
│   ├── cpm_val_aug.py
│   ├── cpm_val.py
│   ├── pseudo_label.py
│   ├── ssl_client.py
│   └── utils.py
```
1. `average_meter.py`: 計算平均值的程式碼
2. `client.py`: 客戶端的程式碼
3. `cpm_semi_threshold_tta_train_soft_tracking_enhance_small.py`: 半監督式學習的訓練程式碼
4. `cpm_semi_threshold_tta_train_soft_tracking.py`: 半監督式學習的訓練程式碼(已棄用)
5. `cpm_train.py`: 訓練程式碼
6. `cpm_val_aug.py`: 驗證程式碼(套用augmentation)
7. `cpm_val.py`: 驗證程式碼
8. `pseudo_label.py`: 生成偽標籤的程式碼
9. `ssl_client.py`: 半監督式學習的客戶端程式碼，與`client.py`的差異在於`ssl_client.py`會使用未標記資料集
10. `utils.py`: 一些工具程式碼

### avgerage_meter.py
- 有一個`AverageMeter`的class，用來計算平均值

### client.py
- 函數`build_train_augmentation`: 建立訓練的augmentation函數

以下介紹`Client` class的重要方法:
1. `_build_dataset_config`: 建立資料集設定，根據`__init__`的參數來建立訓練、驗證、測試資料集
2. `_build_action`: 建立訓練、驗證、測試的action
3. `train`: 訓練的方法，含有一個Lazy init的方法，當第一次呼叫時，才會初始化訓練的資料集，傳入值包含第幾輪、epoch的數量、模型、優化器、ema。
4. `val`: 驗證的方法，含有一個Lazy init的方法，當第一次呼叫時，才會初始化驗證的資料集，傳入值包含第幾輪、模型、後處理的函數、是否為global model的flag。
5. `test`: 測試的方法，含有一個Lazy init的方法，當第一次呼叫時，才會初始化測試的資料集，傳入值包含第幾輪、模型、後處理的函數。

### cpm_semi_threshold_tta_train_soft_tracking_enhance_small.py
半監督式學習訓練的函數，以下介紹重要函數
1. `unsupervised_train_one_step_wrapper`: 半監督學習一個batch的函數裝飾器，使用裝飾器的原因是因為需要設定memory_format與loss_fn，這樣可以減少重複的程式碼。
2. `train_one_step_wrapper`: 監督學習一個batch的函數裝飾器，使用裝飾器的原因是因為需要設定memory_format與loss_fn，這樣可以減少重複的程式碼。
3. `sharpen_prob`: sharpen的函數，用來使偽標籤的機率更加尖銳
4. `burn_in_train`: burn-in的訓練函數，用來訓練偽標籤的模型(目前沒有實作)
5. `train`: 半監督式學習的訓練函數

## dataset
### 資料夾結構
```
├── config
│   ├── transform
│   ├── collate.py
│   ├── cpm_dataset_semi_tta_tracking_enhance_small.py
│   ├── cpm_dataset_semi_tta_tracking.py # (已棄用，因為enhance_small的效果比較好)
│   ├── cpm_dataset_val_aug.py
│   ├── cpm_dataset.py
│   ├── crop_fast.py
│   ├── crop_rand_spacing.py
│   ├── crop_rand_spacing.py
│   ├── crop_semi_tta_tracking.py
│   ├── crop.py
│   ├── split_combine.py
│   └── utils.py
```
1. transform: 存放augmentation的程式碼
2. collate.py: pytorch dataloder collate_fn的程式碼
3. cpm_dataset_semi_tta_tracking_enhance_small.py: 半監督式學習的資料集程式碼
4. cpm_dataset_semi_tta_tracking.py: 半監督式學習的資料集程式碼(已棄用)
5. cpm_dataset_val_aug.py: 驗證資料集的程式碼(套用augmentation)
6. cpm_dataset.py: 資料集的程式碼
7. crop_fast.py: 快速crop的程式碼 (已棄用)
8. crop_rand_spacing.py: 隨機spacing的crop的程式碼 (目前沒有使用)
9. crop_semi_tta_tracking.py: 半監督式學習的crop的程式碼 
10. crop.py: crop的程式碼
11. split_combine.py: 驗證時切割cube與合併結果的程式碼
12. utils.py: 一些工具程式碼

## eval
### 資料夾結構
```
├── config
│   ├── eval.py
│   ├── markdown_writer.py
│   ├── nodule_finding.py
│   └── nodule_typer.py
```
1. eval.py: 評估的程式碼
2. markdown_writer.py: 產生markdown結果log的程式碼
3. nodule_finding.py: 在評估中，用於表示一顆肺結節的程式碼
4. nodule_typer.py: 分類結節的程式碼

## inference
### 資料夾結構
```
├── nodule_counter.py
```
除了`nodule_counter.py`外，這個資料夾其他的程式碼都沒有使用

## model
### 資料夾結構
```
├── model
│   └── cpm_net
│   │   ├── cpm_net.py
│   │   ├── detection_post_process.py
│   │   ├── loss_semi_soft.py
│   │   ├── loss.py
│   │   └── modules.py
```
1. `cpm_net.py`: CPM-Net的模型程式碼
2. `detection_post_process.py`: 後處理的程式碼
3. `loss_semi_soft.py`: 半監督式學習的損失函數程式碼
4. `loss.py`: 損失函數程式碼
5. `modules.py`: 模型的模組程式碼


## optimizer
```
├── optimizer
│   ├── adamW.py
│   ├── ema.py
│   ├── fedprox.py
│   ├── scaffold.py
│   └── scheduler.py
```

1. `adamW.py`: AdamW的優化器程式碼
2. `ema.py`: EMA的程式碼
3. `fedprox.py`: FedProx的程式碼
4. `scaffold.py`: Scaffold的程式碼
5. `scheduler.py`: 學習率調整器的程式碼

## server
```
├── server
│   ├── aggregation
│   │   ├── aggregation.py
│   │   ├── fedavg.py
│   │   └── ssl_fedavg.py
│   ├── server.py
│   └── ssl_server.py
```
1. `aggregation`: 聚合函數的程式碼
    1. `aggregation.py`: 聚合函數的父類別
    2. `fedavg.py`: FedAvg的程式碼
    3. `ssl_fedavg.py`: 半監督式學習的FedAvg的程式碼
2. `server.py`: 伺服器的程式碼
3. `ssl_server.py`: 半監督式學習的伺服器程式碼