@REM python train_stage1.py --extra_info CO0_local_mixP_more_slice --train_set client0_train.txt --val_set client0_val.txt --test_set client0_test.txt --num_epoch 50  --num_workers 4 --pin_memory --mixed_precision --pretrained_model_path "./save/[2024-01-15-2351]_pretrained_mixP_more_slice/best.pth"
@REM python train_stage1.py --extra_info CO1_local_mixP_more_slice --train_set client1_train.txt --val_set client1_val.txt --test_set client1_test.txt --num_epoch 50  --num_workers 4 --pin_memory --mixed_precision --pretrained_model_path "./save/[2024-01-15-2351]_pretrained_mixP_more_slice/best.pth"
@REM python train_stage1.py --extra_info CO2_local_mixP_more_slice --train_set client2_train.txt --val_set client2_val.txt --test_set client2_test.txt --num_epoch 50  --num_workers 4 --pin_memory --mixed_precision --pretrained_model_path "./save/[2024-01-15-2351]_pretrained_mixP_more_slice/best.pth"


@REM python main.py --exp_name test_bug --config_path ./config/cpm_fedavg.yaml --clients_config_path ./config/clients/cpm_clients.yaml
@REM python main.py --exp_name fedavg_baseline --config_path ./config/cpm_fedavg.yaml --clients_config_path ./config/clients/cpm_clients.yaml 


@REM python cpm_train.py --train_set ./data/pretrained_train.txt --val_set ./data/pretrained_val.txt --test_set ./data/all_client_test.txt --mixed_precision --val_mixed_precision --exp_name PT_NoX4_bs5n8TPr06_posIg5_numNeg-1_lrGa01_iouL4_shapeL1_rot30 --start_val_epoch 300 --warmup_epochs 20 --val_interval 10 --epochs 500 --start_val_epoch 300 --model_class fl_modules.model.cpm_net.cpm_net --ema_warmup_epochs 300 --early_end_epoch 400

@REM python main.py --exp_name fedavg_baseline --config_path ./config/cpm_fedavg.yaml --clients_config_path ./config/clients/cpm_clients.yaml --pretrained_model_path ./save/pretrained.pth
@REM python main.py --exp_name fedavg_baseline --config_path ./config/cpm_fedavg.yaml --clients_config_path ./config/clients/cpm_clients.yaml --pretrained_model_path ./save/pretrained.pth --resume_folder ./save/[2024-05-09-2217]_fedavg_baseline

@REM @REM python cpm_val.py --model_path ./save/[2024-05-09-2217]_fedavg_baseline/best_model.pth --val_mixed_precision
@REM @REM python cpm_val.py --model_path ./save/[2024-05-09-2217]_fedavg_baseline/best_model.pth --val_mixed_precision --patch_label_type benign
@REM @REM python cpm_val.py --model_path ./save/[2024-05-09-2217]_fedavg_baseline/best_model.pth --val_mixed_precision --patch_label_type benign --apply_lobe --apply_aug --crop_size 160 --overlap_ratio 0.1
@REM @REM python cpm_val.py --model_path ./save/[2024-05-09-2217]_fedavg_baseline/best_model.pth --val_mixed_precision --apply_lobe --apply_aug --crop_size 160 --overlap_ratio 0.1
@REM python main.py --exp_name fedavg_baseline_noEma --config_path ./config/cpm_fedavg.yaml --clients_config_path ./config/clients/cpm_clients.yaml --pretrained_model_path ./save/pretrained.pth
@REM python main.py --exp_name test_bug --config_path ./config/cpm_fedavg.yaml --clients_config_path ./config/clients/cpm_clients.yaml --pretrained_model_path ./save/pretrained.pth

@REM python main.py --exp_name fedavg_ssl_baseline_r02 --config_path ./config/cpm_ssl_fedavg.yaml --clients_config_path ./config/clients/cpm_clients_ssl_r02.yaml --pretrained_model_path ./save/pretrained.pth --ssl


@REM python main.py --exp_name fedavg_baseline_onlyLabeled --config_path ./config/cpm_fedavg.yaml --clients_config_path ./config/clients/cpm_clients_labeled.yaml --pretrained_model_path ./save/pretrained.pth
@REM python main.py --exp_name fedavg_baseline_onlyLabeled_r02 --config_path ./config/cpm_fedavg.yaml --clients_config_path ./config/clients/cpm_clients_labeled_r02.yaml --pretrained_model_path ./save/pretrained.pth

@REM python main.py --exp_name fedavg_ssl_baseline_r02 --config_path ./config/cpm_ssl_fedavg.yaml --clients_config_path ./config/clients/cpm_clients_ssl_r02.yaml --pretrained_model_path ./save/pretrained.pth --ssl
@REM python main.py --exp_name fedavg_ssl_baseline_r02 --config_path ./config/cpm_ssl_fedavg.yaml --clients_config_path ./config/clients/cpm_clients_ssl_r02_noPseu.yaml --pretrained_model_path ./save/pretrained.pth --ssl

@REM python cpm_train.py --train_set ./data/pretrained_train.txt --val_set ./data/pretrained_val.txt --test_set ./data/all_client_test.txt --mixed_precision --val_mixed_precision --exp_name PT_NoX4_bs5n8TPr06_posIg5_numNeg-1_lrGa01_iouL4_shapeL1_rot30 --start_val_epoch 300 --warmup_epochs 20 --val_interval 10 --epochs 500 --start_val_epoch 300 --model_class fl_modules.model.cpm_net.cpm_net --ema_warmup_epochs 350 --early_end_epoch 450
@REM python cpm_train.py --train_set ./data/pretrained_train.txt --val_set ./data/pretrained_val.txt --test_set ./data/all_client_test.txt --mixed_precision --val_mixed_precision --exp_name PT_NoX4_bs5n8TPr06_posIg5_numNeg-1_lrGa01_iouL4_shapeL1_rot30 --start_val_epoch 300 --batch_size 4 --warmup_epochs 20 --val_interval 10 --epochs 500 --start_val_epoch 300 --model_class fl_modules.model.cpm_net.cpm_net --ema_warmup_epochs 350 --early_end_epoch 450
0
@REM python cpm_val.py --val_set ./data/client0_test.txt --model_path ./save/[2024-05-29-2028]_fedavg_baseline_onlyLabeled_r02/best_model.pth --val_mixed_precision --apply_aug --apply_lobe --crop_size 160 --overlap_ratio 0.1 --patch_label_type tp
@REM python cpm_val.py --val_set ./data/client1_test.txt --model_path ./save/[2024-05-29-2028]_fedavg_baseline_onlyLabeled_r02/best_model.pth --val_mixed_precision --apply_aug --apply_lobe --crop_size 160 --overlap_ratio 0.1 --patch_label_type tp
@REM python cpm_val.py --val_set ./data/client2_test.txt --model_path ./save/[2024-05-29-2028]_fedavg_baseline_onlyLabeled_r02/best_model.pth --val_mixed_precision --apply_aug --apply_lobe --crop_size 160 --overlap_ratio 0.1 --patch_label_type tp
@REM python cpm_val.py --val_set ./data/client0_test.txt --model_path ./save/[2024-05-29-2028]_fedavg_baseline_onlyLabeled_r02/best_model.pth --val_mixed_precision --apply_aug --apply_lobe --crop_size 160 --overlap_ratio 0.1 --patch_label_type benign
@REM python cpm_val.py --val_set ./data/client1_test.txt --model_path ./save/[2024-05-29-2028]_fedavg_baseline_onlyLabeled_r02/best_model.pth --val_mixed_precision --apply_aug --apply_lobe --crop_size 160 --overlap_ratio 0.1 --patch_label_type benign
@REM python cpm_val.py --val_set ./data/client2_test.txt --model_path ./save/[2024-05-29-2028]_fedavg_baseline_onlyLabeled_r02/best_model.pth --val_mixed_precision --apply_aug --apply_lobe --crop_size 160 --overlap_ratio 0.1 --patch_label_type benign

@REM python cpm_val.py --val_set ./data/client0_test.txt --model_path ./save/[2024-05-29-0024]_fedavg_baseline_onlyLabeled/best_model.pth --val_mixed_precision --apply_aug --apply_lobe --crop_size 160 --overlap_ratio 0.1 --patch_label_type tp
@REM python cpm_val.py --val_set ./data/client1_test.txt --model_path ./save/[2024-05-29-0024]_fedavg_baseline_onlyLabeled/best_model.pth --val_mixed_precision --apply_aug --apply_lobe --crop_size 160 --overlap_ratio 0.1 --patch_label_type tp
@REM python cpm_val.py --val_set ./data/client2_test.txt --model_path ./save/[2024-05-29-0024]_fedavg_baseline_onlyLabeled/best_model.pth --val_mixed_precision --apply_aug --apply_lobe --crop_size 160 --overlap_ratio 0.1 --patch_label_type tp
@REM python cpm_val.py --val_set ./data/client0_test.txt --model_path ./save/[2024-05-29-0024]_fedavg_baseline_onlyLabeled/best_model.pth --val_mixed_precision --apply_aug --apply_lobe --crop_size 160 --overlap_ratio 0.1 --patch_label_type benign

@REM python main.py --exp_name fedavg_onlyLabeled_r02_dropout03 --config_path ./config/cpm_fedavg.yaml --clients_config_path ./config/clients/cpm_clients_labeled_r02.yaml --pretrained_model_path ./save/pretrained.pth

@REM python main.py --exp_name fedavg_onlyLabeled_r02_dropout03_lr2e-3_randSP0911 --config_path ./config/cpm_fedavg_randSP.yaml --clients_config_path ./config/clients/cpm_clients_labeled_r02.yaml --pretrained_model_path ./save/pretrained.pth
@REM python cpm_val.py --model_path ./save/[2024-06-27-0310]_fedavg_onlyLabeled_r02_dropout03_lr2e-3_randSP0911/best_model.pth --val_mixed_precision --apply_aug --apply_lobe --crop_size 160 --overlap_ratio 0.1 --patch_label_type benign

@REM cd D:\workspace\python\lung-nodule-detection-federated-learning
@REM python main.py --exp_name fedavg_onlyLabeled_r02_fixBug_lr2e-3 --config_path ./config/cpm_fedavg.yaml --clients_config_path ./config/clients/cpm_clients_labeled_r02.yaml --pretrained_model_path ./save/[2024-07-09-0007]_PT_fixbug/best/best_froc_mean_recall.pth
@REM python cpm_val.py --val_set ./data/all_client_test.txt --model_path ./save/[2024-07-12-0156]_fedavg_onlyLabeled_r02_fixBug_lr2e-3_randIntensity/best_model.pth --val_mixed_precision --apply_aug --apply_lobe --crop_size 160 --overlap_ratio 0.1 --patch_label_type benign

@REM python main.py --exp_name fedavg_onlyLabeled_r02_fixBug_lr2e-3_randIntensity --config_path ./config/cpm_fedavg.yaml --clients_config_path ./config/clients/cpm_clients_labeled_r02.yaml --pretrained_model_path ./save/[2024-07-09-0007]_PT_fixbug/best/best_froc_mean_recall.pth
@REM python cpm_val.py --val_set ./data/all_client_test.txt --model_path ./save/[2024-07-13-0120]_fedAvg_all_fixBug_randI/best_model.pth --val_mixed_precision --apply_aug --apply_lobe --crop_size 160 --overlap_ratio 0.1 --patch_label_type benign

@REM python main.py --exp_name fedAvg_all_fixBug_randI --config_path ./config/cpm_fedavg.yaml --clients_config_path ./config/clients/cpm_clients.yaml --pretrained_model ./save/[2024-07-09-0007]_PT_fixbug/best/best_froc_mean_recall.pth

@REM python cpm_train.py --train_set ./data/pretrained_train.txt --val_set ./data/pretrained_val.txt --test_set ./data/all_client_test.txt --batch_size 4 --num_samples 8 --tp_ratio 0.6 --mixed_precision --val_mixed_precision --exp_name PT_fixbug --warmup_epochs 20 --val_interval 10 --epochs 500 --start_val_epoch 320 --model_class fl_modules.model.cpm_net.cpm_net --ema_warmup_epochs 350 --early_end_epoch 450

@REM python cpm_val.py --val_set ./data/client0_test.txt --model_path ./save/[2024-07-09-0157]_fedAvg_all_fixBug/best_model.pth --val_mixed_precision --apply_aug --apply_lobe --crop_size 160 --overlap_ratio 0.1 --patch_label_type benign
@REM python cpm_val.py --val_set ./data/client1_test.txt --model_path ./save/[2024-07-09-0157]_fedAvg_all_fixBug/best_model.pth --val_mixed_precision --apply_aug --apply_lobe --crop_size 160 --overlap_ratio 0.1 --patch_label_type benign
@REM python cpm_val.py --val_set ./data/client2_test.txt --model_path ./save/[2024-07-09-0157]_fedAvg_all_fixBug/best_model.pth --val_mixed_precision --apply_aug --apply_lobe --crop_size 160 --overlap_ratio 0.1 --patch_label_type benign

@REM python cpm_val.py --model_path ./save/[2024-07-09-0402]_fedavg_onlyLabeled_r02_fixBug_lr2e-3/best_model.pth --val_mixed_precision --apply_aug --apply_lobe --crop_size 160 --overlap_ratio 0.1 --patch_label_type benign
@REM python cpm_val.py --model_path ./save/[2024-07-09-0157]_fedAvg_all_fixBug/best_model.pth --val_mixed_precision --apply_aug --apply_lobe --crop_size 160 --overlap_ratio 0.1 --patch_label_type benign
@REM python cpm_val.py --model_path ./save/[2024-07-12-0156]_fedavg_onlyLabeled_r02_fixBug_lr2e-3_randIntensity/best_model.pth --val_mixed_precision --apply_aug --apply_lobe --crop_size 160 --overlap_ratio 0.1 --patch_label_type benign
@REM python cpm_val.py --model_path ./save/[2024-07-18-1231]_fedavg_onlyLabeled_r01_fixBug_lr2e-3_randI/best_model.pth --val_mixed_precision --apply_aug --apply_lobe --crop_size 160 --overlap_ratio 0.1 --patch_label_type benign
@REM python main.py --exp_name _fedavg_onlyLabeled_r01_fixBug_lr2e-3_randI --config_path ./config/cpm_fedavg.yaml --clients_config_path ./config/clients/cpm_clients_labeled.yaml --pretrained_model ./save/[2024-07-14-1422]_PT_fixbug/best/best_froc_mean_recall.pth

@REM python cpm_val.py --model_path ./save/[2024-07-18-1231]__fedavg_onlyLabeled_r01_fixBug_lr2e-3_randI/best_model.pth --val_mixed_precision --apply_aug --apply_lobe --crop_size 160 --overlap_ratio 0.1 --patch_label_type benign
@REM python cpm_val.py --model_path ./save/[2024-07-18-0934]_fedavg_ssl_baseline_r02_fg06_bg04_lr2e-3_enhanceSmall/best_model.pth --val_mixed_precision --apply_aug --apply_lobe --crop_size 160 --overlap_ratio 0.1 --patch_label_type benign --load_teacher_model

@REM python cpm_val.py --model_path ./save/[2024-07-12-0156]_fedavg_onlyLabeled_r02_fixBug_lr2e-3_randIntensity/best_model.pth --val_mixed_precision --apply_aug --apply_lobe --crop_size 160 --overlap_ratio 0.1 --patch_label_type benign

@REM python cpm_val.py --val_set ./data/client0_test.txt --model_path ./save/[2024-07-19-2146]_fedavg_ssl_fromFLR02/best_model.pth --val_mixed_precision --apply_aug --apply_lobe --crop_size 160 --overlap_ratio 0.1 --patch_label_type benign --load_teacher_model
@REM python cpm_val.py --val_set ./data/client1_test.txt --model_path ./save/[2024-07-19-2146]_fedavg_ssl_fromFLR02/best_model.pth --val_mixed_precision --apply_aug --apply_lobe --crop_size 160 --overlap_ratio 0.1 --patch_label_type benign --load_teacher_model


@REM python cpm_val.py --model_path ./save/[2024-07-19-2146]_fedavg_ssl_fromFLR02/best_model.pth --val_mixed_precision --apply_aug --apply_lobe --crop_size 160 --overlap_ratio 0.1 --patch_label_type benign


@REM python cpm_val.py --model_path ./save/[2024-07-19-2146]_fedavg_ssl_fromFLR02/best_model.pth --val_mixed_precision --patch_label_type benign --lo
@REM python cpm_val.py --model_path ./save/[2024-07-19-2146]_fedavg_ssl_fromFLR02/best_model.pth --val_mixed_precision --patch_label_type benign --lo

@REM python cpm_val.py --val_set ./data/client0_test.txt --model_path ./save/[2024-07-19-2146]_fedavg_ssl_fromFLR02/best_model.pth --val_mixed_precision --apply_aug --apply_lobe --crop_size 160 --overlap_ratio 0.1 --patch_label_type benign --load_teacher_model
@REM python cpm_val.py --val_set ./data/client1_test.txt --model_path ./save/[2024-07-19-2146]_fedavg_ssl_fromFLR02/best_model.pth --val_mixed_precision --apply_aug --apply_lobe --crop_size 160 --overlap_ratio 0.1 --patch_label_type benign --load_teacher_model
@REM python cpm_val.py --val_set ./data/client2_test.txt --model_path ./save/[2024-07-19-2146]_fedavg_ssl_fromFLR02/best_model.pth --val_mixed_precision --apply_aug --apply_lobe --crop_size 160 --overlap_ratio 0.1 --patch_label_type benign --load_teacher_model


@REM python cpm_val.py --model_path ./save/[2024-07-19-2146]_fedavg_ssl_fromFLR02/best_model.pth --val_mixed_precision --apply_aug --apply_lobe --crop_size 160 --overlap_ratio 0.1 --patch_label_type benign --load_teacher_model --val_iou_threshold 0.8


@REM python cpm_val.py --model_path ./save/[2024-07-24-0108]_fedavg_onlyLabeled_fromFLR02/best_model.pth --val_mixed_precision --apply_lobe --apply_aug --crop_size 160 --overlap_ratio 0.1 --patch_label_type benign
@REM python main.py --exp_name fedavg_onlyLabeled_r02_fixBug_lr2e-3_randIntensity_fromFLR02_twoTimes --config_path ./config/cpm_fedavg.yaml --clients_config_path ./config/clients/cpm_clients_labeled_r02.yaml --pretrained_model ./save/[2024-07-24-0108]_fedavg_onlyLabeled_r02_fixBug_lr2e-3_randIntensity_fromFLR02/best_model.pth

@REM python main.py --exp_name fedavg_ssl_baseline_r02_fg06_bg04_lr1e-3_enhanceSmall_fromFSSLR02_twoTimes --config_path ./config/cpm_ssl_fedavg_enhance_small.yaml --clients_config_path ./config/clients/cpm_clients_ssl_r02_fromFSSLR02_twoTimes.yaml --pretrained_model ./save/best_model_from_FL_r02_SSL_2times_r02.pth --ssl

@REM python main.py --exp_name fedavg_test --config_path ./config/cpm_ssl_fedavg_enhance_small.yaml --clients_config_path ./config/clients/cpm_clients_ssl_r02_test.yaml --pretrained_model ./save/[2024-07-14-1422]_PT_fixbug/best/best_froc_mean_recall.pth --ssl


@REM python main.py --exp_name fedavg_test --config_path ./config/cpm_ssl_fedavg_enhance_small.yaml --clients_config_path ./config/clients/cpm_clients_ssl_r02_test.yaml --pretrained_model ./save/[2024-07-14-1422]_PT_fixbug/best/best_froc_mean_recall.pth --ssl
python main.py --exp_name fedavg_test --config_path ./config/cpm_ssl_fedavg_enhance_small_test.yaml --clients_config_path ./config/clients/cpm_clients_ssl_r02_test_Pseu.yaml --pretrained_model ./save/[2024-07-14-1422]_PT_fixbug/best/best_froc_mean_recall.pth --ssl