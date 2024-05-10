python train_stage1.py --extra_info CO0_local_mixP_more_slice --train_set client0_train.txt --val_set client0_val.txt --test_set client0_test.txt --num_epoch 50  --num_workers 4 --pin_memory --mixed_precision --pretrained_model_path "./save/[2024-01-15-2351]_pretrained_mixP_more_slice/best.pth"
python train_stage1.py --extra_info CO1_local_mixP_more_slice --train_set client1_train.txt --val_set client1_val.txt --test_set client1_test.txt --num_epoch 50  --num_workers 4 --pin_memory --mixed_precision --pretrained_model_path "./save/[2024-01-15-2351]_pretrained_mixP_more_slice/best.pth"
python train_stage1.py --extra_info CO2_local_mixP_more_slice --train_set client2_train.txt --val_set client2_val.txt --test_set client2_test.txt --num_epoch 50  --num_workers 4 --pin_memory --mixed_precision --pretrained_model_path "./save/[2024-01-15-2351]_pretrained_mixP_more_slice/best.pth"


python main.py --exp_name test_bug --config_path ./config/cpm_fedavg.yaml --clients_config_path ./config/clients/cpm_clients.yaml
python main.py --exp_name fedavg_baseline --config_path ./config/cpm_fedavg.yaml --clients_config_path ./config/clients/cpm_clients.yaml 


python cpm_train.py --train_set ./data/pretrained_train.txt --val_set ./data/pretrained_val.txt --test_set ./data/all_client_test.txt --mixed_precision --val_mixed_precision --exp_name PT_NoX4_bs5n8TPr06_posIg5_numNeg-1_lrGa01_iouL4_shapeL1_rot30 --start_val_epoch 300 --warmup_epochs 20 --val_interval 10 --epochs 500 --start_val_epoch 300 --model_class fl_modules.model.cpm_net.cpm_net --ema_warmup_epochs 300 --early_end_epoch 400

python main.py --exp_name fedavg_baseline --config_path ./config/cpm_fedavg.yaml --clients_config_path ./config/clients/cpm_clients.yaml --pretrained_model_path ./save/pretrained.pth