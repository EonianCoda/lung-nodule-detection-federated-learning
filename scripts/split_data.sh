cd ..
python split_data.py --series_txt_path ./data/all.txt --use_pretrained --pretrained_train_val_ratios 0.9 0.1 --seed 1029 --n_clusters 20 --train_val_test_ratios 0.8 0.1  0.1 --use_unlabel