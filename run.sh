#!/usr/bin/env bash
python main.py --output_path rebuttal  --seed 2022 --save_path RGCF_IR  --mode meta-train  --load_path RGCF_IR/checkpoint/max_corr.pt --dataset_on True --mixup_on True --gpu 4 --feature_type full --num_episodes 3000 --patience 100 --num_train_updates 2 --task ir --performance_name recall --data_path data_process/test_best_rk.csv --num_inner_tasks 4 --lamda 0.6

python main.py --output_path rebuttal  --seed 2022 --save_path RGCF_IR  --mode search  --load_path RGCF_IR/checkpoint/max_corr.pt --dataset_on True --mixup_on True --gpu 4 --feature_type full --num_episodes 3000 --patience 100 --num_train_updates 2 --task ir --performance_name recall --data_path data_process/test_best_rk.csv --num_inner_tasks 4 --lamda 0.6


python main.py --output_path rebuttal  --seed 2022 --save_path RGCF_IR --mode meta-train  --load_path RGCF_IR/checkpoint/max_corr.pt --dataset_on True --mixup_on True --gpu 4 --feature_type full --num_episodes 1500 --patience 300 --num_train_updates 2 --task rp --data_path data_process/test_best.csv --performance_name rmse --num_inner_tasks 4 --lamda 0.8

python main.py --output_path rebuttal  --seed 2022 --save_path RGCF_IR --mode search  --load_path RGCF_IR/checkpoint/max_corr.pt --dataset_on True --mixup_on True --gpu 4 --feature_type full --num_episodes 1500 --patience 300 --num_train_updates 2 --task rp --data_path data_process/test_best.csv --performance_name rmse --num_inner_tasks 4 --lamda 0.8