#!/bin/bash
#SBATCH --job-name=ACT
#SBATCH --output=result.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=20000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anxingxiao@gmail.com
#SBATCH --gres=gpu:1
#SBATCH --nodelist=crane5

nvidia-smi
source activate IL

task_name="open_fridge"
# python -u imitate_episodes.py --task_name open_fridge --ckpt_dir check_point/${task_name} --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 128 --dim_feedforward 3200 --num_steps 2000  --lr 1e-5 --temporal_agg --seed 0
python act_plus_plus_miscs/imitate_episodes.py --task_name ${task_name} --ckpt_dir check_point/${task_name} --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 128 --dim_feedforward 3200 --num_steps 5000  --lr 1e-5 --temporal_agg --seed 0
# python imitate_episodes.py --task_name open_fridge --ckpt_dir check_point/open_fridge --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 128 --dim_feedforward 3200 --num_steps 2000  --lr 1e-5 --temporal_agg --seed 0
# accelerate launch --config_file=./configs/deepspeed_zero{1,2,3}.yaml --num_processes 2 imitate_episodes.py --task_name open_fridge --ckpt_dir check_point/open_fridge --policy_class Diffusion --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 32 --dim_feedforward 3200 --num_steps 2000  --lr 1e-5 --temporal_agg --seed 0
