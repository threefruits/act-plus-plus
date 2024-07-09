export PYTHONPATH=$PYTHONPATH:/data/home/siwei/Documents/Github/TransformerIL2


#srun -u -o "Main.out" --mem=220000 --gres=gpu:8 --cpus-per-task=64 -w "crane1" --job-name "Main" python -u imitate_episodes.py --task_name sim_transfer_cube_scripted --ckpt_dir check_point --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 32 --dim_feedforward 3200 --num_epochs 2000  --lr 1e-5 --temporal_agg --seed 0 &
#srun -u --mem=150000 --gres=gpu:8 --cpus-per-task=32 -w "crane1" --job-name "Main" python -u imitate_episodes.py --task_name sim_transfer_cube_scripted --ckpt_dir check_point --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 256 --dim_feedforward 3200 --num_epochs 2000  --lr 1e-5 --temporal_agg --seed 0 &

#srun -u -o "Main_no_stats.out" --mem=220000 --gres=gpu:8 --cpus-per-task=48 -w "crane0" --job-name "Main2" python -u imitate_episodes.py --task_name grasp_milk --ckpt_dir check_point_grasp_milk --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 32 --dim_feedforward 3200 --num_epochs 2000  --lr 1e-5 --temporal_agg --seed 0 &
#srun -u -o "open_microwave.out" --mem=100000 --gres=gpu:4 --cpus-per-task=32 --exclude "crane2" --job-name "Main2" python -u imitate_episodes.py --task_name open_microwave --ckpt_dir open_microwave --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 32 --dim_feedforward 3200 --num_epochs 2000  --lr 1e-5 --temporal_agg --seed 0 &

#task_names=("open_fridge" "grasp_food_from_fridge" "placeon_food_counter" "close_fridge"
#            "open_microwave" "grasp_food_from_table" "placein_microwave" "close_microwave"
#            "switchon_microwave" "grasp_food_from_microwave" "placeon_food_dinning_table")
task_names=("placeon_food_counter")

MASTER_PORT_VAR=36688

for task_name in "${task_names[@]}"
do
    MASTER_PORT=$MASTER_PORT_VAR srun -u -o "${task_name}.out" --mem=50000 --gres=gpu:2 --cpus-per-task=20 --exclude "crane2" --job-name "${task_name}" python -u imitate_episodes.py --task_name ${task_name} --ckpt_dir check_point/${task_name} --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 32 --dim_feedforward 3200 --num_epochs 2000  --lr 1e-5 --temporal_agg --seed 0 &
    MASTER_PORT_VAR=$((MASTER_PORT_VAR+1))
    echo $MASTER_PORT_VAR
done