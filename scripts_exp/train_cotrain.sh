# bash scripts_exp/train_cotrain.sh

task_name=PutGreenAppleOnSaucer

exp_name=$task_name                       # the description of the experiment target
repo_id="0703_pi_cotrain"                    # the repo used for dataset and norm-stat
# dataset_path="/data/zeqingwang/vis_test/zarr_data/zarr_data_human|/data/zeqingwang/vis_test/empty"  # link different folders with |
# dataset_path="/data/zeqingwang/vis_test/zarr_data/zarr_data_robot"
dataset_path="/data/zarr_data/zarr_data_robot/robot_mix+$task_name+.zarr"

checkpoint_base_dir="/checkpoints_pi0/pretrained_ckpts/$task_name/"
assets_base_dir="/checkpoints_pi0/assets/$task_name/"
# export HF_HOME="/cephfs/shared/yuanchengbo/hub/huggingface"
export HF_HOME="/cache/huggingface"
export OPENPI_DATA_HOME="/checkpoints_pi0/openpi"
export HF_LEROBOT_HOME="/checkpoints_pi0/lerobot"   
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
# export CUDA_VISIBLE_DEVICES=2,7
# export XLA_FLAGS=--xla_force_host_platform_device_count=2
export WANDB_MODE="offline" ###===###
export WANDB_DIR="/wandb_pi0/$task_name"


logging_time=$(date "+%d-%H.%M.%S")
now_seconds="${logging_time: -8}"
now_date=$(date "+%Y.%m.%d")

alpha=0.5
proprioception_droprate=0.0
num_devices=8
single_batch_size=1
batch_size=$((num_devices * single_batch_size))
echo batch_size $batch_size

num_train_steps=150001
keep_period=7500
# log_interval=250
# save_interval=7500
# val_interval=500
log_interval=12500 ### DEBUG ###
save_interval=100
val_interval=250050

max_token_len=150

single_val_batch_size=1
val_batch_size=$((num_devices * single_val_batch_size))
echo val_batch_size $val_batch_size

# we downsample data-obs and action from 20 Hz to 10 Hz, since pi0 inference only support for 10Hz inference speed. 

 echo "task_name: $task_name"

aws s3 sync s3://tri-ml-sandbox-16011-us-west-2-datasets/kylehatch/video_benchmarking_project/checkpoints_pi0/ /checkpoints_pi0 \
 --exclude "*" \
 --include "assets/$task_name/*"


# ======== pi0 cocktail  =====
# WANDB_DISABLED=True 
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi0_droid_motiontrans \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python3 -u scripts/train.py pi0_droid_motiontrans \
--exp-name="${now_date}_${now_seconds}_${repo_id}_${exp_name}" \
--alpha=${alpha} \
--single_arm 0 \
--checkpoint_base_dir=${checkpoint_base_dir} \
--assets_base_dir=${assets_base_dir} \
--batch-size=$batch_size \
--repo_id=${repo_id} \
--dataset_path=${dataset_path} \
--state_down_sample_steps 2 \
--action_down_sample_steps 2 \
--proprioception_rep "relative" \
--action_rep "relative" \
--proprioception_droprate ${proprioception_droprate} \
--use_val_dataset 1 \
--val_batch_size=$val_batch_size \
--num_train_steps ${num_train_steps} \
--keep_period ${keep_period} \
--log_interval ${log_interval} \
--save_interval ${save_interval} \
--val_interval ${val_interval} \
--model.max_token_len ${max_token_len} \
# --create_train_val_split \
# --compute-norm-stats