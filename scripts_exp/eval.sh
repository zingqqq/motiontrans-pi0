# bash scripts_exp/eval.sh

exp_name="0712_pi0_eval_lrsc"                          # the description of the experiment target
repo_id="0703_pi_cotrain"
dataset_path="/data/zeqingwang/vis_test/zarr_data/zarr_data_human|/data/zeqingwang/vis_test/zarr_data/zarr_data_robot"  # link different folders with |
policy_dir="/data/kylehatch/checkpoints_pi0/pretrained_ckpts/BimanualPlaceAppleFromBowlOnCuttingBoard/pi0_droid_motiontrans/2026.04.13_19.32.32_0703_pi_cotrain_BimanualPlaceAppleFromBowlOnCuttingBoard-p5-b24v4-fsdp2/30000"
max_token_len=100

logging_time=$(date "+%d-%H.%M.%S")
now_seconds="${logging_time: -8}"
now_date=$(date "+%Y.%m.%d")
num_devices=2
single_val_batch_size=1
val_batch_size=$((num_devices * single_val_batch_size))
echo val_batch_size $val_batch_size

checkpoint_base_dir="checkpoints_pi0/pretrained_ckpts"
assets_base_dir="checkpoints_pi0/assets"
export HF_HOME="/cephfs/shared/yuanchengbo/hub/huggingface"
export OPENPI_DATA_HOME="checkpoints_pi0/openpi"
export HF_LEROBOT_HOME="checkpoints_pi0/lerobot"  
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"   
export CUDA_VISIBLE_DEVICES=2,7                                        
# export WANDB_BASE_URL=https://api.bandw.top

uv run scripts/eval.py pi0_droid_motiontrans \
--exp-name=${exp_name} \
--no-single_arm \
--checkpoint_base_dir=${checkpoint_base_dir} \
--policy_dir=${policy_dir} \
--assets_base_dir=${assets_base_dir} \
--repo_id=${repo_id} \
--dataset_path=${dataset_path} \
--state_down_sample_steps 2 \
--action_down_sample_steps 2 \
--proprioception_rep "relative" \
--action_rep "relative" \
--use_val_dataset \
--val_batch_size=$val_batch_size \
--model.max_token_len ${max_token_len} \
