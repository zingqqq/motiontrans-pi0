# NOTE: these unset are important; if not done AWS_PROFILE set in launcher will be ignored
unset AWS_ACCESS_KEY_ID
unset AWS_SECRET_ACCESS_KEY
unset AWS_SESSION_TOKEN
unset AWS_PROFILE
unset WANDB_API_KEY
unset HF_TOKEN

cd ..
source startup_stuff.sh

INSTANCE_COUNT=$1
task_name=$2
NAME=$3
QUEUE_NAME=${4:-cv-p5en}
LOCAL=${5:-0}
BUILD_TYPE=${6:-full}
VERSION=${7:-271}
PRIORITY=${8:-20}

echo "INSTANCE_COUNT: $INSTANCE_COUNT"
echo "NAME: $NAME"
echo "QUEUE_NAME: $QUEUE_NAME"
echo "BUILD_TYPE: $BUILD_TYPE"
echo "VERSION: $VERSION"
echo "PRIORITY: $PRIORITY"

USER_NAME=kbhatch


# PROFILE=manip-cluster
PROFILE=default
REGION=us-west-2
ARN=arn:aws:iam::124224456861:role/SageMaker-SageMakerAllAccess-us-west-2
S3_REMOTE_SYNC=s3://tri-ml-sandbox-16011-us-west-2-datasets/kylehatch/video_benchmarking_project/sagemaker/s3_remote_sync



export WANDB_MODE=online ### DEBUG ###

# task_name=PutGreenAppleOnSaucer
exp_name=$NAME                       # the description of the experiment target
repo_id="0703_pi_cotrain"                    # the repo used for dataset and norm-stat
# dataset_path="/data/zarr_data/zarr_data_robot/robot_mix+$task_name+.zarr"
# dataset_path="/opt/ml/input/data/training/zarr_data/zarr_data_robot/robot_mix+$task_name+.zarr"
# dataset_path="/opt/ml/input/data/training/zarr_data/zarr_data_robot/robot_mix+$task_name+.zarr"
# dataset_path="/opt/ml/input/data/robot_mix+$task_name+.zarr"
# dataset_path="/opt/ml/input/data/zarr_data_robot/robot_mix+$task_name+.zarr"
dataset_path="/opt/ml/input/data/zarr_data_robot_no_corrupted_episodes_no_idle_wrist/robot_mix+$task_name+.zarr"

checkpoint_base_dir="/checkpoints_pi0/pretrained_ckpts/$task_name/"
assets_base_dir="/checkpoints_pi0/assets/$task_name/"
# export HF_HOME="/cache/huggingface"
export OPENPI_DATA_HOME="/checkpoints_pi0/openpi"
export HF_LEROBOT_HOME="/checkpoints_pi0/lerobot"   
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
# export WANDB_DIR="/wandb_pi0/$task_name"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

logging_time=$(date "+%d-%H.%M.%S")
now_seconds="${logging_time: -8}"
now_date=$(date "+%Y.%m.%d")

alpha=0.5
proprioception_droprate=0.0
num_devices=8



if [ "$LOCAL" -eq 1 ]; then
    single_batch_size=2
    single_val_batch_size=2
    export WANDB_MODE=offline
else
    single_batch_size=24
    single_val_batch_size=4
fi

batch_size=$((num_devices * single_batch_size))
echo batch_size $batch_size

num_train_steps=150001
keep_period=7500
log_interval=250
save_interval=7500
val_interval=500
max_token_len=150
val_batch_size=$((num_devices * single_val_batch_size))
echo val_batch_size $val_batch_size

python3 -c "import os; [print(k, '=', v[:10]+'...') for k,v in os.environ.items() if len(v) > 20]"

make docker_build 
bash upload_docker.sh 

AWS_DEFAULT_REGION=${REGION}                        \
    python3 launch_sagemaker.py    \
    --config pi0_droid_motiontrans \
    --user=${USER_NAME}                             \
    --task_name=$task_name \
    --base-job-name=${USER_NAME}-motiontrans-pi0         \
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
    --model_max_token_len ${max_token_len}          \
    --instance-count=${INSTANCE_COUNT}              \
    --profile=${PROFILE}                            \
    --region=${REGION}                              \
    --arn=${ARN}                                    \
    --s3-remote-sync=${S3_REMOTE_SYNC}              \
    --priority=${PRIORITY}                          \
    --queue=${QUEUE_NAME}       		            \
    --name=${NAME}                                  \
    --version=${VERSION}                            \
    --build-type=${BUILD_TYPE}                      \
    --local=$LOCAL




# bash launch_sagemaker.sh 1 BimanualPlaceAppleFromBowlOnCuttingBoard BimanualPlaceAppleFromBowlOnCuttingBoard-b24v4-fsdp2-nocorrupted-noidle ml-p5
# bash launch_sagemaker.sh 1 PutBananaOnSaucer PutBananaOnSaucer-b24v4-fsdp2-nocorrupted-noidle ml-p5
# bash launch_sagemaker.sh 1 PutKiwiInCenterOfTable PutKiwiInCenterOfTable-b24v4-fsdp2-nocorrupted-noidle ml-p5