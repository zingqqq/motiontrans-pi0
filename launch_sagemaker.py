import argparse
import time
import os
import subprocess
import logging
from datetime import datetime
from pathlib import Path

# logging.basicConfig(level=logging.DEBUG)

import boto3
import yaml
from botocore.config import Config
from sagemaker import Session as sm_Session
from sagemaker.pytorch import PyTorch
from sagemaker.estimator import Estimator
from sagemaker.inputs import FileSystemInput
from sagemaker.debugger import ProfilerConfig, FrameworkProfile, ProfilerRule, rule_configs

from rich import print 

try:
    # from sagemaker.batch_queueing.queue import Queue
    from sagemaker.aws_batch.training_queue import TrainingQueue as Queue
    print(f"Loading SageMaker batch queueing.")
    is_sm_queue = True
except Exception as e:
    print(f"Could not load SageMaker batch queueing: {e}.")
    print(f"Trying to install")
    os.system("bash scripts/setup_sm_batch.sh")
    from sagemaker.batch_queueing.queue import Queue
    print(f"Loading SageMaker batch queueing.")
    is_sm_queue = True
except:
    is_sm_queue = False

is_sm_queue = True

NAME = "motiontrans-pi0"
INSTANCE_MAPPER = {
    "p4d": "ml.p4d.24xlarge",
    "p4de": "ml.p4de.24xlarge",
    "p5": "ml.p5.48xlarge",
    "p5en": "ml.p5en.48xlarge",
    "p6": "ml.p6.48xlarge",
    "g6e": "ml.g6e.48xlarge",
    "g6e-small": "ml.g6e.12xlarge",
    "g5": "ml.g5.48xlarge",
    "g5-small": "ml.g5.24xlarge"
}


def run_command(command):
    print(f"=> {command}")
    subprocess.run(command, shell=True, check=True)


def get_image(user, instance_type, version="271", build_type="full", profile="default", region="us-east-1"):
    print(f"Building image for user {user}, instance_type {instance_type}, version {version}, build_type {build_type}")
    os.environ["AWS_PROFILE"] = f"{profile}"
    account = subprocess.getoutput(
        f"aws --region {region} --profile {profile} sts get-caller-identity --query Account --output text"
    )
    docker_dir = Path(__file__).parent
    if instance_type in INSTANCE_MAPPER.keys():
        algorithm_name = f"{user}-{NAME}-{version}"
        dockerfile_base = docker_dir / f"Dockerfile_{version}"
        dockerfile_update = docker_dir / "Dockerfile_update"
    else:
        raise ValueError(f"Unknown instance_type: {instance_type}")
    fullname = f"{account}.dkr.ecr.{region}.amazonaws.com/{algorithm_name}:latest"
    if build_type is None:
        return fullname
    
    print(f"algorithm_name: {algorithm_name}")
    print(f"dockerfile_base: {dockerfile_base}")
    print(f"dockerfile_update: {dockerfile_update}")

    login_cmd = f"aws ecr get-login-password --region {region} --profile {profile} | docker login --username AWS --password-stdin"

    # NOTE (Dian): no "Dockerfile_update" build is needed since the code copy happens late in the SM dockerfile,
    # therefore docker will only update top layers in case of any code changes. the update method might
    # lead to max depth reached issues due to recursive basing.
    print("Building container")
    if build_type == "full":
        print("Building container")
        commands = [
            # Log in to Sagemaker account to get image.
            f"{login_cmd} 763104351884.dkr.ecr.{region}.amazonaws.com",
            f"docker build -f {dockerfile_base} --build-arg AWS_REGION={region} -t {algorithm_name} .",
            f"docker tag {algorithm_name} {fullname}",
            f"{login_cmd} {fullname}",
            (
                f"aws --region {region} --profile {profile} ecr describe-repositories --repository-names {algorithm_name} || "
                f"aws --region {region} --profile {profile} ecr create-repository --repository-name {algorithm_name}"
            ),
        ]
        
    elif build_type == "update":
        print("Updating container")
        commands = [
            f"docker build -f {dockerfile_update} --build-arg BASE_DOCKER={algorithm_name} -t {algorithm_name} .",
            f"docker tag {algorithm_name} {fullname}",
            f"{login_cmd} {fullname}",
        ]
    else:
        raise ValueError(f"Unknown build_type: {build_type}")

    # Create command, making sure to exit if any part breaks.
    command = "\n".join([f"{x} || exit 1" for x in commands])
    run_command(command)
    run_command(f"docker push {fullname}")
    print("Sleeping for 5 seconds to ensure push succeeded")
    time.sleep(5)
    return fullname


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-type", default="full")
    # parser.add_argument("--local", action="store_true")
    parser.add_argument("--local", type=int, default=0)
    parser.add_argument("--user", required=True, help="User name")
    parser.add_argument("--entry_point", type=str, default="scripts/train.py")
    
    # parser.add_argument("--config", help="config base", required=True)
    # parser.add_argument("--hydra_cfg", help="hydra groups to override with", type=str, default=None)
    # parser.add_argument("--experiment", help="which experiment to run", type=str, default=None)

    # AWS profile args
    parser.add_argument("--region", default="us-west-2", help="AWS region")
    parser.add_argument("--profile", default="default", help="AWS profile to use")
    parser.add_argument("--arn", default=None, help="If None, reads from SAGEMAKER_ARN env var")
    parser.add_argument(
        "--s3-remote-sync", default=None, help="S3 path to sync to. If none, reads from S3_REMOTE_SYNC env var"
    )

    # Instance args
    parser.add_argument("--instance-count", default=1, type=int, help="Number of instances")
    parser.add_argument("--instance-type", default="p4de", choices=list(INSTANCE_MAPPER.keys()))
    parser.add_argument("--version", default="271", type=str, help="Choose from: (271, 271-2stage)")
    parser.add_argument("--spot-instance", action="store_true")

    parser.add_argument('--base-job-name', type=str)
    parser.add_argument('--input-source', choices=['s3', 'lustre', 'local'], default='s3')

    # Jobs Queue
    parser.add_argument("--fss-identifier", default="default", help="Share identifier for FSS queue")
    parser.add_argument("--priority", default=5, type=int, help="Priority of the job")
    parser.add_argument("--queue", type=str, default='ml', help="Job queue")
    parser.add_argument("--name", type=str, default=None)

    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    # parser.add_argument("--no-single_arm", type=int, default=None)
    parser.add_argument("--single_arm", type=int, default=None)
    parser.add_argument("--checkpoint_base_dir", type=str, default=None)
    parser.add_argument("--assets_base_dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--repo_id", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--state_down_sample_steps", type=int, default=None)
    parser.add_argument("--action_down_sample_steps", type=int, default=None)
    parser.add_argument("--proprioception_rep", type=str, default=None)
    parser.add_argument("--action_rep", type=str, default=None)
    parser.add_argument("--proprioception_droprate", type=float, default=None)
    parser.add_argument("--use_val_dataset", type=int, default=None)
    parser.add_argument("--val_batch_size", type=int, default=None)
    parser.add_argument("--num_train_steps", type=int, default=None)

    parser.add_argument("--keep_period", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--val_interval", type=int, default=None)
    parser.add_argument("--model_max_token_len", type=int, default=None)

    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--config", type=str, default=None, help="Positional config name passed to train.py (e.g. pi0_droid_motiontrans)")
    # parser.add_argument("--local-dataset-path", type=str, default=None, help="Local dataset path for local mode (uses file:// URI)")


    args = parser.parse_args()

    ### CLAUDE ### Skip cloud queue resolution when running locally
    if not args.local:
        # Map simple queue aliases to the actual FSS queue names and set instance_type appropriately
        QUEUE_MAP = {
            # [aliases]        : (fss-queue-name, instance_type)
            ('mvdp', 'mvdp-p4de',):   ("fss-mvdp-p4de-24xlarge-us-west-2",   'p4de'),
            ('vla', 'vla-p4de',):     ("fss-vla-p4de-24xlarge-us-west-2",    'p4de'),
            ('ml', 'ml-p4de',):            ("fss-ml-p4de-24xlarge-us-west-2",     'p4de'),
            ('ml-p5',):              ("fss-ml-p5-48xlarge-us-west-2",       'p5'),
            ('testing', 'testing-p5',):   ("fss-testing-p5-48xlarge-us-west-2", 'p5'),
            ('testing-p6',):         ("fss-testing-p6-b200-48xlarge-us-west-2", 'p6'),
            ('cv', 'cv-p5en',):            ("fss-cv-ml-p5en-48xlarge-us-west-2",  'p5en'),
        }

        # Find the matching FSS queue
        found = False
        for aliases, (queue_name, instance_type) in QUEUE_MAP.items():
            if args.queue in aliases:
                args.queue = queue_name
                args.instance_type = instance_type
                found = True
                break

        if not found:
            raise ValueError(f'Invalid queue name {args.queue}')
    ### END CLAUDE ###


    main_after_setup_move(args)


def main_after_setup_move(args):
    if args.arn is None:
        assert "SAGEMAKER_ARN" in os.environ, "Please specify --arn or set the SAGEMAKER_ARN environment variable"
        args.arn = os.environ["SAGEMAKER_ARN"]
    
    if args.s3_remote_sync is None:
        assert (
            "S3_REMOTE_SYNC" in os.environ
        ), "Please specify --s3-remote-sync or set the S3_REMOTE_SYNC environment variable"
        args.s3_remote_sync = os.environ["S3_REMOTE_SYNC"]


    print(f"[yellow]args.local:", args.local)

    ### CLAUDE ### Skip ECR build/push for local runs; just compute the image name
    # image_uri = get_image(
    #     args.user,
    #     args.instance_type,
    #     args.version,
    #     region=args.region,
    #     # build_type=None if args.local else args.build_type,
    #     build_type=args.build_type,
    #     profile=args.profile,
    # )
    image_uri = "124224456861.dkr.ecr.us-west-2.amazonaws.com/openpi_server:latest"
    ### END CLAUDE ###
    
    ### CLAUDE ### Force local runs to use estimator.fit(), not the cloud queue
    if args.local:
        is_sm_queue = False
    # elif args.instance_type.startswith("g"):
    #     # g5, g6 series don't go into queue
    #     is_sm_queue = False
    else:
        is_sm_queue = True
    ### END CLAUDE ###

    ##########
    # Create session and make sure of account and region
    ##########
    sagemaker_session = sm_Session(
        boto_session=boto3.session.Session(
            region_name=args.region,
            profile_name=args.profile
        )
    )

    if args.local:
        from sagemaker.local import LocalSession
        sagemaker_session = LocalSession()

    role = args.arn
    # provide a pre-existing role ARN as an alternative to creating a new role
    role_name = role.split(["/"][-1])

    # client = boto3.client("sts", config=boto3_config)
    # account = client.get_caller_identity()["Account"]
    account = '124224456861' # client.get_caller_identity()["Account"]
    # account = subprocess.getoutput(
    #     f"aws --region {args.region} --profile {args.profile} sts get-caller-identity --query Account --output text"
    # )

    # session = boto3.session.Session()
    session = boto3.session.Session(region_name=args.region)
    region = session.region_name

    ##########
    # Configure the training
    ##########
    base_job_name = args.base_job_name # f"{args.user.replace('.', '-')}-{NAME}"

    def get_job_name(base):
        now = datetime.now()
        # Format example: 2023-03-03-10-14-02-324
        now_ms_str = f"{now.microsecond // 1000:03d}"
        date_str = f"{now.strftime('%Y-%m-%d-%H-%M-%S')}-{now_ms_str}"
        job_name = "-".join([base, date_str])
        return job_name

    if args.name is None:
        job_name = get_job_name(base_job_name)
    else:
        job_name = f"{base_job_name}--{args.name}".replace('_', '-')

    output_root = f"{args.s3_remote_sync}/sagemaker/{args.user}/{NAME}/"
    output_s3 = os.path.join(output_root, job_name)
    
    tags = [
        {
            "Key": "tri.project", 
            "Value": "MM:PJ-0077",
        },
        {
            "Key": "tri.owner.email",
            "Value": "kyle.hatch.clb@tri.global",
        },
    ]

    max_run = 25 * 24 * 60 * 60
    max_wait = 25 * 24 * 60 * 60 if args.spot_instance else None
    keep_alive_period_in_seconds = 300 if not args.spot_instance else None  

    entry_point = args.entry_point
    instance_type = "local_gpu" if args.local else INSTANCE_MAPPER[args.instance_type]
    instance_count = args.instance_count
    train_use_spot_instances = args.spot_instance
    
    checkpoint_s3_uri = os.path.join(
        f's3://tri-ml-sandbox-16011-us-west-2-datasets/kylehatch/video_benchmarking_project/motiontrans-pi0_checkpoints/{args.user}/{NAME}', job_name)
    
    checkpoint_s3_uri = None if args.local else checkpoint_s3_uri


    checkpoint_local_path = "/opt/ml/checkpoints"
    checkpoint_local_path = None if args.local else checkpoint_local_path 
    
    hyperparameters = {
        "config": args.config,
        "task_name":args.task_name, 
        "exp-name": args.exp_name,
        "alpha": args.alpha,
        "single_arm": args.single_arm,
        "checkpoint_base_dir": args.checkpoint_base_dir,
        "assets_base_dir": args.assets_base_dir,
        "batch-size": args.batch_size,
        "repo_id": args.repo_id,
        "dataset_path": args.dataset_path,
        "state_down_sample_steps": args.state_down_sample_steps,
        "action_down_sample_steps": args.action_down_sample_steps,
        "proprioception_rep": args.proprioception_rep,
        "action_rep": args.action_rep,
        "proprioception_droprate": args.proprioception_droprate,
        "use_val_dataset": args.use_val_dataset,
        "val_batch_size": args.val_batch_size,
        "num_train_steps": args.num_train_steps,
        "keep_period": args.keep_period,
        "log_interval": args.log_interval,
        "save_interval": args.save_interval,
        "val_interval": args.val_interval,
        "model.max_token_len": args.model_max_token_len,
    }



    print(f"[yellow]hyperparameters:", hyperparameters)
    
    distribution={
        "torch_distributed": {
            "enabled": True,
        }
    }
    environment = {
        "XLA_PYTHON_CLIENT_MEM_FRACTION":os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"],#.get('XLA_PYTHON_CLIENT_MEM_FRACTION', ''),
        # "HF_HOME":os.environ.get('HF_HOME', ''),
        "OPENPI_DATA_HOME":os.environ["OPENPI_DATA_HOME"],#.get('OPENPI_DATA_HOME', ''),
        "HF_LEROBOT_HOME":os.environ["HF_LEROBOT_HOME"],#.get('HF_LEROBOT_HOME', ''),
        # "PKG_CONFIG_PATH":"$CONDA_PREFIX/lib/pkgconfig",
        # "LD_LIBRARY_PATH":"$CONDA_PREFIX/lib:$LD_LIBRARY_PATH",
        # "WANDB_DIR":os.environ.get('WANDB_DIR', ''),

        "WANDB_MODE": os.environ.get('WANDB_MODE', ''),
        "WANDB_API_KEY": os.environ["WANDB_API_KEY"],#.get('WANDB_API_KEY', ''),
        "WANDB_ENTITY": os.environ["WANDB_ENTITY"],#.get('WANDB_ENTITY', ''),
        "WANDB_PROJECT": os.environ["WANDB_PROJECT"],#.get('WANDB_PROJECT', ''),
        "WANDB__SERVICE_WAIT": "300",
        "HF_TOKEN": os.environ["HF_TOKEN"],#.get("HF_TOKEN", ""),
        "INSTANCE_COUNT": str(args.instance_count),
        "SM_USE_RESERVED_CAPACITY": "1",
        "FI_EFA_FORK_SAFE": "1",
        "NVTE_FUSED_ATTN": "0",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "AWS_S3_USE_CRT": "1",
        # "NCCL_DEBUG": "INFO",
        "TOKENIZERS_PARALLELISM": "false",
        "SAGEMAKER_PROGRAM": entry_point,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
        "CUDA_LAUNCH_BLOCKING": "1",
        "TORCH_USE_CUDA_DSA": "1",
        "SM_JOB_NAME": job_name,
        "SAGEMAKER": "enabled",
        "QUEUE": str(args.queue),
        "INSTANCE_TYPE": str(args.instance_type),
        "INSTANCE_COUNT": str(args.instance_count),
        "NCCL_DEBUG": "TRACE",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "NCCL_DEBUG_SUBSYS": "ALL",
        "NCCL_DEBUG_FILE": "/opt/ml/output/nccl_%h_%p.log",      

        # "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID", ""),
        # "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        # "AWS_SESSION_TOKEN": os.environ.get("AWS_SESSION_TOKEN", ""),
    }

    if args.local:
        # environment["WANDB_MODE"] = "offline"
        environment["AWS_ACCESS_KEY_ID"] = os.environ["AWS_ACCESS_KEY_ID"]
        environment["AWS_SECRET_ACCESS_KEY"] = os.environ["AWS_SECRET_ACCESS_KEY"]
        environment["AWS_SESSION_TOKEN"] = os.environ["AWS_SESSION_TOKEN"]

    if args.name is not None:
        environment['JOB_NAME'] = args.name

    print(f"[yellow]environment: {environment}")

    security_group_ids = {
        'us-east-1': [
            'sg-0afb9fb0e79a54061'
        ],
        'us-west-2': [
            'sg-029d1d476bc087e31',
        ],
    }
    subnets = {
        'us-east-1': [
            'subnet-07bf42d7c9cb929e4',
            'subnet-0e260ba29726b9fbb',
        ],
        'us-west-2': [
            'subnet-0610f766a4cd5cdae', 
            'subnet-029adfb9e225d68f8',
            'subnet-01cc1bfeaf20155b5',
        ]
    }

    print()
    print()
    print('#############################################################')
    print(f'SageMaker Execution Role:       {role}')
    print(f'The name of the Execution role: {role_name[-1]}')
    print(f'SM Queue:                       {is_sm_queue}-{args.priority}-{args.fss_identifier}')
    print(f'AWS region:                     {region}')
    print(f'AWS profile:                    {args.profile}')
    print(f'AWS account:                    {account}')
    print(f'Entry point:                    {entry_point}')
    print(f'Image uri:                      {image_uri}')
    print(f'Job name:                       {job_name}')
    print(f'Configuration file:             {hyperparameters}')
    print(f'Instance count:                 {instance_count}')
    print(f'Instance type:                  {instance_type}')
    print(f'Queue:                          {args.queue}')
    print('#############################################################')
    print()
    print()
    
    estimator = Estimator(
        sagemaker_session=sagemaker_session,
        base_job_name=base_job_name,
        hyperparameters=hyperparameters,
        role=role,
        image_uri=image_uri,
        instance_count=instance_count,
        instance_type=instance_type,
        use_spot_instances=train_use_spot_instances,
        output_path=output_s3,
        job_name=job_name,
        checkpoint_s3_uri=checkpoint_s3_uri,
        checkpoint_local_path=checkpoint_local_path,
        max_run=max_run,
        max_wait=max_wait,
        # debugger_hook_config=True,
        environment=environment,
        keep_alive_period_in_seconds=keep_alive_period_in_seconds,
        tags=tags,
        # subnets=subnets[region],
        # security_group_ids=security_group_ids[region],
        volume_size=3000,
        # profiler_config=profiler_config,
        # rules=profiler_rules,
        enable_sagemaker_metrics=True,
        enable_remote_debug=True,
    )

    inputs = None
    if args.local:
        # inputs = {"training": f"file:///home/ubuntu/Desktop/video_benchmarking_project/data"}
        # inputs = {"zarr_data_robot": f"file:///home/ubuntu/Desktop/video_benchmarking_project/data/zarr_data/zarr_data_robot"}
        inputs = {"zarr_data_robot_no_corrupted_episodes_no_idle": f"file:///home/ubuntu/Desktop/video_benchmarking_project/data/zarr_data/zarr_data_robot_no_corrupted_episodes_no_idle"}
        # from sagemaker.inputs import TrainingInput
        # inputs = {f"zarr_data_robot": TrainingInput(s3_data=f"s3://tri-ml-sandbox-16011-us-west-2-datasets/kylehatch/video_benchmarking_project/motiontrans-pi0_data/zarr_data/zarr_data_robot", input_mode="File")}
        
    # elif args.dataset_path and args.dataset_path.startswith("s3://"):
    else:
        from sagemaker.inputs import TrainingInput
        # inputs = {f"robot_mix+{args.task_name}+.zarr": TrainingInput(s3_data=f"s3://tri-ml-sandbox-16011-us-west-2-datasets/kylehatch/video_benchmarking_project/motiontrans-pi0_data/zarr_data/zarr_data_robot/robot_mix+{args.task_name}+.zarr", input_mode="File")}
        # inputs = {f"zarr_data_robot": TrainingInput(s3_data=f"s3://tri-ml-sandbox-16011-us-west-2-datasets/kylehatch/video_benchmarking_project/motiontrans-pi0_data/zarr_data/zarr_data_robot", input_mode="File")}
        inputs = {f"zarr_data_robot_no_corrupted_episodes_no_idle": TrainingInput(s3_data=f"s3://tri-ml-sandbox-16011-us-west-2-datasets/kylehatch/video_benchmarking_project/motiontrans-pi0_data/zarr_data/zarr_data_robot_no_corrupted_episodes_no_idle", input_mode="File")}


    if is_sm_queue:
        # queue_name = "fss-ml-p5-48xlarge-us-west-2"
        queue = Queue(args.queue)
        print(f"Starting training job on queue: {queue.queue_name}")

        queued_jobs = queue.map(
            estimator,
            inputs=[inputs],
            job_names=[job_name],
            priority=args.priority,
            share_identifier=args.fss_identifier,
            timeout={"attemptDurationSeconds": max_run},
        )
        print(f"Queued jobs: {queued_jobs}")
    else:
        # estimator.fit()
        estimator.fit(inputs=inputs)

        


"/data/zarr_data/zarr_data_robot/robot_mix+$task_name+.zarr"

if __name__ == "__main__":
    main()
