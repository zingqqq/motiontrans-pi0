SHELL := /bin/bash
# .RECIPEPREFIX += . # use if you want spaces instead of tabs for Makefiles

# Type "make help" for usage instructions

TEAM ?= TRI-ML
DATA_ROOT ?= /data
TEST_PATH ?= tests/
# See Dockerfile for explanation of this variable.
SHELL_SETUP_FILE ?= /usr/local/bin/efm_env_setup.sh
# This flag is used to determine whether to run the docker commands interactively.
# This is used to allow for running in settings where we can't run interactively
# (mainly when running github actions workflows on ec2).
INTERACTIVE := yes
INITSUBMODULES := yes

reponame := openpi_server

docker_image_name := $(reponame)
WANDB_DOCKER = $(docker_image_name)

# data_host_dir := $(HOME)/Desktop/video_benchmarking_project/workspace/data
# data_local_dir := /data

# hf_cache_host_dir := $(HOME)/Desktop/video_benchmarking_project/workspace/cache/huggingface
# hf_cache_local_dir := /workspace/cache/huggingface # /opt/ml/input/cache/huggingface

DOCKER_OPTS := --rm
DOCKER_OPTS += -e XAUTHORITY -e DISPLAY=$(DISPLAY) -v /tmp/.X11-unix:/tmp/.X11-unix
DOCKER_OPTS += --shm-size 64G
DOCKER_OPTS += --ipc=host --network=host --pid=host --privileged
DOCKER_OPTS += -e AWS_DEFAULT_REGION -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e S3_BUCKET_NAME
DOCKER_OPTS += -e WANDB_API_KEY -e WANDB_DOCKER
DOCKER_OPTS += -e OPENAI_API_KEY
DOCKER_OPTS += -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility # Needed for compiling with CUDA
# DOCKER_OPTS += -v $(data_host_dir):$(data_local_dir)
# DOCKER_OPTS += -v $(hf_cache_host_dir):$(hf_cache_local_dir)


DOCKER_OPTS += -v ~/.ssh:/root/.ssh:ro
DOCKER_OPTS += -v $(HOME)/Desktop/video_benchmarking_project/workspace:/workspace
DOCKER_OPTS += -v $(HOME)/Desktop/video_benchmarking_project/data:/data
DOCKER_OPTS += -v $(HOME)/Desktop/video_benchmarking_project/cache:/cache
DOCKER_OPTS += -v $(HOME)/Desktop/video_benchmarking_project/checkpoints_pi0:/checkpoints_pi0
DOCKER_OPTS += -v $(HOME)/Desktop/video_benchmarking_project/wandb_pi0:/wandb_pi0

ifeq ($(INTERACTIVE),yes)
  DOCKER_OPTS += -it
endif


.PHONY: docker_build
docker_build:
	docker build -f scripts/docker/serve_policy.Dockerfile -t openpi_server:latest .
# 	docker compose -f scripts/docker/compose.yml up --build



.PHONY: docker_interactive
docker_interactive:
	docker run $(DOCKER_OPTS) --gpus all --name $(reponame) $(docker_image_name):latest bash
