# Dockerfile for serving a PI policy.
# Based on UV's instructions: https://docs.astral.sh/uv/guides/integration/docker/#developing-in-a-container

# Build the container:
# docker build . -t openpi_server -f scripts/docker/serve_policy.Dockerfile

# Run the container:
# docker run --rm -it --network=host -v .:/app --gpus=all openpi_server /bin/bash

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

# WORKDIR /app

# Needed because LeRobot uses git-lfs.
# linux-libc-dev, build-essential, clang are needed to build evdev from source.
RUN apt-get update && apt-get install -y git git-lfs linux-libc-dev build-essential clang

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Write the virtual environment outside of the project directory so it doesn't
# leak out of the container when we mount the application code.
ENV UV_PROJECT_ENVIRONMENT=/.venv

# Skip downloading LFS files when cloning git dependencies (e.g. lerobot)
ENV GIT_LFS_SKIP_SMUDGE=1

# Install the project's dependencies using the lockfile and settings
RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT
ENV PATH="/.venv/bin:$PATH"
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/pyproject.toml,target=packages/openpi-client/pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/src,target=packages/openpi-client/src \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project --no-dev


RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev


RUN apt-get update && apt-get install -y curl unzip && \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf awscliv2.zip aws

# RUN uv sync 

# CMD /bin/bash -c "uv run scripts/serve_policy.py $SERVER_ARGS"

RUN mkdir -p /workspace

# RUN uv pip install pip
# RUN uv add ipdb

COPY . /workspace/motiontrans-pi0-zeqing
WORKDIR /workspace/motiontrans-pi0-zeqing
RUN uv sync --frozen --no-dev

# CMD cd /workspace/motiontrans-pi0-zeqing && uv sync --frozen --no-dev && /bin/bash

WORKDIR /workspace

# 

# CMD ["/bin/bash"]

# ===========



COPY .bash_aliases .bash_aliases
COPY .vimrc .vimrc

RUN cat .bash_aliases >> ~/.bash_aliases
RUN cp .vimrc ~/.vimrc

# SageMaker training entrypoint: reads hyperparameters from JSON config
# and passes them as CLI args to the script specified by SAGEMAKER_PROGRAM.
RUN printf '#!/bin/bash\nset -e\nPROGRAM="${SAGEMAKER_PROGRAM:-scripts/train.py}"\nHYPERPARAMS_FILE="/opt/ml/input/config/hyperparameters.json"\nARGS=""\nif [ -f "$HYPERPARAMS_FILE" ]; then\n    ARGS=$(python3 -c "import json; hp=json.load(open(\"$HYPERPARAMS_FILE\")); print(\" \".join([\"--\"+k+\" \"+str(v).replace(chr(34),\"\") for k,v in hp.items() if v is not None and not k.startswith(\"sagemaker\")]))")\nfi\nif [ -d "/opt/ml/code" ]; then\n    cd /opt/ml/code\nelse\n    cd /workspace/motiontrans-pi0-zeqing\nfi\nexec python $PROGRAM $ARGS\n' > /usr/local/bin/train && chmod +x /usr/local/bin/train