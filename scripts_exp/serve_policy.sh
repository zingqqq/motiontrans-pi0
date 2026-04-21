# bash scripts_exp/serve_policy.sh

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=2,3 

uv run scripts/serve_policy.py \
--env MOTIONTRANS \
--default_prompt "default" \