#!/bin/bash
# Run PPO training. Execute from repo root.
# Usage: bash script/ppo.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7

python "${REPO_ROOT}/mahjax_pt/examples/ppo_with_reg.py" \
    --num_envs 12 \
    --num_steps 128 \
    --total_timesteps 50000 \
    --device npu:0
