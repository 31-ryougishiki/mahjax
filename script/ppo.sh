#!/bin/bash
# Run PPO training with BatchState-native parallel environment.
# Execute from repo root.
#
# Usage:
#   bash script/ppo.sh              # GPU (cuda:0), 128 envs (dev)
#   bash script/ppo.sh gpu 1024     # GPU, 1024 envs (production)
#   bash script/ppo.sh npu 1024     # NPU (Ascend), 1024 envs
#
# Key parameters (override via env vars):
#   DEVICE      — torch device (default: cuda:0)
#   NUM_ENVS    — number of parallel environments (default: 128)
#   NUM_STEPS   — rollout steps per update (default: 128)
#   TOTAL_STEPS — total env steps (default: 50000 for quick test, 100000000 for prod)
#   ROUND_MODE  — single / east / half (default: single)
#   CKPT_DIR    — checkpoint directory (default: none)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# ── Device selection ──
PLATFORM="${1:-gpu}"
NUM_ENVS="${2:-128}"

DEVICE="${DEVICE:-cuda:0}"
if [ "$PLATFORM" = "npu" ]; then
    DEVICE="${DEVICE:-npu:0}"
    export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-4,5,6,7}"
fi

# ── Training scale ──
NUM_STEPS="${NUM_STEPS:-128}"
TOTAL_STEPS="${TOTAL_STEPS:-50000}"
ROUND_MODE="${ROUND_MODE:-single}"
LR="${LR:-3e-4}"
MINIBATCH="${MINIBATCH:-4096}"

# ── Optional: checkpointing ──
CKPT_ARGS=""
if [ -n "${CKPT_DIR:-}" ]; then
    CKPT_ARGS="--checkpoint_dir ${CKPT_DIR}"
fi

# ── Optional: wandb ──
WANDB_ARGS=""
if [ "${USE_WANDB:-0}" = "1" ]; then
    WANDB_ARGS="--use_wandb --wandb_project ${WANDB_PROJECT:-mahjax-ppo}"
fi

echo "══════════════════════════════════════════════════════"
echo " PPO Training (BatchState-native)"
echo "   Platform:   ${PLATFORM}  |  Device: ${DEVICE}"
echo "   Num envs:   ${NUM_ENVS}  |  Steps:  ${NUM_STEPS}"
echo "   Total:      ${TOTAL_STEPS}  |  Round:  ${ROUND_MODE}"
echo "   LR:         ${LR}  |  Minibatch: ${MINIBATCH}"
echo "══════════════════════════════════════════════════════"

python "${REPO_ROOT}/mahjax_pt/examples/ppo_with_reg.py" \
    --num_envs "${NUM_ENVS}" \
    --num_steps "${NUM_STEPS}" \
    --total_timesteps "${TOTAL_STEPS}" \
    --round_mode "${ROUND_MODE}" \
    --lr "${LR}" \
    --minibatch_size "${MINIBATCH}" \
    --device "${DEVICE}" \
    ${CKPT_ARGS} \
    ${WANDB_ARGS}
