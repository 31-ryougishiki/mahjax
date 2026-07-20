#!/usr/bin/env bash
# ============================================================================
# PPO 强化学习训练 — GPU 版
# ============================================================================
# 依赖：需要先跑 run_bc.sh 生成 BC 预训练模型
#
# 产出：
#   checkpoints/ppo_ckpt_*.pt            — 周期 checkpoint（每 eval_interval 步）
#   params/red_mahjong-seed=0.pt         — 最终模型
#   logs/ppo_train.log                   — 训练日志
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PARAMS_DIR="${SCRIPT_DIR}/params"
CKPT_DIR="${SCRIPT_DIR}/checkpoints"
LOG_DIR="${SCRIPT_DIR}/logs"

mkdir -p "${PARAMS_DIR}" "${CKPT_DIR}" "${LOG_DIR}"

PRETRAINED="${PARAMS_DIR}/red_mahjong_bc_params.pt"
LOG_FILE="${LOG_DIR}/ppo_train.log"

# ═══════════════════════════════════════════════════════════════════════════
# 参数（可按需修改）
# ═══════════════════════════════════════════════════════════════════════════
ENV_NAME="red_mahjong"
ROUND_MODE="single"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda:0}"

NUM_ENVS="${NUM_ENVS:-1024}"
NUM_STEPS="${NUM_STEPS:-256}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-100000000}"

LR="${LR:-3e-4}"
ENT_COEF="${ENT_COEF:-0.01}"
CLIP_EPS="${CLIP_EPS:-0.2}"
VF_COEF="${VF_COEF:-0.5}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-4096}"
MAG_COEF="${MAG_COEF:-0.2}"

EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-1000}"

USE_WANDB="${USE_WANDB:-}"
WANDB_PROJECT="${WANDB_PROJECT:-mahjax-ppo}"

# ═══════════════════════════════════════════════════════════════════════════
# 检查 BC 模型
# ═══════════════════════════════════════════════════════════════════════════
if [ ! -f "${PRETRAINED}" ]; then
    echo "[PPO] ERROR: BC pretrained model not found at ${PRETRAINED}"
    echo "[PPO] Please run run_bc.sh first."
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════
# wandb
# ═══════════════════════════════════════════════════════════════════════════
WANDB_ARGS=()
if [ -n "${USE_WANDB}" ]; then
    WANDB_ARGS=(--use_wandb --wandb_project "${WANDB_PROJECT}")
fi

# ═══════════════════════════════════════════════════════════════════════════
# PPO 训练
# ═══════════════════════════════════════════════════════════════════════════
echo "════════════════════════════════════════════" | tee -a "${LOG_FILE}"
echo "[PPO] Starting PPO training on ${DEVICE}"     | tee -a "${LOG_FILE}"
echo "  BC model:    ${PRETRAINED}"                 | tee -a "${LOG_FILE}"
echo "  Checkpoints: ${CKPT_DIR}"                   | tee -a "${LOG_FILE}"
echo "  Log:         ${LOG_FILE}"                   | tee -a "${LOG_FILE}"
echo "  Config: seed=${SEED} num_envs=${NUM_ENVS} num_steps=${NUM_STEPS}" | tee -a "${LOG_FILE}"
echo "════════════════════════════════════════════" | tee -a "${LOG_FILE}"

cd "${PROJECT_ROOT}"

python mahjax_pt/examples/ppo_with_reg.py \
    --env_name "${ENV_NAME}" \
    --round_mode "${ROUND_MODE}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --num_envs "${NUM_ENVS}" \
    --num_steps "${NUM_STEPS}" \
    --total_timesteps "${TOTAL_TIMESTEPS}" \
    --lr "${LR}" \
    --ent_coef "${ENT_COEF}" \
    --clip_eps "${CLIP_EPS}" \
    --vf_coef "${VF_COEF}" \
    --update_epochs "${UPDATE_EPOCHS}" \
    --minibatch_size "${MINIBATCH_SIZE}" \
    --mag_coef "${MAG_COEF}" \
    --pretrained_model_path "${PRETRAINED}" \
    --checkpoint_dir "${CKPT_DIR}" \
    --eval_interval "${EVAL_INTERVAL}" \
    --eval_num_envs "${EVAL_NUM_ENVS}" \
    "${WANDB_ARGS[@]}" \
    2>&1 | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "════════════════════════════════════════════" | tee -a "${LOG_FILE}"
echo "[PPO] Training finished!"                     | tee -a "${LOG_FILE}"
echo "  Checkpoints: ${CKPT_DIR}"                  | tee -a "${LOG_FILE}"
echo "  Log:         ${LOG_FILE}"                  | tee -a "${LOG_FILE}"
echo "════════════════════════════════════════════" | tee -a "${LOG_FILE}"
