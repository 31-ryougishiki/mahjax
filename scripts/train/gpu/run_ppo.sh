#!/usr/bin/env bash
# ============================================================================
# PPO 强化学习训练 — GPU 版（JAX 路径，原生 GPU 加速）
# ============================================================================
# 与 NPU 版不同：GPU 上 JAX 通过 jax.jit + jax.vmap 原生编译 CUDA kernel，
# 性能优于 PyTorch eager 模式，是 GPU 训练的首选路径。
#
# 产出：
#   checkpoints/ppo_ckpt_*.pkl            — 周期 checkpoint（每 eval_interval 步）
#   params/{env_name}-seed={N}.ckpt       — 最终模型
#   logs/ppo_train.log                    — 训练日志
#   fig/ppo_with_reg_agent_game.svg       — 可视化
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PARAMS_DIR="${SCRIPT_DIR}/params"
CKPT_DIR="${SCRIPT_DIR}/checkpoints"
LOG_DIR="${SCRIPT_DIR}/logs"
FIG_DIR="${SCRIPT_DIR}/fig"

mkdir -p "${PARAMS_DIR}" "${CKPT_DIR}" "${LOG_DIR}" "${FIG_DIR}"

ENV_NAME="${ENV_NAME:-no_red_mahjong}"
PRETRAINED="${PARAMS_DIR}/${ENV_NAME}_bc_params.pkl"
LOG_FILE="${LOG_DIR}/ppo_train.log"

# ═══════════════════════════════════════════════════════════════════════════
# 参数（通过环境变量覆盖）
# ═══════════════════════════════════════════════════════════════════════════
SEED="${SEED:-0}"
ROUND_MODE="${ROUND_MODE:-single}"
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

# ═══════════════════════════════════════════════════════════════════════════
# 检查 BC 模型
# ═══════════════════════════════════════════════════════════════════════════
if [ ! -f "${PRETRAINED}" ]; then
    echo "[PPO] ERROR: BC pretrained model not found at ${PRETRAINED}"
    echo "[PPO] Please run run_bc.sh first."
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════
# PPO 训练 (JAX 路径)
# ═══════════════════════════════════════════════════════════════════════════
echo "════════════════════════════════════════════" | tee -a "${LOG_FILE}"
echo "[PPO] Starting PPO training (JAX, GPU native)" | tee -a "${LOG_FILE}"
echo "  BC model:    ${PRETRAINED}"                 | tee -a "${LOG_FILE}"
echo "  Log:         ${LOG_FILE}"                   | tee -a "${LOG_FILE}"
echo "  Config: seed=${SEED} num_envs=${NUM_ENVS} num_steps=${NUM_STEPS}" | tee -a "${LOG_FILE}"
echo "════════════════════════════════════════════" | tee -a "${LOG_FILE}"

cd "${PROJECT_ROOT}"

# JAX PPO 使用 OmegaConf CLI 格式（key=value）
python examples/ppo_with_reg.py \
    "env_name=${ENV_NAME}" \
    "round_mode=${ROUND_MODE}" \
    "seed=${SEED}" \
    "num_envs=${NUM_ENVS}" \
    "num_steps=${NUM_STEPS}" \
    "total_timesteps=${TOTAL_TIMESTEPS}" \
    "lr=${LR}" \
    "ent_coef=${ENT_COEF}" \
    "clip_eps=${CLIP_EPS}" \
    "vf_coef=${VF_COEF}" \
    "update_epochs=${UPDATE_EPOCHS}" \
    "minibatch_size=${MINIBATCH_SIZE}" \
    "mag_coef=${MAG_COEF}" \
    "pretrained_model_path=${PRETRAINED}" \
    "viz_out_dir=${FIG_DIR}" \
    "viz_filename=ppo_with_reg_agent_game.svg" \
    2>&1 | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "════════════════════════════════════════════" | tee -a "${LOG_FILE}"
echo "[PPO] Training finished!"                     | tee -a "${LOG_FILE}"
echo "  Log:  ${LOG_FILE}"                          | tee -a "${LOG_FILE}"
echo "  Fig:  ${FIG_DIR}"                           | tee -a "${LOG_FILE}"
echo "════════════════════════════════════════════" | tee -a "${LOG_FILE}"
