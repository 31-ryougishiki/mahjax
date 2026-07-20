#!/usr/bin/env bash
# ============================================================================
# PPO 强化学习训练 — NPU 版
# ============================================================================
# 配置来源：config.json（同目录下）+ 环境变量覆盖
# 产出：
#   checkpoints/ppo_ckpt_*.pt
#   params/{env}-seed={N}.pt
#   logs/ppo_train.log
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
load_config

mkdir -p "${CONFIG_ckpt_dir}" "${CONFIG_log_dir}"

LOG_FILE="${CONFIG_log_dir}/ppo_train.log"

# ═══════════════════════════════════════════════════════════════════════════
# 检查 BC 模型
# ═══════════════════════════════════════════════════════════════════════════
if [ ! -f "${CONFIG_bc_model}" ]; then
    echo "[PPO] ERROR: BC model not found at ${CONFIG_bc_model}"
    echo "[PPO] Please run run_bc.sh first."
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════
# wandb 参数
# ═══════════════════════════════════════════════════════════════════════════
WANDB_ARGS=()
if [ "${CONFIG_logging.use_wandb}" = "true" ]; then
    WANDB_ARGS=(--use_wandb --wandb_project "${CONFIG_logging.wandb_project}")
fi

# ═══════════════════════════════════════════════════════════════════════════
# PPO 训练
# ═══════════════════════════════════════════════════════════════════════════
print_config | tee -a "${LOG_FILE}"

echo "[PPO] BC model:    ${CONFIG_bc_model}"      | tee -a "${LOG_FILE}"
echo "[PPO] Checkpoints: ${CONFIG_ckpt_dir}"      | tee -a "${LOG_FILE}"
echo "[PPO] Log:         ${LOG_FILE}"             | tee -a "${LOG_FILE}"
echo ""                                           | tee -a "${LOG_FILE}"

cd "${PROJECT_ROOT}"

python mahjax_pt/examples/ppo_with_reg.py \
    --env_name "${CONFIG_env.name}" \
    --round_mode "${CONFIG_env.round_mode}" \
    --seed "${CONFIG_ppo.seed}" \
    --device "${CONFIG_device_full}" \
    --num_envs "${CONFIG_ppo.num_envs}" \
    --num_steps "${CONFIG_ppo.num_steps}" \
    --total_timesteps "${CONFIG_ppo.total_timesteps}" \
    --lr "${CONFIG_ppo.lr}" \
    --ent_coef "${CONFIG_ppo.ent_coef}" \
    --clip_eps "${CONFIG_ppo.clip_eps}" \
    --vf_coef "${CONFIG_ppo.vf_coef}" \
    --update_epochs "${CONFIG_ppo.update_epochs}" \
    --minibatch_size "${CONFIG_ppo.minibatch_size}" \
    --mag_coef "${CONFIG_ppo.mag_coef}" \
    --pretrained_model_path "${CONFIG_bc_model}" \
    --checkpoint_dir "${CONFIG_ckpt_dir}" \
    --eval_interval "${CONFIG_ppo.eval_interval}" \
    --eval_num_envs "${CONFIG_ppo.eval_num_envs}" \
    "${WANDB_ARGS[@]}" \
    2>&1 | tee -a "${LOG_FILE}"

echo ""
echo "════════════════════════════════════════════" | tee -a "${LOG_FILE}"
echo "[PPO] Done!"                                   | tee -a "${LOG_FILE}"
echo "  Checkpoints: ${CONFIG_ckpt_dir}"            | tee -a "${LOG_FILE}"
echo "  Log:         ${LOG_FILE}"                   | tee -a "${LOG_FILE}"
echo "════════════════════════════════════════════" | tee -a "${LOG_FILE}"
