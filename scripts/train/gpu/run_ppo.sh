#!/usr/bin/env bash
# ============================================================================
# PPO 强化学习训练 — GPU 版（JAX 路径，原生 GPU 加速）
# ============================================================================
# 配置来源：config.json（同目录下）+ 环境变量覆盖
# JAX 通过 jax.jit + jax.vmap 原生编译 CUDA kernel。
# CLI 使用 OmegaConf 格式（key=value）。
#
# 产出：
#   checkpoints/ppo_ckpt_*.pkl
#   params/{env}-seed={N}.ckpt
#   logs/ppo_train.log
#   fig/ppo_with_reg_agent_game.svg
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
load_config

mkdir -p "${CONFIG_ckpt_dir}" "${CONFIG_log_dir}" "${CONFIG_fig_dir}"

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
# 构建 OmegaConf 参数
# ═══════════════════════════════════════════════════════════════════════════
OMEGA_ARGS=(
    "env_name=${CONFIG_env.name}"
    "round_mode=${CONFIG_env.round_mode}"
    "seed=${CONFIG_ppo.seed}"
    "num_envs=${CONFIG_ppo.num_envs}"
    "num_steps=${CONFIG_ppo.num_steps}"
    "total_timesteps=${CONFIG_ppo.total_timesteps}"
    "lr=${CONFIG_ppo.lr}"
    "ent_coef=${CONFIG_ppo.ent_coef}"
    "clip_eps=${CONFIG_ppo.clip_eps}"
    "vf_coef=${CONFIG_ppo.vf_coef}"
    "update_epochs=${CONFIG_ppo.update_epochs}"
    "minibatch_size=${CONFIG_ppo.minibatch_size}"
    "mag_coef=${CONFIG_ppo.mag_coef}"
    "pretrained_model_path=${CONFIG_bc_model}"
    "viz_out_dir=${CONFIG_fig_dir}"
    "viz_filename=ppo_with_reg_agent_game.svg"
)

# wandb（JAX 版默认启用，可通过 config 关闭）
if [ "${CONFIG_logging.use_wandb}" = "true" ]; then
    OMEGA_ARGS+=("wandb_project=${CONFIG_logging.wandb_project}")
fi

# ═══════════════════════════════════════════════════════════════════════════
# PPO 训练
# ═══════════════════════════════════════════════════════════════════════════
print_config | tee -a "${LOG_FILE}"

echo "[PPO] Backend:   JAX (jax.jit, native CUDA)"  | tee -a "${LOG_FILE}"
echo "[PPO] BC model:  ${CONFIG_bc_model}"          | tee -a "${LOG_FILE}"
echo "[PPO] Log:       ${LOG_FILE}"                 | tee -a "${LOG_FILE}"
echo ""                                             | tee -a "${LOG_FILE}"

cd "${PROJECT_ROOT}"

python examples/ppo_with_reg.py \
    "${OMEGA_ARGS[@]}" \
    2>&1 | tee -a "${LOG_FILE}"

echo ""
echo "════════════════════════════════════════════" | tee -a "${LOG_FILE}"
echo "[PPO] Done!"                                   | tee -a "${LOG_FILE}"
echo "  Log: ${LOG_FILE}"                            | tee -a "${LOG_FILE}"
echo "  Fig: ${CONFIG_fig_dir}"                      | tee -a "${LOG_FILE}"
echo "════════════════════════════════════════════" | tee -a "${LOG_FILE}"
