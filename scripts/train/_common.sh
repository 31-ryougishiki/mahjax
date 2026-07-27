#!/usr/bin/env bash
# ============================================================================
# _common.sh — 从 config.json 读取配置的共享函数
# ============================================================================
# 用法：
#   source "${SCRIPT_DIR}/../_common.sh"
#   load_config                          # 从 config.json 加载所有参数
#   print_config                         # 打印当前配置摘要
# ============================================================================

# ── 加载 config.json，所有变量以 CONFIG_ 前缀导出 ──
load_config() {
    local config_file="${SCRIPT_DIR}/config.json"

    if [ ! -f "${config_file}" ]; then
        echo "[ERROR] config.json not found at ${config_file}" >&2
        exit 1
    fi

    # 用 Python 解析并生成 shell eval 语句
    eval "$(python3 - "$config_file" <<'PYEOF'
import json, sys

with open(sys.argv[1], 'r') as f:
    cfg = json.load(f)

def flatten(obj, prefix=''):
    items = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            items.extend(flatten(v, prefix + k + '_'))
    else:
        key = prefix.rstrip('_')
        if isinstance(obj, bool):
            val = 'true' if obj else 'false'
        elif isinstance(obj, str):
            val = obj
        else:
            val = str(obj)
        items.append((key, val))
    return items

for key, val in flatten(cfg):
    # 转义单引号，用单引号包裹值
    safe_val = val.replace("'", "'\\''")
    print(f"CONFIG_{key}='{safe_val}'")
PYEOF
)"

    # ── 派生路径 ──
    CONFIG_device_full="${CONFIG_device_type}:${CONFIG_device_id}"

    # BC 模型格式：NPU 用 .pt（PyTorch），GPU 用 .pkl（JAX flax）
    local ext="pkl"
    [ "${CONFIG_device_type}" = "npu" ] && ext="pt"
    CONFIG_bc_model="${SCRIPT_DIR}/${CONFIG_paths_params}/${CONFIG_env_name}_bc_params.${ext}"

    CONFIG_dataset="${SCRIPT_DIR}/${CONFIG_paths_offline_data}/${CONFIG_env_name}_offline_data.pkl"
    CONFIG_ckpt_dir="${SCRIPT_DIR}/${CONFIG_paths_checkpoints}"
    CONFIG_log_dir="${SCRIPT_DIR}/${CONFIG_paths_logs}"
    CONFIG_fig_dir="${SCRIPT_DIR}/${CONFIG_paths_fig}"

    # ── 环境变量覆盖（向后兼容） ──
    CONFIG_ppo_seed="${SEED:-${CONFIG_ppo_seed}}"
    CONFIG_device_full="${DEVICE:-${CONFIG_device_full}}"
    CONFIG_pipeline_skip_data="${SKIP_DATA:-${CONFIG_pipeline_skip_data}}"
    CONFIG_pipeline_skip_bc="${SKIP_BC:-${CONFIG_pipeline_skip_bc}}"
    CONFIG_pipeline_skip_ppo="${SKIP_PPO:-${CONFIG_pipeline_skip_ppo}}"
    CONFIG_logging_use_wandb="${USE_WANDB:-${CONFIG_logging_use_wandb}}"
}

# ── 打印当前配置 ──
print_config() {
    echo ""
    echo "════════════════════════════════════════════"
    echo "  Configuration (config.json)"
    echo "════════════════════════════════════════════"
    echo "  env:           ${CONFIG_env_name}"
    echo "  round_mode:    ${CONFIG_env_round_mode}"
    echo "  device:        ${CONFIG_device_full}"
    echo "  num_envs:      ${CONFIG_ppo_num_envs}"
    echo "  num_steps:     ${CONFIG_ppo_num_steps}"
    echo "  total_steps:   ${CONFIG_ppo_total_timesteps}"
    echo "  lr:            ${CONFIG_ppo_lr}"
    echo "  seed:          ${CONFIG_ppo_seed}"
    echo "  gae_lambda:    ${CONFIG_ppo_gae_lambda}"
    echo "  use_wandb:     ${CONFIG_logging_use_wandb}"
    echo "════════════════════════════════════════════"
    echo ""
}
