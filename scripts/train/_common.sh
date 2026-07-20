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
            items.extend(flatten(v, prefix + k + '.'))
    else:
        key = prefix.rstrip('.')
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
    CONFIG_device_full="${CONFIG_device.type}:${CONFIG_device.id}"

    # BC 模型格式：NPU 用 .pt（PyTorch），GPU 用 .pkl（JAX flax）
    local ext="pkl"
    [ "${CONFIG_device.type}" = "npu" ] && ext="pt"
    CONFIG_bc_model="${SCRIPT_DIR}/${CONFIG_paths.params}/${CONFIG_env.name}_bc_params.${ext}"

    CONFIG_dataset="${SCRIPT_DIR}/${CONFIG_paths.offline_data}/${CONFIG_env.name}_offline_data.pkl"
    CONFIG_ckpt_dir="${SCRIPT_DIR}/${CONFIG_paths.checkpoints}"
    CONFIG_log_dir="${SCRIPT_DIR}/${CONFIG_paths.logs}"
    CONFIG_fig_dir="${SCRIPT_DIR}/${CONFIG_paths.fig}"

    # ── 环境变量覆盖（向后兼容） ──
    CONFIG_ppo.seed="${SEED:-${CONFIG_ppo.seed}}"
    CONFIG_device_full="${DEVICE:-${CONFIG_device_full}}"
    CONFIG_pipeline.skip_data="${SKIP_DATA:-${CONFIG_pipeline.skip_data}}"
    CONFIG_pipeline.skip_bc="${SKIP_BC:-${CONFIG_pipeline.skip_bc}}"
    CONFIG_pipeline.skip_ppo="${SKIP_PPO:-${CONFIG_pipeline.skip_ppo}}"
    CONFIG_logging.use_wandb="${USE_WANDB:-${CONFIG_logging.use_wandb}}"
}

# ── 打印当前配置 ──
print_config() {
    echo ""
    echo "════════════════════════════════════════════"
    echo "  Configuration (config.json)"
    echo "════════════════════════════════════════════"
    echo "  env:           ${CONFIG_env.name}"
    echo "  round_mode:    ${CONFIG_env.round_mode}"
    echo "  device:        ${CONFIG_device_full}"
    echo "  num_envs:      ${CONFIG_ppo.num_envs}"
    echo "  num_steps:     ${CONFIG_ppo.num_steps}"
    echo "  total_steps:   ${CONFIG_ppo.total_timesteps}"
    echo "  lr:            ${CONFIG_ppo.lr}"
    echo "  seed:          ${CONFIG_ppo.seed}"
    echo "  gae_lambda:    ${CONFIG_ppo.gae_lambda}"
    echo "  use_wandb:     ${CONFIG_logging.use_wandb}"
    echo "════════════════════════════════════════════"
    echo ""
}
