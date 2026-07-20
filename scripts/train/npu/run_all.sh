#!/usr/bin/env bash
# ============================================================================
# 一键全流程 — NPU 版
# ============================================================================
# 所有参数从 config.json 读取。
# 通过 config.json 中 pipeline.skip_* 控制跳过哪些阶段。
# 也可用环境变量 SKIP_DATA=1 / SKIP_BC=1 / SKIP_PPO=1 临时覆盖。
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
load_config

print_config

# ═══════════════════════════════════════════════════════════════════════════
# Step 1: BC（含离线数据收集）
# ═══════════════════════════════════════════════════════════════════════════
if [ "${CONFIG_pipeline.skip_bc}" = "true" ]; then
    echo "[Pipeline] Skipping BC (skip_bc=true)"
else
    bash "${SCRIPT_DIR}/run_bc.sh"
fi

# ═══════════════════════════════════════════════════════════════════════════
# Step 2: PPO
# ═══════════════════════════════════════════════════════════════════════════
if [ "${CONFIG_pipeline.skip_ppo}" = "true" ]; then
    echo "[Pipeline] Skipping PPO (skip_ppo=true)"
else
    bash "${SCRIPT_DIR}/run_ppo.sh"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Pipeline Complete!                                     ║"
echo "║  Outputs: ${SCRIPT_DIR}"
echo "║    params/        — model weights                       ║"
echo "║    checkpoints/   — PPO snapshots                       ║"
echo "║    logs/          — training logs                       ║"
echo "╚══════════════════════════════════════════════════════════╝"
