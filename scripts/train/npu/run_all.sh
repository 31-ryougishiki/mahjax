#!/usr/bin/env bash
# ============================================================================
# 一键全流程训练 — NPU 版
# ============================================================================
# 依次执行：收集离线数据 → BC 预训练 → PPO 训练
# 通过环境变量切换行为：
#   SKIP_DATA=1     跳过离线数据收集
#   SKIP_BC=1       跳过 BC 预训练
#   SKIP_PPO=1      跳过 PPO 训练
#   USE_WANDB=1     启用 wandb 日志
#
# 产出目录（均在 scripts/train/npu/ 下）：
#   offline_data/   离线数据集
#   params/         模型参数（BC + PPO 最终模型）
#   checkpoints/    PPO 周期 checkpoint
#   logs/           训练日志
#   fig/            可视化
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║     MahJax NPU Training Pipeline                        ║"
echo "║     Scripts: ${SCRIPT_DIR}"
echo "║     Device:  npu:0                                      ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Step 1: BC 预训练（含离线数据收集）
# ═══════════════════════════════════════════════════════════════════════════
if [ "${SKIP_BC:-0}" != "1" ]; then
    if [ "${SKIP_DATA:-0}" = "1" ]; then
        echo "[Pipeline] Skipping data collection (SKIP_DATA=1)"
    fi
    bash "${SCRIPT_DIR}/run_bc.sh"
else
    echo "[Pipeline] Skipping BC (SKIP_BC=1)"
fi

# ═══════════════════════════════════════════════════════════════════════════
# Step 2: PPO 训练
# ═══════════════════════════════════════════════════════════════════════════
if [ "${SKIP_PPO:-0}" != "1" ]; then
    bash "${SCRIPT_DIR}/run_ppo.sh"
else
    echo "[Pipeline] Skipping PPO (SKIP_PPO=1)"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║     Pipeline Complete!                                  ║"
echo "║     Outputs: ${SCRIPT_DIR}                              ║"
echo "║       params/        — model weights                    ║"
echo "║       checkpoints/   — PPO snapshots                    ║"
echo "║       logs/          — training logs                    ║"
echo "╚══════════════════════════════════════════════════════════╝"
