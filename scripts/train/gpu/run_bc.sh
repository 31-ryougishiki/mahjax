#!/usr/bin/env bash
# ============================================================================
# BC (Behavior Cloning) 预训练 — GPU 版
# ============================================================================
# 产出：
#   offline_data/red_mahjong_offline_data.pkl  — 离线数据集
#   params/red_mahjong_bc_params.pt            — BC 预训练模型
#   logs/bc_train.log                          — 训练日志
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="${SCRIPT_DIR}/offline_data"
PARAMS_DIR="${SCRIPT_DIR}/params"
LOG_DIR="${SCRIPT_DIR}/logs"

mkdir -p "${DATA_DIR}" "${PARAMS_DIR}" "${LOG_DIR}"

DATASET_PATH="${DATA_DIR}/red_mahjong_offline_data.pkl"
MODEL_PATH="${PARAMS_DIR}/red_mahjong_bc_params.pt"
LOG_FILE="${LOG_DIR}/bc_train.log"

# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Collect offline data
# ═══════════════════════════════════════════════════════════════════════════
echo "════════════════════════════════════════════" | tee -a "${LOG_FILE}"
echo "[BC] Step 1/2: Collecting offline data ..."     | tee -a "${LOG_FILE}"
echo "  Dataset: ${DATASET_PATH}"                     | tee -a "${LOG_FILE}"
echo "════════════════════════════════════════════" | tee -a "${LOG_FILE}"

cd "${PROJECT_ROOT}"

python mahjax_pt/examples/collect_offline_data.py \
    --env_name red_mahjong \
    --dataset_path "${DATASET_PATH}" \
    2>&1 | tee -a "${LOG_FILE}"

echo "[BC] Offline data collected." | tee -a "${LOG_FILE}"

# ═══════════════════════════════════════════════════════════════════════════
# Step 2: BC training
# ═══════════════════════════════════════════════════════════════════════════
echo "════════════════════════════════════════════" | tee -a "${LOG_FILE}"
echo "[BC] Step 2/2: Training BC model ..."         | tee -a "${LOG_FILE}"
echo "  Model:  ${MODEL_PATH}"                     | tee -a "${LOG_FILE}"
echo "  Device: cuda:0"                            | tee -a "${LOG_FILE}"
echo "════════════════════════════════════════════" | tee -a "${LOG_FILE}"

python mahjax_pt/examples/bc.py \
    --env_name red_mahjong \
    --dataset_path "${DATASET_PATH}" \
    --save_model_path "${MODEL_PATH}" \
    --device cuda:0 \
    2>&1 | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "════════════════════════════════════════════" | tee -a "${LOG_FILE}"
echo "[BC] All done!"                                | tee -a "${LOG_FILE}"
echo "  Model:  ${MODEL_PATH}"                      | tee -a "${LOG_FILE}"
echo "  Log:    ${LOG_FILE}"                        | tee -a "${LOG_FILE}"
echo "════════════════════════════════════════════" | tee -a "${LOG_FILE}"
