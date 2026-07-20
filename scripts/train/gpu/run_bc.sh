#!/usr/bin/env bash
# ============================================================================
# BC (Behavior Cloning) 预训练 — GPU 版（JAX 路径）
# ============================================================================
# 产出：
#   offline_data/no_red_mahjong_offline_data.pkl  — 离线数据集
#   params/no_red_mahjong_bc_params.pkl           — BC 预训练模型
#   logs/bc_train.log                             — 训练日志
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="${SCRIPT_DIR}/offline_data"
PARAMS_DIR="${SCRIPT_DIR}/params"
LOG_DIR="${SCRIPT_DIR}/logs"

mkdir -p "${DATA_DIR}" "${PARAMS_DIR}" "${LOG_DIR}"

ENV_NAME="${ENV_NAME:-no_red_mahjong}"
DATASET_PATH="${DATA_DIR}/${ENV_NAME}_offline_data.pkl"
MODEL_PATH="${PARAMS_DIR}/${ENV_NAME}_bc_params.pkl"
LOG_FILE="${LOG_DIR}/bc_train.log"

# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Collect offline data
# ═══════════════════════════════════════════════════════════════════════════
echo "════════════════════════════════════════════" | tee -a "${LOG_FILE}"
echo "[BC] Step 1/2: Collecting offline data ..."     | tee -a "${LOG_FILE}"
echo "  Env:     ${ENV_NAME}"                         | tee -a "${LOG_FILE}"
echo "  Dataset: ${DATASET_PATH}"                     | tee -a "${LOG_FILE}"
echo "════════════════════════════════════════════" | tee -a "${LOG_FILE}"

cd "${PROJECT_ROOT}"

python examples/collect_offline_data.py \
    "env_name=${ENV_NAME}" \
    "dataset_path=${DATASET_PATH}" \
    2>&1 | tee -a "${LOG_FILE}"

echo "[BC] Offline data collected." | tee -a "${LOG_FILE}"

# ═══════════════════════════════════════════════════════════════════════════
# Step 2: BC training (JAX)
# ═══════════════════════════════════════════════════════════════════════════
echo "════════════════════════════════════════════" | tee -a "${LOG_FILE}"
echo "[BC] Step 2/2: Training BC model (JAX) ..."   | tee -a "${LOG_FILE}"
echo "  Model:  ${MODEL_PATH}"                     | tee -a "${LOG_FILE}"
echo "════════════════════════════════════════════" | tee -a "${LOG_FILE}"

python examples/bc.py \
    "env_name=${ENV_NAME}" \
    "dataset_path=${DATASET_PATH}" \
    "save_model_path=${MODEL_PATH}" \
    2>&1 | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "════════════════════════════════════════════" | tee -a "${LOG_FILE}"
echo "[BC] All done!"                                | tee -a "${LOG_FILE}"
echo "  Model:  ${MODEL_PATH}"                      | tee -a "${LOG_FILE}"
echo "  Log:    ${LOG_FILE}"                        | tee -a "${LOG_FILE}"
echo "════════════════════════════════════════════" | tee -a "${LOG_FILE}"
