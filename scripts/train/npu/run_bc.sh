#!/usr/bin/env bash
# ============================================================================
# BC (Behavior Cloning) 预训练 — NPU 版
# ============================================================================
# 配置来源：config.json（同目录下）
# 产出：
#   offline_data/{env}_offline_data.pkl
#   params/{env}_bc_params.pt
#   logs/bc_train.log
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
load_config

mkdir -p "${CONFIG_dataset%/*}" "${CONFIG_bc_model%/*}" "${CONFIG_log_dir}"

LOG_FILE="${CONFIG_log_dir}/bc_train.log"

# ═══════════════════════════════════════════════════════════════════════════
# Step 1: 收集离线数据
# ═══════════════════════════════════════════════════════════════════════════
if [ "${CONFIG_pipeline.skip_data}" = "true" ]; then
    echo "[BC] Skipping data collection (skip_data=true in config.json)"
else
    echo "════════════════════════════════════════════" | tee -a "${LOG_FILE}"
    echo "[BC] Step 1/2: Collecting offline data ..."     | tee -a "${LOG_FILE}"
    echo "  Env:     ${CONFIG_env.name}"                 | tee -a "${LOG_FILE}"
    echo "  Dataset: ${CONFIG_dataset}"                  | tee -a "${LOG_FILE}"
    echo "════════════════════════════════════════════" | tee -a "${LOG_FILE}"

    cd "${PROJECT_ROOT}"
    python mahjax_pt/examples/collect_offline_data.py \
        --env_name "${CONFIG_env.name}" \
        --dataset_path "${CONFIG_dataset}" \
        2>&1 | tee -a "${LOG_FILE}"

    echo "[BC] Offline data collected." | tee -a "${LOG_FILE}"
fi

# ═══════════════════════════════════════════════════════════════════════════
# Step 2: BC 训练
# ═══════════════════════════════════════════════════════════════════════════
echo "════════════════════════════════════════════" | tee -a "${LOG_FILE}"
echo "[BC] Step 2/2: Training BC model ..."         | tee -a "${LOG_FILE}"
echo "  Model:  ${CONFIG_bc_model}"                | tee -a "${LOG_FILE}"
echo "  Device: ${CONFIG_device_full}"              | tee -a "${LOG_FILE}"
echo "  Batch:  ${CONFIG_bc.batch_size}"            | tee -a "${LOG_FILE}"
echo "  LR:     ${CONFIG_bc.lr}"                    | tee -a "${LOG_FILE}"
echo "  Epochs: ${CONFIG_bc.num_epochs}"           | tee -a "${LOG_FILE}"
echo "════════════════════════════════════════════" | tee -a "${LOG_FILE}"

cd "${PROJECT_ROOT}"
python mahjax_pt/examples/bc.py \
    --env_name "${CONFIG_env.name}" \
    --dataset_path "${CONFIG_dataset}" \
    --save_model_path "${CONFIG_bc_model}" \
    --device "${CONFIG_device_full}" \
    --batch_size "${CONFIG_bc.batch_size}" \
    --lr "${CONFIG_bc.lr}" \
    --num_epochs "${CONFIG_bc.num_epochs}" \
    2>&1 | tee -a "${LOG_FILE}"

echo ""
echo "════════════════════════════════════════════" | tee -a "${LOG_FILE}"
echo "[BC] Done!"                                   | tee -a "${LOG_FILE}"
echo "  Model: ${CONFIG_bc_model}"                 | tee -a "${LOG_FILE}"
echo "  Log:   ${LOG_FILE}"                         | tee -a "${LOG_FILE}"
echo "════════════════════════════════════════════" | tee -a "${LOG_FILE}"
