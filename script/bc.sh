#!/bin/bash
# Run BC training. Execute from repo root.
# Usage: bash script/bc.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7

python "${REPO_ROOT}/mahjax_pt/examples/bc.py" \
    --num_epochs 5 \
    --batch_size 1024 \
    --device npu:0 \
    --dataset_path "${REPO_ROOT}/mahjax_pt/examples/offline_data/red_mahjong_offline_data.pkl"
