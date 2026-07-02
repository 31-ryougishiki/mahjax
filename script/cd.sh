#!/bin/bash
# Collect offline data. Execute from repo root.
# Usage: bash script/cd.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

python "${REPO_ROOT}/mahjax_pt/examples/collect_offline_data.py" \
    --num_samples 2000 \
    --num_envs 4
