#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../../../../.." && pwd)

source "${PROJECT_ROOT}/src/poredlm/training/set_env.sh"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0

cd "${SCRIPT_DIR}"
python3 "${SCRIPT_DIR}/generate_and_plot.py" \
  --config "${CONFIG_PATH:-${SCRIPT_DIR}/config.yaml}" \
  "$@"
