#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE3_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SCRIPTS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM}"
source "${PROJECT_ROOT}/src/poredlm/training/set_env.sh"

CONFIG_PATH="${1:-${SCRIPT_DIR}/porelm_flow_matching.yaml}"
NUM_PROCESSES="4"

export PYTHONPATH="${STAGE3_ROOT}/src:${PYTHONPATH:-}"
export WANDB_API_KEY=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

cd "${SCRIPTS_ROOT}"

nohup torchrun --standalone --nproc-per-node="${NUM_PROCESSES}" \
  "${STAGE3_ROOT}/scripts/train_porelm.py" "${CONFIG_PATH}" "${@:2}" \
  > "${SCRIPT_DIR}/nohup.out" 2>&1 &