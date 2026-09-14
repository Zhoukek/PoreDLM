#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM
source ${PROJECT_ROOT}/src/poredlm/training/set_env.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-${ROOT_DIR}/configs/porelm_flow_matching.yaml}"
NUM_PROCESSES="2"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export WANDB_API_KEY=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


nohup torchrun --standalone --nproc-per-node="${NUM_PROCESSES}" \
  "${ROOT_DIR}/scripts/train_porelm.py" "${CONFIG_PATH}" "${@:2}"