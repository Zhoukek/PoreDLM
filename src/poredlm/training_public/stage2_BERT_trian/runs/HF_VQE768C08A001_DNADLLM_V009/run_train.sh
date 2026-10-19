#!/usr/bin/env bash
set -euo pipefail

export WANDB_API_KEY=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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../../../.." && pwd)"
ENV_SCRIPT="${PROJECT_ROOT}/src/poredlm/training/set_env.sh"
if [[ -f "${ENV_SCRIPT}" ]]; then
  source "${ENV_SCRIPT}"
fi
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/train_config.yaml}"
NPROC_PER_NODE="8"
MASTER_PORT="29501"

cd "${SCRIPT_DIR}"

nohup torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  "${PROJECT_ROOT}/src/poredlm/training_public/stage2_BERT_trian/train_stage2_bert_memmap.py" \
  --config "${CONFIG_PATH}"
