#!/usr/bin/env bash
set -euo pipefail

export WANDB_API_KEY=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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/train_config.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="29524"

cd "${SCRIPT_DIR}"

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  train_stage2_bert_memmap.py \
  --config "${CONFIG_PATH}"
