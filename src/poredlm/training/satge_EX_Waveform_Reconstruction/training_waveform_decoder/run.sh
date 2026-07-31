#!/usr/bin/env bash
set -euo pipefail

source /mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/set_env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/config.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29502}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-/mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/satge_EX_Waveform_Reconstruction/training_waveform_decoder/train.py}"

cd "${SCRIPT_DIR}"

nohup torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  "${TRAIN_SCRIPT}" \
  --config "${CONFIG_PATH}"
