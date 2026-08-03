#!/usr/bin/env bash
set -euo pipefail

source /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/set_env.sh

export CUDA_VISIBLE_DEVICES=0,1
export WANDB_API_KEY=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


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/config.yaml}"
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NPROC_PER_NODE=2
MASTER_PORT=29054
TRAIN_SCRIPT="${TRAIN_SCRIPT:-/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/satge_EX_Waveform_Reconstruction/training_waveform_decoder/train.py}"

cd "${SCRIPT_DIR}"

nohup torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  "${TRAIN_SCRIPT}" \
  --config "${CONFIG_PATH}"
