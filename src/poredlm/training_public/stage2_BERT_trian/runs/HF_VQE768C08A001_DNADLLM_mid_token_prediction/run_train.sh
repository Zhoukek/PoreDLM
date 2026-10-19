#!/usr/bin/env bash
set -euo pipefail

source /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/set_env.sh


export WANDB_API_KEY=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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/train_config.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29501}"
PYTHON_BIN="/mnt/zzbnew/rnamodel/zhoukexuan/miniconda3/envs/PoreDLM/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: PoreDLM Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ "${NPROC_PER_NODE}" != "2" ]]; then
  echo "ERROR: this run is configured for exactly two GPUs; got NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
# ESC8000 currently advertises an additional USB network interface and its
# NCCL P2P/CUMEM path hangs during DDP parameter broadcast. Use the stable
# Ethernet socket path for this single-node two-GPU job.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eno1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
GPU_COUNT="$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')"
if [[ "${GPU_COUNT}" -lt 2 ]]; then
  echo "ERROR: PoreDLM environment sees ${GPU_COUNT} CUDA devices; two are required." >&2
  exit 1
fi

# Avoid rank-0-only online WandB initialization while bringing up DDP.
# Set POREDLM_DISABLE_WANDB=0 after the two-GPU run is stable if online logging
# is required.
export POREDLM_DISABLE_WANDB="${POREDLM_DISABLE_WANDB:-1}"

cd "${SCRIPT_DIR}"

nohup "${PYTHON_BIN}" -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="127.0.0.1:${MASTER_PORT}" \
  /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training_public/stage2_BERT_trian/train_stage2_bert_memmap.py \
  --config "${CONFIG_PATH}"
