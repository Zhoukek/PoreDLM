#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(cd "${RUN_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${TRAIN_DIR}/../../../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29532}"
ENV_SCRIPT="${ENV_SCRIPT:-${PROJECT_ROOT}/src/poredlm/training/set_env.sh}"
if [[ -f "${ENV_SCRIPT}" ]]; then source "${ENV_SCRIPT}"; fi

# Use the same fixed training environment as the working launchers. This
# avoids relying on PATH when the script is started through nohup.
PYTHON_BIN="${PYTHON_BIN:-/mnt/zzbnew/rnamodel/zhoukexuan/miniconda3/envs/PoreDLM/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: PoreDLM Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -c 'import torch' >/dev/null 2>&1; then
  echo "ERROR: PyTorch cannot be imported with ${PYTHON_BIN}" >&2
  exit 1
fi

# ESC8000's P2P/CUMEM path can hang during DDP parameter broadcast. Keep the
# same stable NCCL settings as the known-working launcher.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eno1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

GPU_COUNT="$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')"
if [[ "${GPU_COUNT}" -lt "${NPROC_PER_NODE}" ]]; then
  echo "ERROR: PoreDLM environment sees ${GPU_COUNT} CUDA devices; ${NPROC_PER_NODE} are required." >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}/src:${TRAIN_DIR}:${PROJECT_ROOT}/src/poredlm:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_API_KEY=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

CONFIG_PATH="${CONFIG_PATH:-${RUN_DIR}/config.yaml}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/run.log}"
USE_NOHUP="${USE_NOHUP:-1}"
export NPROC_PER_NODE

echo "RUN_DIR=${RUN_DIR}"
echo "TRAIN_DIR=${TRAIN_DIR}"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "WANDB_MODE=${WANDB_MODE}"

cd "${RUN_DIR}"
LAUNCH_CMD=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --nnodes=1
  --nproc_per_node="${NPROC_PER_NODE}"
  --rdzv_backend=c10d
  --rdzv_endpoint="127.0.0.1:${MASTER_PORT}"
  "${TRAIN_DIR}/train.py"
  --config "${CONFIG_PATH}"
)

if [[ "${USE_NOHUP}" == "1" ]]; then
  nohup "${LAUNCH_CMD[@]}" > "${LOG_FILE}" 2>&1 &
  echo "Started background training. PID=$! LOG_FILE=${LOG_FILE}"
else
  "${LAUNCH_CMD[@]}" 2>&1 | tee "${LOG_FILE}"
fi
