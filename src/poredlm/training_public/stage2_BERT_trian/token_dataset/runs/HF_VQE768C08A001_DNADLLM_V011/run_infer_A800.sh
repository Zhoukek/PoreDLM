#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(cd "${RUN_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${TRAIN_DIR}/../../../../.." && pwd)"

# Keep the inference launcher's two-GPU defaults, while allowing overrides.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29527}"

ENV_SCRIPT="${ENV_SCRIPT:-${PROJECT_ROOT}/src/poredlm/training/set_env.sh}"
if [[ -f "${ENV_SCRIPT}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_SCRIPT}"
fi

# Use the fixed PoreDLM environment, including when launched through nohup.
PYTHON_BIN="${PYTHON_BIN:-/mnt/zzbnew/rnamodel/zhoukexuan/miniconda3/envs/PoreDLM/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: PoreDLM Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -c 'import torch' >/dev/null 2>&1; then
  echo "ERROR: PyTorch cannot be imported with ${PYTHON_BIN}" >&2
  exit 1
fi

# Stable NCCL settings for multi-GPU inference on the A800 host.
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

export NPROC_PER_NODE
export PYTHONPATH="${PROJECT_ROOT}/src:${TRAIN_DIR}:${RUN_DIR}:${PROJECT_ROOT}/src/poredlm:${PYTHONPATH:-}"

CONFIG_PATH="${CONFIG_PATH:-${RUN_DIR}/infer_config.yaml}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/infer.log}"
USE_NOHUP="${USE_NOHUP:-1}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: Inference config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

echo "RUN_DIR=${RUN_DIR}"
echo "TRAIN_DIR=${TRAIN_DIR}"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "GPU_COUNT=${GPU_COUNT}"
echo "MASTER_PORT=${MASTER_PORT}"

cd "${RUN_DIR}"
LAUNCH_CMD=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --nnodes=1
  --nproc_per_node="${NPROC_PER_NODE}"
  --rdzv_backend=c10d
  --rdzv_endpoint="127.0.0.1:${MASTER_PORT}"
  "${RUN_DIR}/infer.py"
  --config "${CONFIG_PATH}"
)

if [[ "${USE_NOHUP}" == "1" ]]; then
  nohup "${LAUNCH_CMD[@]}" > "${LOG_FILE}" 2>&1 &
  echo "Started background tokenization. PID=$! LOG_FILE=${LOG_FILE}"
else
  "${LAUNCH_CMD[@]}" 2>&1 | tee "${LOG_FILE}"
fi
