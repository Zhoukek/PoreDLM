#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE3_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SCRIPTS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM}"
ENV_SCRIPT="${ENV_SCRIPT:-${PROJECT_ROOT}/src/poredlm/training/set_env.sh}"
if [[ -f "${ENV_SCRIPT}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_SCRIPT}"
fi

CONFIG_PATH="${1:-${SCRIPT_DIR}/porelm_flow_matching.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29516}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/zzbnew/rnamodel/zhoukexuan/miniconda3/envs/PoreDLM/bin/python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: PoreDLM Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -c 'import torch' >/dev/null 2>&1; then
  echo "ERROR: PyTorch cannot be imported with ${PYTHON_BIN}" >&2
  exit 1
fi

# Stable NCCL settings for the current A800 host. The default P2P/CUMEM path
# has been observed to hang during multi-GPU collectives on this machine.
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

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: Config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

export NPROC_PER_NODE
export PYTHONPATH="${STAGE3_ROOT}/src:${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export WANDB_API_KEY=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

cd "${SCRIPTS_ROOT}"

echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "STAGE3_ROOT=${STAGE3_ROOT}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "GPU_COUNT=${GPU_COUNT}"
echo "MASTER_PORT=${MASTER_PORT}"

LAUNCH_CMD=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --nnodes=1
  --nproc_per_node="${NPROC_PER_NODE}"
  --rdzv_backend=c10d
  --rdzv_endpoint="127.0.0.1:${MASTER_PORT}"
  "${STAGE3_ROOT}/scripts/train_porelm.py"
  "${CONFIG_PATH}"
  "${@:2}"
)

nohup "${LAUNCH_CMD[@]}" \
  > "${SCRIPT_DIR}/nohup.out" 2>&1 &

echo "Training started. PID=$! Log: ${SCRIPT_DIR}/nohup.out"
