#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(cd "${RUN_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${TRAIN_DIR}/../../../.." && pwd)"

# Set launcher defaults before sourcing the shared environment script.  The
# shared script has its own cluster default for NPROC_PER_NODE.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

ENV_SCRIPT="${ENV_SCRIPT:-${PROJECT_ROOT}/src/poredlm/training/set_env.sh}"
if [[ -f "${ENV_SCRIPT}" ]]; then source "${ENV_SCRIPT}"; fi

export PYTHONPATH="${PROJECT_ROOT}/src:${TRAIN_DIR}:${PROJECT_ROOT}/src/poredlm:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_API_KEY=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


CONFIG_PATH="${CONFIG_PATH:-${RUN_DIR}/config.yaml}"
MASTER_PORT="${MASTER_PORT:-29530}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/run.log}"
USE_NOHUP="${USE_NOHUP:-1}"
export NPROC_PER_NODE

echo "RUN_DIR=${RUN_DIR}"
echo "TRAIN_DIR=${TRAIN_DIR}"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "CUDA device count=$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")"
echo "MASTER_PORT=${MASTER_PORT}"
echo "WANDB_MODE=${WANDB_MODE}"

cmd=(
  torchrun
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_port="${MASTER_PORT}"
  "${TRAIN_DIR}/train.py"
  --config "${CONFIG_PATH}"
)

cd "${RUN_DIR}"
if [[ "${USE_NOHUP}" == "1" ]]; then
  nohup "${cmd[@]}" > "${LOG_FILE}" 2>&1 &
  echo "Started background training. PID=$! LOG_FILE=${LOG_FILE}"
else
  "${cmd[@]}" 2>&1 | tee "${LOG_FILE}"
fi
