#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(cd "${RUN_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${TRAIN_DIR}/../../../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
ENV_SCRIPT="${ENV_SCRIPT:-${PROJECT_ROOT}/src/poredlm/training/set_env.sh}"
if [[ -f "${ENV_SCRIPT}" ]]; then source "${ENV_SCRIPT}"; fi

export PYTHONPATH="${PROJECT_ROOT}/src:${TRAIN_DIR}:${PROJECT_ROOT}/src/poredlm:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
export NPROC_PER_NODE
CONFIG_PATH="${CONFIG_PATH:-${RUN_DIR}/config.yaml}"
MASTER_PORT="${MASTER_PORT:-29533}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/run.log}"
USE_NOHUP="${USE_NOHUP:-1}"

echo "CONFIG_PATH=${CONFIG_PATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "WANDB_MODE=${WANDB_MODE}"

cmd=(torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" \
  "${TRAIN_DIR}/train.py" --config "${CONFIG_PATH}")
cd "${RUN_DIR}"
if [[ "${USE_NOHUP}" == "1" ]]; then
  nohup "${cmd[@]}" > "${LOG_FILE}" 2>&1 &
  echo "Started background training. PID=$! LOG_FILE=${LOG_FILE}"
else
  "${cmd[@]}" 2>&1 | tee "${LOG_FILE}"
fi
