#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(cd "${RUN_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${TRAIN_DIR}/../../../.." && pwd)"

ENV_SCRIPT="${ENV_SCRIPT:-${PROJECT_ROOT}/src/poredlm/training/set_env.sh}"
if [[ -f "${ENV_SCRIPT}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_SCRIPT}"
fi

export PYTHONPATH="${PROJECT_ROOT}/src:${TRAIN_DIR}:${PROJECT_ROOT}/src/poredlm:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

CONFIG_PATH="${CONFIG_PATH:-${RUN_DIR}/config.yaml}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
  CONFIG_PATH="${TRAIN_DIR}/config/train_vq_distill_config.yaml"
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29513}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/run.log}"
USE_NOHUP="${USE_NOHUP:-0}"

echo "RUN_DIR=${RUN_DIR}"
echo "TRAIN_DIR=${TRAIN_DIR}"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
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
