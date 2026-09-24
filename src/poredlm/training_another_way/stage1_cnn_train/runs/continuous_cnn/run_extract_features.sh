#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(cd "${RUN_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${TRAIN_DIR}/../../../.." && pwd)"

ENV_SCRIPT="${ENV_SCRIPT:-${PROJECT_ROOT}/src/poredlm/training/set_env.sh}"
if [[ -f "${ENV_SCRIPT}" ]]; then source "${ENV_SCRIPT}"; fi

export PYTHONPATH="${PROJECT_ROOT}/src:${TRAIN_DIR}:${PROJECT_ROOT}/src/poredlm:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WANDB_MODE="${WANDB_MODE:-offline}"

CONFIG_PATH="${CONFIG_PATH:-${RUN_DIR}/config.yaml}"
CHECKPOINT="${1:?checkpoint required}"
SPLIT="${2:?train or valid required}"
OUTPUT_DIR="${3:?output directory required}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29531}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/extract_${SPLIT}.log}"
USE_NOHUP="${USE_NOHUP:-0}"

echo "RUN_DIR=${RUN_DIR}"
echo "TRAIN_DIR=${TRAIN_DIR}"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "SPLIT=${SPLIT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MASTER_PORT=${MASTER_PORT}"

cmd=(
  torchrun
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_port="${MASTER_PORT}"
  "${TRAIN_DIR}/extract_features.py"
  --config "${CONFIG_PATH}"
  --checkpoint "${CHECKPOINT}"
  --split "${SPLIT}"
  --output-dir "${OUTPUT_DIR}"
)

cd "${RUN_DIR}"
if [[ "${USE_NOHUP}" == "1" ]]; then
  nohup "${cmd[@]}" > "${LOG_FILE}" 2>&1 &
  echo "Started background extraction. PID=$! LOG_FILE=${LOG_FILE}"
else
  "${cmd[@]}" 2>&1 | tee "${LOG_FILE}"
fi
