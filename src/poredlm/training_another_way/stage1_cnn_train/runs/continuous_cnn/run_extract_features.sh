#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(cd "${RUN_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${TRAIN_DIR}/../../../.." && pwd)"

# Set launcher defaults before sourcing the shared environment script.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

ENV_SCRIPT="${ENV_SCRIPT:-${PROJECT_ROOT}/src/poredlm/training/set_env.sh}"
if [[ -f "${ENV_SCRIPT}" ]]; then source "${ENV_SCRIPT}"; fi

export PYTHONPATH="${PROJECT_ROOT}/src:${TRAIN_DIR}:${PROJECT_ROOT}/src/poredlm:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"

CONFIG_PATH="${CONFIG_PATH:-${RUN_DIR}/config.yaml}"
CHECKPOINT="/mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training_another_way/stage1_cnn_train/runs/continuous_cnn/save/continuous_cnn/step_5000"
SPLIT="train"
OUTPUT_DIR="/mnt/si002562jbsc/poregpt/datasets/DNA_S1_HG00200_MIX_250F701901011/trank_stoneq00/features_train"
MASTER_PORT="${MASTER_PORT:-29531}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/extract_${SPLIT}.log}"
USE_NOHUP="${USE_NOHUP:-0}"
export NPROC_PER_NODE

echo "RUN_DIR=${RUN_DIR}"
echo "TRAIN_DIR=${TRAIN_DIR}"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "SPLIT=${SPLIT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "CUDA device count=$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")"
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
