#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${RUN_DIR}/../../../.." && pwd)"

ENV_SCRIPT="${ENV_SCRIPT:-${PROJECT_ROOT}/src/poredlm/training/set_env.sh}"
if [[ -f "${ENV_SCRIPT}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_SCRIPT}"
fi

export PYTHONPATH="${PROJECT_ROOT}/src:${RUN_DIR}:${PROJECT_ROOT}/src/poredlm:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

CONFIG_PATH="${CONFIG_PATH:-${RUN_DIR}/infer_config.yaml}"
NPROC_PER_NODE="4"
MASTER_PORT="29527"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/infer.log}"
USE_NOHUP="1"

echo "RUN_DIR=${RUN_DIR}"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MASTER_PORT=${MASTER_PORT}"

cmd=(
  torchrun
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_port="${MASTER_PORT}"
  "${RUN_DIR}/infer.py"
  --config "${CONFIG_PATH}"
)

cd "${RUN_DIR}"
if [[ "${USE_NOHUP}" == "1" ]]; then
  nohup "${cmd[@]}" > "${LOG_FILE}" 2>&1 &
  echo "Started background tokenization. PID=$! LOG_FILE=${LOG_FILE}"
else
  "${cmd[@]}" 2>&1 | tee "${LOG_FILE}"
fi
