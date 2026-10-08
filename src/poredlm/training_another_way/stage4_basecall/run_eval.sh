#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE4_ROOT="$(cd "${RUN_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${STAGE4_ROOT}/../../../.." && pwd)"

ENV_SCRIPT="${ENV_SCRIPT:-${PROJECT_ROOT}/src/poredlm/training/set_env.sh}"
if [[ -f "${ENV_SCRIPT}" ]]; then source "${ENV_SCRIPT}"; fi

export PYTHONPATH="${PROJECT_ROOT}/src:${STAGE4_ROOT}:${PROJECT_ROOT}/src/poredlm:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

CONFIG_PATH="${CONFIG_PATH:-${RUN_DIR}/config.yaml}"
CKPT="${CKPT:-${RUN_DIR}/save/continuous_basecall/ckpt_best.pt}"
INPUT_PATH="${INPUT_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/eval_out}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"

if [[ -z "${INPUT_PATH}" ]]; then
  echo "Usage: INPUT_PATH=/path/to/*_chunks.npy bash $0"
  exit 1
fi
if [[ ! -e "${INPUT_PATH}" ]]; then
  echo "ERROR: input path not found: ${INPUT_PATH}"
  exit 1
fi
if [[ ! -e "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
cd "${RUN_DIR}"
python "${STAGE4_ROOT}/eval.py" \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CKPT}" \
  --input "${INPUT_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  2>&1 | tee "${OUTPUT_DIR}/eval.log"

echo "Evaluation done. Outputs are in ${OUTPUT_DIR}"
