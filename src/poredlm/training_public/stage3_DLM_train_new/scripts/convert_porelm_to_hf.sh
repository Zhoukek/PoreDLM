#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STAGE3_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${STAGE3_ROOT}/../../../../.." && pwd)}"

ENV_SCRIPT="${ENV_SCRIPT:-${PROJECT_ROOT}/src/poredlm/training/set_env.sh}"
if [[ -f "${ENV_SCRIPT}" ]]; then
  # shellcheck source=/dev/null
  source "${ENV_SCRIPT}"
fi

export PYTHONPATH="${STAGE3_ROOT}/src:${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

INPUT_DIR="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training_public/stage3_DLM_train_new/scripts/runs/HF_VQE768C08A001_DNADLLM_V006_self_flow/step96000-unsharded"
OUTPUT_DIR="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training_public/stage3_DLM_train_new/scripts/runs/HF_VQE768C08A001_DNADLLM_V006_self_flow/hf_dlm_step96000"
TOKENIZER_JSON_PATH="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/data/stage2_BERT_Encoder/tokenizer-64k.json"

ARGS=(
  --input_dir "${INPUT_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --tokenizer_json_path "${TOKENIZER_JSON_PATH}"
)

if [[ "${OVERWRITE:-1}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

if [[ "${INCLUDE_TOKENIZER:-1}" == "0" ]]; then
  ARGS+=(--no_tokenizer)
fi

if [[ "${SAFE_SERIALIZATION:-1}" == "0" ]]; then
  ARGS+=(--no_safe_serialization)
fi

python3 "${SCRIPT_DIR}/convert_porelm_to_hf.py" "${ARGS[@]}"
