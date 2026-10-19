#!/usr/bin/env bash
set -euo pipefail

# Precompute frozen DLM context_encoder features for Stage 4 basecalling.
# Run this from the same Python environment used by basecall training.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE4_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${STAGE4_ROOT}/../../../.." && pwd)"

export PYTHONPATH="${STAGE4_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# Override these variables when using another model or dataset, for example:
#   MODEL_PATH=/path/to/model DATA_PATHS=/path/to/jsonl_dir OUTPUT_DIR=/path/to/cache ./run_precompute_context_hidden.sh
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/src/poredlm/training_public/stage3_DLM_train/runs/HF_RSQ741C12V523_DNABERT_V543/hf_dlm_elf_mixed_with_cls_token_step65000-unsharded}"
DATA_FORMAT="${DATA_FORMAT:-jsonl}" # jsonl or npy
DATA_PATHS="${DATA_PATHS:-${SCRIPT_DIR}/RNA_S0_HEK293T_MIX_250NC02106021/basecall_20000_chunks}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/context_hidden_cache/HF_RSQ741C12V523_DNABERT_V543/basecall_20000_chunks}"

TOKENIZER_TYPE="${TOKENIZER_TYPE:-bwav}"
TOKENIZER_TOKEN_OFFSET="${TOKENIZER_TOKEN_OFFSET:-128}"
TOKEN_OFFSET="${TOKEN_OFFSET:-0}"
ADD_CLS_TOKEN="${ADD_CLS_TOKEN:-true}"
CLS_TOKEN_ID="${CLS_TOKEN_ID:-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"
BACKBONE_CHUNK_SIZE="${BACKBONE_CHUNK_SIZE:-1540}"
DEVICE="${DEVICE:-cuda}"
RECURSIVE="${RECURSIVE:-false}"
OVERWRITE="${OVERWRITE:-false}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model directory does not exist: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -e "${DATA_PATHS%%,*}" ]]; then
  echo "Data path does not exist: ${DATA_PATHS%%,*}" >&2
  exit 1
fi
if [[ "${DATA_FORMAT}" != "jsonl" && "${DATA_FORMAT}" != "npy" ]]; then
  echo "DATA_FORMAT must be 'jsonl' or 'npy', got: ${DATA_FORMAT}" >&2
  exit 1
fi

data_args=()
if [[ "${DATA_FORMAT}" == "jsonl" ]]; then
  data_args+=(--jsonl_paths "${DATA_PATHS}")
else
  data_args+=(--npy_paths "${DATA_PATHS}")
fi

special_token_args=()
if [[ "${ADD_CLS_TOKEN}" == "true" ]]; then
  special_token_args+=(--add_cls_token --cls_token_id "${CLS_TOKEN_ID}")
elif [[ "${ADD_CLS_TOKEN}" != "false" ]]; then
  echo "ADD_CLS_TOKEN must be 'true' or 'false', got: ${ADD_CLS_TOKEN}" >&2
  exit 1
fi

optional_args=()
if [[ "${RECURSIVE}" == "true" ]]; then
  optional_args+=(--recursive)
elif [[ "${RECURSIVE}" != "false" ]]; then
  echo "RECURSIVE must be 'true' or 'false', got: ${RECURSIVE}" >&2
  exit 1
fi
if [[ "${OVERWRITE}" == "true" ]]; then
  optional_args+=(--overwrite)
elif [[ "${OVERWRITE}" != "false" ]]; then
  echo "OVERWRITE must be 'true' or 'false', got: ${OVERWRITE}" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_DIR}")"
echo "Model:  ${MODEL_PATH}"
echo "Data:   ${DATA_PATHS} (${DATA_FORMAT})"
echo "Cache:  ${OUTPUT_DIR}"
echo "Tokens: CLS=${ADD_CLS_TOKEN} CLS_ID=${CLS_TOKEN_ID} tokenizer_offset=${TOKENIZER_TOKEN_OFFSET} token_offset=${TOKEN_OFFSET}"

python "${SCRIPT_DIR}/precompute_context_hidden.py" \
  "${data_args[@]}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_name_or_path "${MODEL_PATH}" \
  --tokenizer_type "${TOKENIZER_TYPE}" \
  --tokenizer_token_offset "${TOKENIZER_TOKEN_OFFSET}" \
  --token_offset "${TOKEN_OFFSET}" \
  "${special_token_args[@]}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --backbone_chunk_size "${BACKBONE_CHUNK_SIZE}" \
  --device "${DEVICE}" \
  "${optional_args[@]}"
