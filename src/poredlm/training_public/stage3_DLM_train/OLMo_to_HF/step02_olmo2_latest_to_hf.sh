#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../../../../.." && pwd)
OLMO_ROOT=${PROJECT_ROOT}/src/poredlm/training_public/stage3_DLM_train/OLMo
RUN_ROOT=${RUN_ROOT:-${PROJECT_ROOT}/src/poredlm/training_public/stage3_DLM_train/runs/HF_RSQ741C12V523_DNABERT_V543}
INPUT_DIR=${INPUT_DIR:-${RUN_ROOT}/model_elf_mixed/step65000-unsharded}
OUTPUT_DIR=${OUTPUT_DIR:-${RUN_ROOT}/hf_dlm_elf_mixed_with_cls_token_step65000-unsharded_new}
TOKENIZER_JSON_PATH=${TOKENIZER_JSON_PATH:-/mnt/zzbnew/poregpt/models/HF_RSQ741C12V523_DNABERT_V543/base/tokenizer.json}
PYTHON_BIN=${PYTHON_BIN:-/mnt/zzbnew/rnamodel/zhoukexuan/miniconda3/envs/PoreDLM/bin/python}

source ${PROJECT_ROOT}/src/poredlm/training/set_env.sh

export PYTHONPATH=${OLMO_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0


"${PYTHON_BIN}" ${OLMO_ROOT}/scripts/convert_olmo2_to_hf_dlm.py \
   --input_dir "${INPUT_DIR}" \
   --output_dir "${OUTPUT_DIR}" \
   --tokenizer_json_path "${TOKENIZER_JSON_PATH}" \
   --overwrite
