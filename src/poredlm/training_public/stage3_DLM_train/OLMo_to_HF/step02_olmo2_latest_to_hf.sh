#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../../../../.." && pwd)
OLMO_ROOT=${PROJECT_ROOT}/src/poredlm/training_public/stage3_DLM_train/OLMo
RUN_ROOT=${PROJECT_ROOT}/src/poredlm/training_public/stage3_DLM_train/runs/HF_VQE768C08A001_DNADLLM_V002

source ${PROJECT_ROOT}/src/poredlm/training/set_env.sh

export PYTHONPATH=${OLMO_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0


python3 ${OLMO_ROOT}/scripts/convert_olmo2_to_hf_dlm.py \
   --input_dir "${RUN_ROOT}/model_mixed/step43000-unsharded"  \
   --output_dir "${RUN_ROOT}/hf_dlm_condition"  \
   --tokenizer_json_path "${PROJECT_ROOT}/src/poredlm/data/stage2_BERT_Encoder/tokenizer-64k.json" \
   --overwrite
