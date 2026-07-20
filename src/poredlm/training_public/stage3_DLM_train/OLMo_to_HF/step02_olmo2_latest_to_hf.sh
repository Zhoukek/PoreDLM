#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM
OLMO_ROOT=${PROJECT_ROOT}/src/poredlm/training_public/stage3_DLM_train/OLMo
RUN_ROOT=${PROJECT_ROOT}/src/poredlm/training_public/stage3_DLM_train/runs/test

source ${PROJECT_ROOT}/src/poredlm/training/set_env.sh

export PYTHONPATH=${OLMO_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_API_KEY=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


python3 ${OLMO_ROOT}/scripts/convert_olmo2_to_hf_dlm.py \
   --input_dir "${RUN_ROOT}/model/step100000-unsharded"  \
   --output_dir "${RUN_ROOT}/hf_dlm"  \
   --tokenizer_json_path "${PROJECT_ROOT}/src/poredlm/data/stage2_BERT_Encoder/tokenizer-64k.json" \
   --overwrite