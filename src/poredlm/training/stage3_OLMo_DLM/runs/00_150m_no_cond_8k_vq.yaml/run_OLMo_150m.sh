#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM
OLMO_ROOT=${PROJECT_ROOT}/src/poredlm/training/stage3_OLMo_DLM/OLMo
RUN_ROOT=${PROJECT_ROOT}/src/poredlm/training/stage3_OLMo_DLM/runs/test_zhou

source ${PROJECT_ROOT}/src/poredlm/training/set_env.sh

export PYTHONPATH=${OLMO_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_API_KEY=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

cd ${OLMO_ROOT}

torchrun --nproc_per_node=2 --rdzv_endpoint=localhost:29502 \
    scripts/train_DLM.py ${RUN_ROOT}/config_150m_a100.yaml \
    --run_name="poredlm-stage3-olmo-150m-dlm" \
    --wandb.entity="zhoukek-zhejiang-university" \
    --wandb.project="poredlm-stage3" \
    --load_path="" \
    --save_folder="${RUN_ROOT}/model/olmo_150m_dlm" 2>&1 | tee ${RUN_ROOT}/run.log
