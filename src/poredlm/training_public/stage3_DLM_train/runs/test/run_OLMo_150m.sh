#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM
OLMO_ROOT=${PROJECT_ROOT}/src/poredlm/training_public/stage3_DLM_train/OLMo
RUN_ROOT=${PROJECT_ROOT}/src/poredlm/training_public/stage3_DLM_train/runs/test

source ${PROJECT_ROOT}/src/poredlm/training/set_env.sh

export PYTHONPATH=${OLMO_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_API_KEY=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

cd ${OLMO_ROOT}

nohup torchrun --nproc_per_node=4 --rdzv_endpoint=localhost:29511 \
    scripts/train_DLM.py ${RUN_ROOT}/config_150m.yaml \
    --run_name="dlm_ELF_B_64K" \
    --wandb.entity="zhoukek-zhejiang-university" \
    --wandb.project="stage3_dlm_public" \
    --save_folder="${RUN_ROOT}/model" \
    > "${RUN_ROOT}/nohup.out" 2>&1 &
