#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM
OLMO_ROOT=${PROJECT_ROOT}/src/poredlm/training/stage3_OLMo_DLM/OLMo
RUN_ROOT=${PROJECT_ROOT}/src/poredlm/training/stage3_OLMo_DLM/runs/00_150m_no_cond_8k_vq

source ${PROJECT_ROOT}/src/poredlm/training/set_env.sh

export PYTHONPATH=${OLMO_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_API_KEY=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

cd ${OLMO_ROOT}

# DELAY=3600

# # 如果传入 delay 参数，则延迟执行
# if [[ -n "${DELAY:-}" ]]; then
#     echo "Waiting ${DELAY} seconds before starting..."
#     sleep ${DELAY}
# fi

nohup torchrun --nproc_per_node=4 --rdzv_endpoint=localhost:29509 \
    scripts/train_DLM.py ${RUN_ROOT}/config_150m.yaml \
    --run_name="poredlm-stage3-olmo-150m-dlm" \
    --wandb.entity="zhoukek-zhejiang-university" \
    --wandb.project="poredlm-stage3" \
    --load_path="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage3_OLMo_DLM/runs/00_150m_no_cond_8k_vq/model/olmo_150m_dlm/step3000-unsharded" \
    --save_folder="${RUN_ROOT}/model/olmo_150m_dlm" \
    > "${outdir}/nohup.out" 2>&1 &
