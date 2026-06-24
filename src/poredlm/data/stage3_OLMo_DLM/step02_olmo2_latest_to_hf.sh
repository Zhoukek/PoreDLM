#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM
OLMO_ROOT=${PROJECT_ROOT}/src/poredlm/training/stage3_OLMo_DLM/OLMo

source ${PROJECT_ROOT}/src/poredlm/training/set_env.sh

export PYTHONPATH=${OLMO_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_API_KEY=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


python3 /mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage3_OLMo_DLM/OLMo/scripts/convert_olmo2_to_hf_dlm.py \
   --input_dir "/mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage3_OLMo_DLM/runs/02_150m_no_cond_8k_vq_context_1200/model/02_150m_no_cond_8k_vq_context_1200/step22000-unsharded"  \
   --output_dir "/mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage3_OLMo_DLM/runs/02_150m_no_cond_8k_vq_context_1200/hf_bert"  \
   --tokenizer_json_path "/mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/S0_HG002_UNMOD-35g/stage2_BERT/03_S0_HG002_UNMOD_35g_model_type_1_cnn_type_0_distill_0.1_8k_vq_apple_split_600/tokenizer-8k.json"