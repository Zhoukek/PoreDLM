#!/bin/bash
set -euo pipefail

# 先加载MACA环境
source /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/set_env.sh

export PYTHONPATH=/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src:/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm:/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/data/stage2_BERT_Encoder:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0,1,2,3

project_root="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM"

python_script="${project_root}/data/DNA_modifiction/LB07_AND_LB06/preprocess_full_signal_tokenize_1600.py"

split_dir="${project_root}/data/DNA_modifiction/LB07_AND_LB06/LB07/split"
output_dir="${project_root}/data/DNA_modifiction/LB07_AND_LB06/LB07/stage2_fullapple_token1600"
splits="train,validation,test"

model_ckpt="${project_root}/src/poredlm/training/stage1_tokenizer/runs/03_S0_HG002_UNMOD_35g_model_type_1_cnn_type_0_distill_0.1_8k_vq_apple/models/porepgt_vqe_tokenizer.final.pth"

device="cuda:0"
target_token_length=1600
pad_token_id=1
pad_token_text="<|pad|>"
codebook_vocab_offset=5
strategy="apple"
spike_window_size=6000

mkdir -p "${output_dir}"

python "${python_script}" \
  --split-dir "${split_dir}" \
  --output-dir "${output_dir}" \
  --splits "${splits}" \
  --model-ckpt "${model_ckpt}" \
  --device "${device}" \
  --target-token-length "${target_token_length}" \
  --pad-token-id "${pad_token_id}" \
  --pad-token-text "${pad_token_text}" \
  --codebook-vocab-offset "${codebook_vocab_offset}" \
  --strategy "${strategy}" \
  --spike-window-size "${spike_window_size}"
