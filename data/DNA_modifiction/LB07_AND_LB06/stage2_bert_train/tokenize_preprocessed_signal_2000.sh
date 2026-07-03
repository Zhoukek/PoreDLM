#!/bin/bash
set -euo pipefail

# 先加载MACA环境
source /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/set_env.sh

export PYTHONPATH=/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src:/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm:/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/data/stage2_BERT_Encoder:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0,1

project_root="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM"
python_script="${project_root}/data/DNA_modifiction/LB07_AND_LB06/stage2_bert_train/tokenize_preprocessed_signal_2000.py"

split_dir="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/LB07_AND_LB06/stage2_bert_train/validation"
output_dir="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/LB07_AND_LB06/stage2_bert_train/validation"
splits="validation_signal_cropped"

model_ckpt="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage1_tokenizer/runs/LB07_AND_LB06_model_type_1_cnn_type_0_distill_0.1_8k_vq_apple_mix/models/porepgt_vqe_tokenizer.final.pth"

device="cuda:0"
target_token_length=2000
min_signal_len=1000
max_signal_len=10000
pad_token_id=1
pad_token_text="<|pad|>"
codebook_vocab_offset=5

mkdir -p "${output_dir}"

python "${python_script}" \
  --split-dir "${split_dir}" \
  --output-dir "${output_dir}" \
  --splits "${splits}" \
  --model-ckpt "${model_ckpt}" \
  --device "${device}" \
  --target-token-length "${target_token_length}" \
  --min-signal-len "${min_signal_len}" \
  --max-signal-len "${max_signal_len}" \
  --pad-token-id "${pad_token_id}" \
  --pad-token-text "${pad_token_text}" \
  --codebook-vocab-offset "${codebook_vocab_offset}"
