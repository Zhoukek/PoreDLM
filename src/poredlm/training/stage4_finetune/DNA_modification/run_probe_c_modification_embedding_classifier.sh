#!/bin/bash
set -euo pipefail

# 先加载MACA环境
source /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/set_env.sh

export PYTHONPATH=/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src:/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm:/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0

project_root="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM"

python_script="${project_root}/src/poredlm/training/stage4_finetune/DNA_modification/probe_c_modification_embedding_classifier.py"

model_name_or_path="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage3_OLMo_DLM/runs/LB07_AND_LB06_MIX/hf_dlm"
jsonl="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/LB07_AND_LB06/stage4_modification/validation_seq1_to_seq17_ref_target_cropped_token_c_modlabel.jsonl.gz"
output_dir="${project_root}/src/poredlm/training/stage4_finetune/DNA_modification/outputs/c_modification_probe"

# embedding_source:
#   bert = context_hidden，BERT/context encoder 输出
#   dlm  = ode_hidden，DLM/ELF ODE refinement 后输出
embedding_source="bert"

limit_reads=1000
train_frac=0.70
val_frac=0.15
seed=42

device="cuda:0"
classifier_device="cuda:0"
dtype="auto"
batch_size=4
classifier_batch_size=8192
max_length=2000
pad_token_id=1
backbone_chunk_size=2000
elf_ode_steps=4
elf_ode_start_t=0.85
elf_self_cond_cfg_scale=1.0

epochs=10000
lr=0.001
weight_decay=0.001

# 每个 split 最多保留多少未修饰/修饰 C token；0 表示不截断
max_negative_tokens=200000
max_positive_tokens=0

mkdir -p "${output_dir}/${embedding_source}"

python "${python_script}" \
  --model-name-or-path "${model_name_or_path}" \
  --jsonl "${jsonl}" \
  --output-dir "${output_dir}/${embedding_source}" \
  --embedding-source "${embedding_source}" \
  --limit-reads "${limit_reads}" \
  --train-frac "${train_frac}" \
  --val-frac "${val_frac}" \
  --device "${device}" \
  --classifier-device "${classifier_device}" \
  --dtype "${dtype}" \
  --batch-size "${batch_size}" \
  --classifier-batch-size "${classifier_batch_size}" \
  --max-length "${max_length}" \
  --pad-token-id "${pad_token_id}" \
  --backbone-chunk-size "${backbone_chunk_size}" \
  --elf-ode-steps "${elf_ode_steps}" \
  --elf-ode-start-t "${elf_ode_start_t}" \
  --elf-self-cond-cfg-scale "${elf_self_cond_cfg_scale}" \
  --epochs "${epochs}" \
  --lr "${lr}" \
  --weight-decay "${weight_decay}" \
  --max-negative-tokens "${max_negative_tokens}" \
  --max-positive-tokens "${max_positive_tokens}" \
  --seed "${seed}"
