#!/bin/bash
set -euo pipefail

# 先加载MACA环境
source /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/set_env.sh

export PYTHONPATH=/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src:/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm:/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0,1

project_root="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM"

python_script="${project_root}/src/poredlm/training/stage4_finetune/DNA_modification/probe_c_modification_embedding_classifier.py"

model_name_or_path="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage3_OLMo_DLM/runs/LB07_AND_LB06_MIX/hf_dlm"
jsonl="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/LB07_AND_LB06/stage4_modification/validation_seq1_to_seq17_ref_target_cropped_token_c_modlabel.jsonl.gz"
lb07_jsonl="${project_root}/data/DNA_modifiction/LB07_AND_LB06/all_data/split_LB07_only/validation_seq1_to_seq17_ref_target_cropped_token_c_modlabel.jsonl.gz"
lb06_jsonl="${project_root}/data/DNA_modifiction/LB07_AND_LB06/stage4_modification/validation_seq1_to_seq17_ref_target_cropped_token_c_modlabel.jsonl.gz"
output_dir="${project_root}/src/poredlm/training/stage4_finetune/DNA_modification/outputs/c_modification_probe"

# embedding_source:
#   bert = context_hidden，BERT/context encoder 输出
#   dlm  = ode_hidden，DLM/ELF ODE refinement 后输出
embedding_source="dlm"

# probe_mode:
#   same-site-c        LB06修饰C vs LB07同位点C，和compare_mode=c-mod-sites一致
#   original-c-labels  旧逻辑：单个jsonl里label=2 vs label=1
probe_mode="same-site-c"

# same-site-c时生效：
#   separate  LB06和LB07分别forward后取embedding
#   mixed     LB06和LB07交错放进同一批batch forward后取embedding
c_mod_site_batch_mode="separate"
sequence_key="label"

# train_scope:
#   all-sequences  17个seq合在一起训练一个classifier
#   per-sequence   每个seq单独训练一个classifier
train_scope="per-sequence"

limit_reads=1000
limit_lb07_reads=0
limit_lb06_reads=0
train_frac=0.70
val_frac=0.15
seed=42

device="cuda:1"
classifier_device="cuda:1"
dtype="auto"
batch_size=4
classifier_batch_size=32
max_length=2000
pad_token_id=1
backbone_chunk_size=2000
elf_ode_steps=8
elf_ode_start_t=0.5
elf_self_cond_cfg_scale=1.0

epochs=100
lr=0.001
weight_decay=0.001

# 每个 split 最多保留多少未修饰/修饰 C token；0 表示不截断
max_negative_tokens=200000
max_positive_tokens=0

run_output_dir="${output_dir}/${probe_mode}_${c_mod_site_batch_mode}_${embedding_source}"
if [[ "${train_scope}" == "per-sequence" ]]; then
  run_output_dir="${run_output_dir}_per_sequence"
fi
mkdir -p "${run_output_dir}"

extra_args=()
if [[ "${limit_reads}" -gt 0 ]]; then
  extra_args+=(--limit-reads "${limit_reads}")
fi
if [[ "${limit_lb07_reads}" -gt 0 ]]; then
  extra_args+=(--limit-lb07-reads "${limit_lb07_reads}")
fi
if [[ "${limit_lb06_reads}" -gt 0 ]]; then
  extra_args+=(--limit-lb06-reads "${limit_lb06_reads}")
fi

python "${python_script}" \
  --model-name-or-path "${model_name_or_path}" \
  --probe-mode "${probe_mode}" \
  --jsonl "${jsonl}" \
  --lb07-jsonl "${lb07_jsonl}" \
  --lb06-jsonl "${lb06_jsonl}" \
  --output-dir "${run_output_dir}" \
  --embedding-source "${embedding_source}" \
  --sequence-key "${sequence_key}" \
  --c-mod-site-batch-mode "${c_mod_site_batch_mode}" \
  --train-scope "${train_scope}" \
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
  --seed "${seed}" \
  "${extra_args[@]}"
