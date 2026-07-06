#!/bin/bash
set -euo pipefail

# 先加载MACA环境
source /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/set_env.sh

export PYTHONPATH=/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src:/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm:/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0

project_root="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM"

python_script="${project_root}/src/poredlm/training/stage4_finetune/DNA_modification/plot_lb06_lb07_same_site_c_embedding_distribution.py"

model_name_or_path="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage3_OLMo_DLM/runs/LB07_AND_LB06_MIX/hf_dlm"

lb07_jsonl="${project_root}/data/DNA_modifiction/LB07_AND_LB06/all_data/split_LB07_only/validation_seq1_to_seq17_ref_target_cropped_token_c_modlabel.jsonl.gz"
lb06_jsonl="${project_root}/data/DNA_modifiction/LB07_AND_LB06/stage4_modification/validation_seq1_to_seq17_ref_target_cropped_token_c_modlabel.jsonl.gz"
output_dir="${project_root}/src/poredlm/training/stage4_finetune/DNA_modification/outputs/LB06_vs_LB07_same_site_C"

# embedding_source:
#   bert = context_hidden，BERT/context encoder 输出
#   dlm  = ode_hidden，DLM/ELF ODE refinement 后输出
embedding_source="dlm"

# sequence_key:
#   label 默认按 seq_1 ... seq_17 分组
#   ref   按 ref 字符串分组
#   seq   按 seq 字符串分组
sequence_key="label"

# plot_mode:
#   all          所有 seq 聚合成一张图
#   per-sequence 每个 seq 单独一张图
#   both         两种都画
plot_mode="per-sequence"

limit_lb07_reads=0
limit_lb06_reads=0

device="cuda:0"
dtype="auto"
batch_size=4
max_length=2000
pad_token_id=1
backbone_chunk_size=2000
elf_ode_steps=32
elf_ode_start_t=0.5
elf_self_cond_cfg_scale=1.0

# 点太多时可以抽样；0 表示不抽样
max_lb07_points=100000
max_lb06_points=0
seed=42

mkdir -p "${output_dir}/${embedding_source}"

extra_args=()
if [[ "${limit_lb07_reads}" -gt 0 ]]; then
  extra_args+=(--limit-lb07-reads "${limit_lb07_reads}")
fi
if [[ "${limit_lb06_reads}" -gt 0 ]]; then
  extra_args+=(--limit-lb06-reads "${limit_lb06_reads}")
fi

python "${python_script}" \
  --model-name-or-path "${model_name_or_path}" \
  --lb07-jsonl "${lb07_jsonl}" \
  --lb06-jsonl "${lb06_jsonl}" \
  --output-dir "${output_dir}/${embedding_source}_step_${elf_ode_steps}_t_${elf_ode_start_t}" \
  --embedding-source "${embedding_source}" \
  --sequence-key "${sequence_key}" \
  --plot-mode "${plot_mode}" \
  --device "${device}" \
  --dtype "${dtype}" \
  --batch-size "${batch_size}" \
  --max-length "${max_length}" \
  --pad-token-id "${pad_token_id}" \
  --backbone-chunk-size "${backbone_chunk_size}" \
  --elf-ode-steps "${elf_ode_steps}" \
  --elf-ode-start-t "${elf_ode_start_t}" \
  --elf-self-cond-cfg-scale "${elf_self_cond_cfg_scale}" \
  --max-lb07-points "${max_lb07_points}" \
  --max-lb06-points "${max_lb06_points}" \
  --seed "${seed}" \
  "${extra_args[@]}"
