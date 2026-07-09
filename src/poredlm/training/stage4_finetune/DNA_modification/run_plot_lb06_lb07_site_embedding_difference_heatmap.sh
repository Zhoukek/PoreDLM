#!/bin/bash
set -euo pipefail

# 先加载MACA环境
source /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/set_env.sh

export PYTHONPATH=/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src:/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm:/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0

project_root="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM"

python_script="${project_root}/src/poredlm/training/stage4_finetune/DNA_modification/plot_lb06_lb07_site_embedding_difference_heatmap.py"

model_name_or_path="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage3_OLMo_DLM/runs/LB07_AND_LB06_MIX/hf_dlm"
lb07_jsonl="${project_root}/data/DNA_modifiction/LB07_AND_LB06/all_data/split_LB07_only/validation_seq1_to_seq17_ref_target_cropped_token_c_modlabel.jsonl.gz"
lb06_jsonl="${project_root}/data/DNA_modifiction/LB07_AND_LB06/stage4_modification/validation_seq1_to_seq17_ref_target_cropped_token_c_modlabel.jsonl.gz"
output_dir="${project_root}/src/poredlm/training/stage4_finetune/DNA_modification/outputs/LB06_vs_LB07_site_embedding_heatmap"

# embedding_source:
#   bert = context_hidden，BERT/context encoder 输出
#   dlm  = ode_hidden，DLM/ELF ODE refinement 后输出
embedding_source="dlm"

# 每个seq每个数据集抽多少条read。可以只写一个，比如 "20"
read_depths="5,10,20,50"

# 每个base site可能覆盖多个token：
#   mean   覆盖token的embedding均值
#   center 只取覆盖区间中心token
#   max    覆盖token逐维max
token_pool="mean"
samples_per_token=5

# site差异指标：
#   l2            LB06均值embedding与LB07均值embedding的L2距离
#   cosine        1 - cosine_similarity
#   norm-diff     ||LB06均值|| - ||LB07均值||
#   abs-norm-diff abs(norm-diff)
#   cohen-l2      L2距离 / pooled std
metric="cosine"
min_reads_per_site=2

sequence_key="label"
limit_lb07_reads=0
limit_lb06_reads=0
seed=42

device="cuda:0"
dtype="auto"
batch_size=4
max_length=2000
pad_token_id=1
backbone_chunk_size=2000
elf_ode_steps=8
elf_ode_start_t=0.95
elf_self_cond_cfg_scale=0.0

cmap="inferno"
vmin=""
vmax=""

run_output_dir="${output_dir}/${embedding_source}_${metric}_${token_pool}"
mkdir -p "${run_output_dir}"

extra_args=()
if [[ "${limit_lb07_reads}" -gt 0 ]]; then
  extra_args+=(--limit-lb07-reads "${limit_lb07_reads}")
fi
if [[ "${limit_lb06_reads}" -gt 0 ]]; then
  extra_args+=(--limit-lb06-reads "${limit_lb06_reads}")
fi
if [[ -n "${vmin}" ]]; then
  extra_args+=(--vmin "${vmin}")
fi
if [[ -n "${vmax}" ]]; then
  extra_args+=(--vmax "${vmax}")
fi

python "${python_script}" \
  --model-name-or-path "${model_name_or_path}" \
  --lb07-jsonl "${lb07_jsonl}" \
  --lb06-jsonl "${lb06_jsonl}" \
  --output-dir "${run_output_dir}" \
  --embedding-source "${embedding_source}" \
  --sequence-key "${sequence_key}" \
  --read-depths "${read_depths}" \
  --samples-per-token "${samples_per_token}" \
  --token-pool "${token_pool}" \
  --metric "${metric}" \
  --min-reads-per-site "${min_reads_per_site}" \
  --seed "${seed}" \
  --device "${device}" \
  --dtype "${dtype}" \
  --batch-size "${batch_size}" \
  --max-length "${max_length}" \
  --pad-token-id "${pad_token_id}" \
  --backbone-chunk-size "${backbone_chunk_size}" \
  --elf-ode-steps "${elf_ode_steps}" \
  --elf-ode-start-t "${elf_ode_start_t}" \
  --elf-self-cond-cfg-scale "${elf_self_cond_cfg_scale}" \
  --cmap "${cmap}" \
  "${extra_args[@]}"
