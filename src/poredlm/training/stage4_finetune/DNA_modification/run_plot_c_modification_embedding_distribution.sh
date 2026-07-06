#!/bin/bash
set -euo pipefail

# 先加载MACA环境
source /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/set_env.sh

export PYTHONPATH=/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src:/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm:/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0,1

project_root="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM"

python_script="${project_root}/src/poredlm/training/stage4_finetune/DNA_modification/plot_c_modification_embedding_distribution.py"

model_name_or_path="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage3_OLMo_DLM/runs/LB07_AND_LB06_MIX/hf_dlm"
jsonl="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/LB07_AND_LB06/stage4_modification/validation_seq1_to_seq17_ref_target_cropped_token_c_modlabel.jsonl.gz"
output_dir="${project_root}/src/poredlm/training/stage4_finetune/DNA_modification/outputs/c_modification_embedding"

# plot_mode:
#   per-read   每条 read 单独一张图
#   aggregate  多条 read 聚合到一张或多张图
#   both       两种都画
plot_mode="per-read"

# aggregate 模式下每张图放多少条 read；0 表示所有选中的 read 画到一张图
reads_per_plot=50
limit_reads=500

# embedding_source:
#   bert = context_hidden，BERT/context encoder 输出
#   dlm  = ode_hidden，DLM/ELF ODE refinement 后输出
embedding_source="dlm"

device="cuda:1"
dtype="auto"
batch_size=32
max_length=2000
pad_token_id=1
backbone_chunk_size=2000
elf_ode_steps=16
elf_ode_start_t=0.5
elf_self_cond_cfg_scale=1.0

# 点太多时可以抽样；0 表示不抽样
max_unmodified_points=20000
max_modified_points=0
seed=42

mkdir -p "${output_dir}"

python "${python_script}" \
  --model-name-or-path "${model_name_or_path}" \
  --jsonl "${jsonl}" \
  --output-dir "${output_dir}/${embedding_source}_${plot_mode}" \
  --plot-mode "${plot_mode}" \
  --reads-per-plot "${reads_per_plot}" \
  --limit-reads "${limit_reads}" \
  --embedding-source "${embedding_source}" \
  --device "${device}" \
  --dtype "${dtype}" \
  --batch-size "${batch_size}" \
  --max-length "${max_length}" \
  --pad-token-id "${pad_token_id}" \
  --backbone-chunk-size "${backbone_chunk_size}" \
  --elf-ode-steps "${elf_ode_steps}" \
  --elf-ode-start-t "${elf_ode_start_t}" \
  --elf-self-cond-cfg-scale "${elf_self_cond_cfg_scale}" \
  --max-unmodified-points "${max_unmodified_points}" \
  --max-modified-points "${max_modified_points}" \
  --seed "${seed}"
