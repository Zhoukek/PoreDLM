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

# compare_mode:
#   c-mod-sites  LB06修饰C vs LB07同位点未修饰C
#   c-mod-5mer   以修饰C为中心取5-mer，mean pooling后画LB06 vs LB07
#   lb06-c-labels  LB06内部正常C(label=1) vs 修饰C(label=2)
#   base         普通碱基分布，按LB07/LB06作为标签，不使用修饰标签
#   single-dataset-base  单独看LB07或LB06内部A/C/G/T碱基分布
compare_mode="c-mod-5mer"

# compare_mode=base时生效，比如 "A"、"T"、"A,T,G"、"A,C,G,T"
base_types="A,T,G"
samples_per_token=5

# base/single-dataset-base模式下生效：
#   center-unique 每个碱基只取span中心token，若多个碱基中心落到同一个token则跳过该token
#   all-overlap   旧逻辑，取span覆盖到的所有token，可能有重叠复用
base_token_mode="center-unique"

# compare_mode=c-mod-sites时生效：
#   separate  LB06和LB07分别forward后取embedding
#   mixed     LB06和LB07交错放进同一批batch forward后取embedding
c_mod_site_batch_mode="separate"

# PCA图上标注修饰位点：
#   none                不标注
#   modified-centroids  每个修饰位点标注一次，标在该位点点云中心
#   modified-points     每个修饰C点都标注位点，点多时会比较挤
annotate_site_labels="modified-points"

# compare_mode=single-dataset-base时生效：LB07 / LB06 / both
single_dataset="both"

# compare_mode=c-mod-5mer时生效：5-mer覆盖到的token去重后至少保留多少个才画
min_kmer_tokens=5

# compare_mode=c-mod-5mer时生效：
#   class  LB06修饰点统一用一种颜色
#   read   LB06修饰点按read_id分别上色
modified_color_by="class"
# modified_color_by=read时，0表示不显示read_id图注
max_read_legend_items=0

# color_by:
#   dataset      同一张图只区分 LB07 / LB06
#   dataset_base 同一张图区分 LB07-A / LB06-A / LB07-T / LB06-T 等组合
color_by="dataset"

# sequence_key:
#   label 默认按 seq_1 ... seq_17 分组
#   ref   按 ref 字符串分组
#   seq   按 seq 字符串分组
sequence_key="label"

# plot_mode:
#   all          所有 seq 聚合成一张图
#   per-sequence 每个 seq 单独一张图，七个修饰位点在同一张图里
#   per-site     每个 seq 的每个修饰位点单独一张图
#   per-read     LB06每条read单独一张图，图内按修饰位点分组
#   lb06-per-sequence-sites  只画LB06，每个seq一张图，图内按七个修饰位点分组
#   both         all + per-sequence
#   all-modes    all + per-sequence + per-site + per-read + lb06-per-sequence-sites
plot_mode="per-site"

# compare_mode=c-mod-sites且plot_mode=per-read/all-modes时生效；0表示画全部LB06 reads
max_lb06_read_plots=0

limit_lb07_reads=0
limit_lb06_reads=0

device="cuda:0"
dtype="auto"
batch_size=4
max_length=2000
pad_token_id=1
backbone_chunk_size=2000
elf_ode_steps=0
elf_ode_start_t=1
elf_self_cond_cfg_scale=0

# 点太多时可以抽样；0 表示不抽样
# compare_mode=lb06-c-labels时，max_lb07_points控制正常C，max_lb06_points控制修饰C
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
  --output-dir "${output_dir}/${embedding_source}_step_${elf_ode_steps}_t_${elf_ode_start_t}_elf_self_cond_cfg_scale_${elf_self_cond_cfg_scale}_${compare_mode}" \
  --compare-mode "${compare_mode}" \
  --base-types "${base_types}" \
  --single-dataset "${single_dataset}" \
  --samples-per-token "${samples_per_token}" \
  --base-token-mode "${base_token_mode}" \
  --c-mod-site-batch-mode "${c_mod_site_batch_mode}" \
  --annotate-site-labels "${annotate_site_labels}" \
  --modified-color-by "${modified_color_by}" \
  --max-read-legend-items "${max_read_legend_items}" \
  --color-by "${color_by}" \
  --embedding-source "${embedding_source}" \
  --sequence-key "${sequence_key}" \
  --plot-mode "${plot_mode}" \
  --max-lb06-read-plots "${max_lb06_read_plots}" \
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
  --min-kmer-tokens "${min_kmer_tokens}" \
  --seed "${seed}" \
  "${extra_args[@]}"
