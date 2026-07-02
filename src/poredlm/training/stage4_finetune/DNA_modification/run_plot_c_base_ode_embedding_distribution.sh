#!/bin/bash
set -euo pipefail

project_root="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM"

source "${project_root}/src/poredlm/training/set_env.sh"

export PYTHONPATH="${project_root}/src/poredlm/training/stage4_finetune:${project_root}/src/poredlm/training:${project_root}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0,1

python_script="${project_root}/src/poredlm/training/stage4_finetune/DNA_modification/plot_c_base_ode_embedding_distribution.py"

model_name_or_path="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage3_OLMo_DLM/runs/LB07_AND_LB06/model/hf_dlm_step40000"
input_jsonl="${project_root}/data/DNA_modifiction/LB07_AND_LB06/LB06/stage2_fullapple_token1600/validation/validation_fullapple_token1600_modlabel.jsonl.gz"
output_dir="${project_root}/src/poredlm/training/stage4_finetune/DNA_modification/runs/LB07_LB06_embedding_shift/LB06_C_ode_embedding_plots_offset0_dlm"

device="cuda:0"
batch_size=4
max_length=1600
pad_token_id=1
backbone_chunk_size=1600
elf_ode_steps=4
elf_ode_start_t=0.85
elf_self_cond_cfg_scale=1.0
embedding_source="ode_hidden"
debug_compare_context_ode=1
samples_per_token=5
base_span_offset=3
modified_base_positions="14,33,52,71,90,109,128"

mkdir -p "${output_dir}"

python "${python_script}" \
  --model-name-or-path "${model_name_or_path}" \
  --jsonl "${input_jsonl}" \
  --output-dir "${output_dir}" \
  --device "${device}" \
  --batch-size "${batch_size}" \
  --max-length "${max_length}" \
  --pad-token-id "${pad_token_id}" \
  --backbone-chunk-size "${backbone_chunk_size}" \
  --elf-ode-steps "${elf_ode_steps}" \
  --elf-ode-start-t "${elf_ode_start_t}" \
  --elf-self-cond-cfg-scale "${elf_self_cond_cfg_scale}" \
  --embedding-source "${embedding_source}" \
  --samples-per-token "${samples_per_token}" \
  --base-span-offset "${base_span_offset}" \
  --modified-base-positions "${modified_base_positions}" \
  $([[ "${debug_compare_context_ode}" == "1" ]] && echo "--debug-compare-context-ode")

