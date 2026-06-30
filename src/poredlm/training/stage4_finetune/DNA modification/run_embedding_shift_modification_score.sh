#!/bin/bash
set -euo pipefail

project_root="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM"

source "${project_root}/src/poredlm/training/set_env.sh"

export PYTHONPATH="${project_root}/src:${project_root}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0

model_name_or_path="${project_root}/src/poredlm/training/stage3_OLMo_DLM/runs/02_150m_no_cond_8k_vq_context_1200/hf_dlm"
tokenizer_json="${model_name_or_path}/tokenizer.json"

lb07_jsonl="${project_root}/data/DNA_modifiction/LB07_AND_LB06/tokenized_1600/LB07.jsonl.gz"
lb06_jsonl="${project_root}/data/DNA_modifiction/LB07_AND_LB06/tokenized_1600/LB06.jsonl.gz"
outdir="${project_root}/src/poredlm/training/stage4_finetune/DNA modification/runs/LB07_LB06_embedding_shift"

device="cuda:0"
batch_size=4
max_length=1600
pad_token_id=1
unk_token_id=0
backbone_chunk_size=1600
elf_ode_steps=4
elf_ode_start_t=0.85
elf_self_cond_cfg_scale=1.0
top_k=20

mkdir -p "${outdir}"

python "${project_root}/src/poredlm/training/stage4_finetune/DNA modification/embedding_shift_modification_score.py" \
  --model-name-or-path "${model_name_or_path}" \
  --tokenizer-json "${tokenizer_json}" \
  --jsonl "${lb07_jsonl}" \
  --output-jsonl "${outdir}/LB07_embedding_shift_scores.jsonl.gz" \
  --score-array-dir "${outdir}/LB07_score_arrays" \
  --device "${device}" \
  --batch-size "${batch_size}" \
  --max-length "${max_length}" \
  --pad-token-id "${pad_token_id}" \
  --unk-token-id "${unk_token_id}" \
  --backbone-chunk-size "${backbone_chunk_size}" \
  --elf-ode-steps "${elf_ode_steps}" \
  --elf-ode-start-t "${elf_ode_start_t}" \
  --elf-self-cond-cfg-scale "${elf_self_cond_cfg_scale}" \
  --top-k "${top_k}"

python "${project_root}/src/poredlm/training/stage4_finetune/DNA modification/embedding_shift_modification_score.py" \
  --model-name-or-path "${model_name_or_path}" \
  --tokenizer-json "${tokenizer_json}" \
  --jsonl "${lb06_jsonl}" \
  --output-jsonl "${outdir}/LB06_embedding_shift_scores.jsonl.gz" \
  --score-array-dir "${outdir}/LB06_score_arrays" \
  --device "${device}" \
  --batch-size "${batch_size}" \
  --max-length "${max_length}" \
  --pad-token-id "${pad_token_id}" \
  --unk-token-id "${unk_token_id}" \
  --backbone-chunk-size "${backbone_chunk_size}" \
  --elf-ode-steps "${elf_ode_steps}" \
  --elf-ode-start-t "${elf_ode_start_t}" \
  --elf-self-cond-cfg-scale "${elf_self_cond_cfg_scale}" \
  --top-k "${top_k}"
