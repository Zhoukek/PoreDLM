#!/bin/bash
set -euo pipefail

project_root="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM"

source "${project_root}/src/poredlm/training/set_env.sh"

export PYTHONPATH=/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage4_finetune:/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training:/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_API_KEY=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

model_name_or_path="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage3_OLMo_DLM/runs/LB07_AND_LB06/model/hf_dlm_step40000"
tokenizer_json="${model_name_or_path}/tokenizer.json"

lb07_jsonl="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/LB07_AND_LB06/LB07/stage2_fullapple_token1600/validation/validation_fullapple_token1600.jsonl.gz"
lb06_jsonl="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/LB07_AND_LB06/LB06/stage2_fullapple_token1600/validation/validation_fullapple_token1600.jsonl.gz"
outdir="${project_root}/src/poredlm/training/stage4_finetune/DNA_modification/runs/LB07_LB06_embedding_shift"

device="cuda:0"
batch_size=4
max_length=1600
pad_token_id=1
unk_token_id=0
backbone_chunk_size=1600
elf_ode_steps=4
elf_ode_start_t=0.85
elf_self_cond_cfg_scale=1.0
top_k=30

mkdir -p "${outdir}"

# python "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage4_finetune/DNA_modification/embedding_shift_modification_score.py" \
#   --model-name-or-path "${model_name_or_path}" \
#   --tokenizer-json "${tokenizer_json}" \
#   --jsonl "${lb07_jsonl}" \
#   --output-jsonl "${outdir}/LB07_embedding_shift_scores.jsonl.gz" \
#   --score-array-dir "${outdir}/LB07_score_arrays" \
#   --device "${device}" \
#   --batch-size "${batch_size}" \
#   --max-length "${max_length}" \
#   --pad-token-id "${pad_token_id}" \
#   --unk-token-id "${unk_token_id}" \
#   --backbone-chunk-size "${backbone_chunk_size}" \
#   --elf-ode-steps "${elf_ode_steps}" \
#   --elf-ode-start-t "${elf_ode_start_t}" \
#   --elf-self-cond-cfg-scale "${elf_self_cond_cfg_scale}" \
#   --top-k "${top_k}"

python "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage4_finetune/DNA_modification/embedding_shift_modification_score.py" \
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
