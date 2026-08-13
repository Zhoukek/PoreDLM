#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="${repo_root}/script/dcbasecaller:${repo_root}/script/basecallx:${PYTHONPATH:-}"

model_name="HF_RSQ742C12A511_DNA595G_C02K"
data_root="${repo_root}/00.data/${model_name}"
base_model="/mnt/zzbnew/poregpt/models/${model_name}/base"
run_time="$(date +%Y%m%d_%H%M%S)"
outdir="${repo_root}/01.result/${model_name}/basecallx_smoke_${run_time}"

mkdir -p "${outdir}"

accelerate launch --num_processes 1 -m basecallx.train \
  --jsonl_paths "${data_root}" \
  --model_name_or_path "${base_model}" \
  --output_dir "${outdir}" \
  --head_type ctc \
  --train_decoder ctc_viterbi \
  --pre_head_type none \
  --feature_source hidden \
  --group_by record \
  --streaming \
  --max_steps_per_epoch 100 \
  --batch_size 32 \
  --num_epochs 1 \
  --num_workers 0
