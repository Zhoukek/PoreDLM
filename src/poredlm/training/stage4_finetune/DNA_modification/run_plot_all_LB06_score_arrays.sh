#!/bin/bash
set -euo pipefail
shopt -s nullglob

project_root="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM"

source "${project_root}/src/poredlm/training/set_env.sh"

script="${project_root}/src/poredlm/training/stage4_finetune/DNA_modification/plot_score_array_npz.py"

score_root="${project_root}/src/poredlm/training/stage4_finetune/DNA_modification/runs/LB07_LB06_embedding_shift"
npz_dir="${score_root}/LB06_score_arrays"
output_dir="${score_root}/LB06_score_plots"
log_dir="${output_dir}/logs"

score_jsonl="${score_root}/LB06_embedding_shift_scores.jsonl.gz"
label_jsonl="${project_root}/data/DNA_modifiction/LB07_AND_LB06/LB06/stage2_fullapple_token1600/validation/validation_fullapple_token1600_modlabel.jsonl.gz"

top_k=8
title_prefix="LB06 embedding shift"

mkdir -p "${output_dir}" "${log_dir}"

npz_files=("${npz_dir}"/*.npz)
total=${#npz_files[@]}
if [[ "${total}" -eq 0 ]]; then
  echo "No npz files found under: ${npz_dir}"
  exit 1
fi

echo "Found ${total} npz files under: ${npz_dir}"
echo "Output png dir: ${output_dir}"
echo "Log dir: ${log_dir}"

index=0
for npz_path in "${npz_files[@]}"; do
  index=$((index + 1))
  stem="$(basename "${npz_path}" .npz)"
  output_png="${output_dir}/${stem}.png"
  log_path="${log_dir}/${stem}.log"

  echo "[${index}/${total}] plotting ${stem}"
  python "${script}" \
    --npz "${npz_path}" \
    --score-jsonl "${score_jsonl}" \
    --label-jsonl "${label_jsonl}" \
    --output "${output_png}" \
    --top-k "${top_k}" \
    --title "${title_prefix}: ${stem}" \
    > "${log_path}" 2>&1
done

echo "Done. Plots saved to: ${output_dir}"
