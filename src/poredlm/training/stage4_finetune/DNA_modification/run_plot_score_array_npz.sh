#!/bin/bash
set -euo pipefail

project_root="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM"

source "${project_root}/src/poredlm/training/set_env.sh"

script="${project_root}/src/poredlm/training/stage4_finetune/DNA_modification/plot_score_array_npz.py"

score_root="${project_root}/src/poredlm/training/stage4_finetune/DNA_modification/runs/LB07_LB06_embedding_shift"
npz_path="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage4_finetune/DNA_modification/runs/LB07_LB06_embedding_shift/LB07_score_arrays/250F600084012_1_8_119_2235818_6343.npz"
score_jsonl="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/LB07_AND_LB06/LB06/stage2_fullapple_token1600/validation/validation_fullapple_token1600.jsonl.gz"
output_png="${score_root}/LB07_example_score_plot.png"

top_k=50
title="LB07 embedding shift"

python "${script}" \
  --npz "${npz_path}" \
  --score-jsonl "${score_jsonl}" \
  --output "${output_png}" \
  --top-k "${top_k}" \
  --title "${title}"
