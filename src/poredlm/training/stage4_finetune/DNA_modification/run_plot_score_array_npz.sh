#!/bin/bash
set -euo pipefail

project_root="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM"

source "${project_root}/src/poredlm/training/set_env.sh"

script="${project_root}/src/poredlm/training/stage4_finetune/DNA_modification/plot_score_array_npz.py"

score_root="${project_root}/src/poredlm/training/stage4_finetune/DNA_modification/runs/LB07_LB06_embedding_shift"
npz_path="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage4_finetune/DNA_modification/runs/LB07_LB06_embedding_shift/LB06_score_arrays/250F600084012_7_76_4545_124093044_17861.npz"
score_jsonl="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage4_finetune/DNA_modification/runs/LB07_LB06_embedding_shift/LB06_embedding_shift_scores.jsonl.gz"
label_jsonl="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/LB07_AND_LB06/LB06/stage2_fullapple_token1600/validation/validation_fullapple_token1600_modlabel.jsonl.gz"
output_png="${score_root}/LB06_example_score_plot.png"

top_k=10
title="LB06 embedding shift"

python "${script}" \
  --npz "${npz_path}" \
  --score-jsonl "${score_jsonl}" \
  --label-jsonl "${label_jsonl}" \
  --output "${output_png}" \
  --top-k "${top_k}" \
  --title "${title}"
