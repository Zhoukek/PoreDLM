#!/bin/bash
set -euo pipefail

project_root="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM"

python_script="${project_root}/data/DNA_modifiction/LB07_AND_LB06/LB06/add_modification_label_to_validation.py"

input_jsonl="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/LB07_AND_LB06/LB06/stage2_fullapple_token1600/validation/validation_fullapple_token1600.jsonl.gz"
output_jsonl="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/LB07_AND_LB06/LB06/stage2_fullapple_token1600/validation/validation_fullapple_token1600_modlabel.jsonl.gz"

modified_base_positions="14,33,52,71,90,109,128"
samples_per_token=5

python "${python_script}" \
  --input-jsonl "${input_jsonl}" \
  --output-jsonl "${output_jsonl}" \
  --modified-base-positions "${modified_base_positions}" \
  --samples-per-token "${samples_per_token}"
