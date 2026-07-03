#!/bin/bash
set -euo pipefail

project_root="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM"
python_script="${project_root}/data/DNA_modifiction/LB07_AND_LB06/stage2_bert_train/crop_signal_by_ref_span.py"

input_jsonl="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/LB07_AND_LB06/all_data/split/test.jsonl"
output_jsonl="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/LB07_AND_LB06/stage2_bert_train/test/test_signal_cropped.jsonl"
stats_json="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/LB07_AND_LB06/stage2_bert_train/test/test.signal_cropped.stats.json"

python "${python_script}" \
  --input-jsonl "${input_jsonl}" \
  --output-jsonl "${output_jsonl}" \
  --stats-json "${stats_json}"
