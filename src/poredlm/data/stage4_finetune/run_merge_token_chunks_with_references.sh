#!/usr/bin/env bash
set -euo pipefail

# Add references.npy rows as the "bases" field in every matching chunks jsonl.gz.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CHUNKS_DIR="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/S0_HG002_UNMOD-35g/stage2_BERT/03_S0_HG002_UNMOD_35g_model_type_1_cnn_type_0_distill_0.1_8k_vq_apple/temp/validation"
REFERENCES_DIR="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/S0_HG002_UNMOD-35g/stage2_BERT/03_S0_HG002_UNMOD_35g_model_type_1_cnn_type_0_distill_0.1_8k_vq_apple/temp/reference"
OUTPUT_DIR="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/S0_HG002_UNMOD-35g/stage4_finetune/temp1"
PATTERN="${PATTERN:-*_chunks.jsonl.gz}"
GZIP_COMPRESSLEVEL="${GZIP_COMPRESSLEVEL:-1}"

python "${SCRIPT_DIR}/merge_token_chunks_with_references.py" \
  --chunks-dir "${CHUNKS_DIR}" \
  --references-dir "${REFERENCES_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --pattern "${PATTERN}" \
  --gzip-compresslevel "${GZIP_COMPRESSLEVEL}" \
  --overwrite
