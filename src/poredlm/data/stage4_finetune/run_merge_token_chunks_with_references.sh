#!/usr/bin/env bash
set -euo pipefail

# Add references.npy rows as the "bases" field in every matching chunks jsonl.gz.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CHUNKS_DIR="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/S0_HG002_UNMOD-35g/stage2_BERT/00_S0_HG002_UNMOD_35g_model_type_0_cnn_type_0_8k_vq/train"
REFERENCES_DIR="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/S0_HG002_UNMOD-35g/stage1_tokenizer_mongo/train/reference"
OUTPUT_DIR="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/S0_HG002_UNMOD-35g/stage4_finetune/basecall_train"
PATTERN="${PATTERN:-*_chunks.jsonl.gz}"
GZIP_COMPRESSLEVEL="${GZIP_COMPRESSLEVEL:-1}"

python "${SCRIPT_DIR}/merge_token_chunks_with_references.py" \
  --chunks-dir "${CHUNKS_DIR}" \
  --references-dir "${REFERENCES_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --pattern "${PATTERN}" \
  --gzip-compresslevel "${GZIP_COMPRESSLEVEL}" \
  --overwrite
