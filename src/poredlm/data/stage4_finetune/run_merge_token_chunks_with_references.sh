#!/usr/bin/env bash
set -euo pipefail

# Merge Stage2 token chunks with reference arrays into Stage4 jsonl.gz files.
# Directory layout:
#   train/
#     <prefix>_chunks.npy
#     reference/
#       <prefix>_references.npy
#       <prefix>_reference_lengths.npy

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_DIR="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/S0_HG002_UNMOD-35g/stage1_tokenizer_mongo/test"
REFERENCE_DIR="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/S0_HG002_UNMOD-35g/stage1_tokenizer_mongo/test/reference"
OUTPUT_DIR="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/S0_HG002_UNMOD-35g/stage4_finetune"
WORKERS="${WORKERS:-4}"
GZIP_COMPRESSLEVEL="${GZIP_COMPRESSLEVEL:-1}"

python "${SCRIPT_DIR}/merge_token_chunks_with_references.py" \
  --input-dir "${INPUT_DIR}" \
  --reference-dir "${REFERENCE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --workers "${WORKERS}" \
  --gzip-compresslevel "${GZIP_COMPRESSLEVEL}" \
  --overwrite
