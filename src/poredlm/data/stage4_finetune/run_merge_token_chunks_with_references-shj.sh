#!/usr/bin/env bash
set -euo pipefail

# Add references.npy rows as the "bases" field in every matching chunks jsonl.gz.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHUNKS_DIR="/mnt/zzbnew/rnamodel/shenhaojie/data/ONT-R10-data"
REFERENCES_DIR="/mnt/zzbnew/rnamodel/shenhaojie/data/ONT-R10-data"
OUTPUT_DIR="//mnt/zzbnew/rnamodel/shenhaojie/data/ONT-R10-basecall"
PATTERN="${PATTERN:-*_chunks.jsonl.gz}"
GZIP_COMPRESSLEVEL="${GZIP_COMPRESSLEVEL:-1}"

python "${SCRIPT_DIR}/merge_token_chunks_with_references.py" \
  --chunks-dir "${CHUNKS_DIR}" \
  --references-dir "${REFERENCES_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --pattern "${PATTERN}" \
  --gzip-compresslevel "${GZIP_COMPRESSLEVEL}" \
  --overwrite
