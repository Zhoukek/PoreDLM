#!/usr/bin/env bash
set -euo pipefail

# Add references.npy rows as the "bases" field in every matching chunks jsonl.gz.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CHUNKS_DIR="/mnt/si002562jbsc/poregpt/models/HF_VQE768C08A001_DNADLLM_V003/basecall/DNA_S1_HG00200_MIX_250F701901011_validation_1_to_50_stone/stage2/train"
REFERENCES_DIR="/mnt/si002562jbsc/poregpt/models/HF_VQE768C08A001_DNADLLM_V003/basecall/DNA_S1_HG00200_MIX_250F701901011_validation_1_to_50_stone/train/reference"
OUTPUT_DIR="/mnt/si002562jbsc/poregpt/models/HF_VQE768C08A001_DNADLLM_V003/basecall/DNA_S1_HG00200_MIX_250F701901011_validation_1_to_50_stone/basecall_data"
PATTERN="${PATTERN:-*_chunks.jsonl.gz}"
GZIP_COMPRESSLEVEL="${GZIP_COMPRESSLEVEL:-1}"

python "${SCRIPT_DIR}/merge_token_chunks_with_references.py" \
  --chunks-dir "${CHUNKS_DIR}" \
  --references-dir "${REFERENCES_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --pattern "${PATTERN}" \
  --gzip-compresslevel "${GZIP_COMPRESSLEVEL}" \
  --overwrite
