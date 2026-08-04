#!/usr/bin/env bash
set -euo pipefail

# Override any value at launch, for example:
# BERT=/path/to/step_235000 INPUT_NPY=/path/to/chunks.npy bash run_eval_masked_tokens.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CODEC="${CODEC:-/path/to/pore_vq_codec_checkpoint}"
BERT="${BERT:-${SCRIPT_DIR}/../runs/test/models/latest}"
INPUT_NPY="${INPUT_NPY:-/path/to/signal_chunks.npy}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output}"
DEVICE="${DEVICE:-cuda:0}"

python "${SCRIPT_DIR}/eval_masked_tokens.py" \
  --codec "${CODEC}" \
  --bert "${BERT}" \
  --input-npy "${INPUT_NPY}" \
  --input-index "${INPUT_INDEX:-0}" \
  --signal-start "${SIGNAL_START:-0}" \
  --signal-length "${SIGNAL_LENGTH:-6000}" \
  --mask-mode "${MASK_MODE:-contiguous}" \
  --mask-token-start "${MASK_TOKEN_START:--1}" \
  --mask-token-length "${MASK_TOKEN_LENGTH:-4}" \
  --mask-probability "${MASK_PROBABILITY:-0.15}" \
  --seed "${SEED:-42}" \
  --device "${DEVICE}" \
  --output-dir "${OUTPUT_DIR}"
