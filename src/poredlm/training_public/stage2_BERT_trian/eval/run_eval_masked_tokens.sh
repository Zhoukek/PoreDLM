#!/usr/bin/env bash
set -euo pipefail

# The evaluator reads data.valid_dir/train_dir and token semantics from this config.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_CONFIG="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training_public/stage2_BERT_trian/runs/test/train_config.yaml"
BERT="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training_public/stage2_BERT_trian/runs/test/models/step_235000"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output}"
DEVICE="${DEVICE:-cuda:0}"
CODEC="/mnt/zzbnew/poregpt/models/HF_VQE768C08A001_DNADLLM_V001/encoder"


ARGS=(
  --bert "${BERT}"
  --codec "${CODEC}"
  --training-config "${TRAIN_CONFIG}"
  --split "${SPLIT:-valid}"
  --sample-index "0"
  --num-samples "${NUM_SAMPLES:-1}"
  --mask-mode "${MASK_MODE:-contiguous}"
  --mask-token-start "10"
  --mask-token-length "5"
  --plot-context-tokens "${PLOT_CONTEXT_TOKENS:-20}"
  --seed "${SEED:-42}"
  --device "${DEVICE}"
  --output-dir "${OUTPUT_DIR}"
)
if [[ -n "${DATA_DIR:-}" ]]; then
  ARGS+=(--data-dir "${DATA_DIR}")
fi
if [[ -n "${MASK_PROBABILITY:-}" ]]; then
  ARGS+=(--mask-probability "${MASK_PROBABILITY}")
fi

python "${SCRIPT_DIR}/eval_masked_tokens.py" "${ARGS[@]}"
