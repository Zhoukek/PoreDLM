#!/usr/bin/env bash
set -e

# Masked BERT token reconstruction v3 script
# Run with:
#   bash run_masked_reconstruction_v3_linux.sh

# =========================
# 1. Environment
# =========================
source /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/set_env.sh

export PYTHONPATH=/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src:/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm:${PYTHONPATH:-}

# If tokenizer_model_v0.py needs bonito, uncomment and adjust this line:
# export PYTHONPATH=/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/dcbasecaller:${PYTHONPATH:-}

# =========================
# 2. Input paths
# =========================
STAGE1_CKPT="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage1_tokenizer/runs/03_S0_HG002_UNMOD_35g_model_type_1_cnn_type_0_distill_0.1_8k_vq_apple/models/porepgt_vqe_tokenizer.final.pth"

STAGE2_BERT="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage2_BERT_Encoder/runs/01_BERT_S0_HG002_UNMOD_35G/models/step_best"

INPUT_NPY="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/S0_HG002_UNMOD-35g/stage1_tokenizer_apple/validation/250F601844011_0_0_0_1_chunks.npy"

PYTHON_SCRIPT="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage2_BERT_Encoder/eval/masked_reconstruction_stage1_stage2_v3.py"

# =========================
# 3. Output paths
# =========================
OUTPUT_DIR="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage2_BERT_Encoder/eval/output/outputs_masked_bert_v3_2026-6-23"
OUTPUT_NPZ="${OUTPUT_DIR}/result.npz"
OUTPUT_PLOT="${OUTPUT_DIR}/compare.png"

# =========================
# 4. Runtime argsß
# =========================
INPUT_INDEX=4
INPUT_MODE="auto"
DEVICE="cuda:0"
MASK_TOKEN_ID="4"
BWAV_VOCAB_OFFSET="4"
MAX_LENGTH="600"
TOKEN_BATCH_SIZE="8000"

SIGNAL_START="0"
SIGNAL_LENGTH="6000"
MASK_TOKEN_LENGTH="4"
MASK_TOKEN_START="15"

PLOT_START="0"
PLOT_NUM_SAMPLES="200"
SEED="42"

mkdir -p "${OUTPUT_DIR}"

# =========================
# 5. Basic checks
# =========================
if [ ! -d "${STAGE1_CKPT}" ]; then
  echo "[ERROR] Stage1 checkpoint directory not found: ${STAGE1_CKPT}" >&2
  exit 1
fi

if [ ! -d "${STAGE2_BERT}" ]; then
  echo "[ERROR] Stage2 BERT directory not found: ${STAGE2_BERT}" >&2
  exit 1
fi

if [ ! -f "${INPUT_NPY}" ]; then
  echo "[ERROR] Input npy not found: ${INPUT_NPY}" >&2
  exit 1
fi

if [ ! -f "${PYTHON_SCRIPT}" ]; then
  echo "[ERROR] Python script not found: ${PYTHON_SCRIPT}" >&2
  exit 1
fi

# =========================
# 6. Run reconstruction
# =========================
echo "Running masked BERT token reconstruction v3..."
echo "STAGE1_CKPT=${STAGE1_CKPT}"
echo "STAGE2_BERT=${STAGE2_BERT}"
echo "INPUT_NPY=${INPUT_NPY}"
echo "INPUT_INDEX=${INPUT_INDEX}"
echo "INPUT_MODE=${INPUT_MODE}"
echo "SIGNAL_START=${SIGNAL_START}"
echo "SIGNAL_LENGTH=${SIGNAL_LENGTH}"
echo "MASK_TOKEN_LENGTH=${MASK_TOKEN_LENGTH}"
echo "MASK_TOKEN_START=${MASK_TOKEN_START}"
echo "OUTPUT_NPZ=${OUTPUT_NPZ}"
echo "OUTPUT_PLOT=${OUTPUT_PLOT}"

python3 "${PYTHON_SCRIPT}" \
  --stage1-ckpt "${STAGE1_CKPT}" \
  --stage2-bert "${STAGE2_BERT}" \
  --input-npy "${INPUT_NPY}" \
  --output-npz "${OUTPUT_NPZ}" \
  --output-plot "${OUTPUT_PLOT}" \
  --device "${DEVICE}" \
  --mask-token-id "${MASK_TOKEN_ID}" \
  --max-length "${MAX_LENGTH}" \
  --token-batch-size "${TOKEN_BATCH_SIZE}" \
  --input-index "${INPUT_INDEX}" \
  --input-mode "${INPUT_MODE}" \
  --signal-start "${SIGNAL_START}" \
  --signal-length "${SIGNAL_LENGTH}" \
  --mask-token-length "${MASK_TOKEN_LENGTH}" \
  --mask-token-start "${MASK_TOKEN_START}" \
  --plot-start "${PLOT_START}" \
  --plot-num-samples "${PLOT_NUM_SAMPLES}" \
  --seed "${SEED}"
