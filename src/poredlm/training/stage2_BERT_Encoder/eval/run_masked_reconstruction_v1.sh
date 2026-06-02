#!/usr/bin/env bash
set -e

# Masked BERT reconstruction script
# Run with:
#   bash run_masked_reconstruction_percentage_linux.sh

# =========================
# 1. Environment
# =========================
# source /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/set_env.sh

export PYTHONPATH=/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src:/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm:${PYTHONPATH:-}

# If tokenizer_model_v0.py needs bonito, uncomment and adjust this line:
# export PYTHONPATH=/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/dcbasecaller:${PYTHONPATH:-}

# =========================
# 2. Input paths
# =========================
STAGE1_CKPT="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage1_tokenizer/runs/01_without_modfiction_model_type1_cnn_type_0_distill_1.0_8k/models/porepgt_vqe_tokenizer.final.pth"

STAGE2_BERT="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage2_BERT_Encoder/runs/test_zhou/models/stage2_BERT_Encoder/step_best"

INPUT_NPY="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/without_modifiction/stage1_tokenizer/validation/chunks_validation.npy"

PYTHON_SCRIPT="/mnt/zzbnew/rnamodel/shenhaojie/PoreDLM/test/masked_reconstruction_stage1_stage2.py"

# =========================
# 3. Output paths
# =========================
OUTPUT_DIR="/mnt/zzbnew/rnamodel/shenhaojie/PoreDLM/test/output/outputs_masked_bert"
OUTPUT_NPZ="${OUTPUT_DIR}/result.npz"
OUTPUT_PLOT="${OUTPUT_DIR}/compare.png"

# =========================
# 4. Runtime args
# =========================
INPUT_INDEX=1
INPUT_MODE="auto"
DEVICE="cuda:0"
MASK_TOKEN_ID="4"
MAX_LENGTH="512"
TOKEN_BATCH_SIZE="8000"

SIGNAL_START="500"
SIGNAL_LENGTH="1000"
MASK_PERCENTAGE="15"

PLOT_START="500"
PLOT_NUM_SAMPLES="1000"
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

PYTHON_HELP="$(python3 "${PYTHON_SCRIPT}" -h 2>&1 || true)"

if ! printf "%s\n" "${PYTHON_HELP}" | grep -q -- "--input-index"; then
  echo "[ERROR] ${PYTHON_SCRIPT} is an old version and does not support --input-index." >&2
  echo "[ERROR] Please copy the updated masked_reconstruction_stage1_stage2.py to the Linux server first." >&2
  exit 1
fi

if ! printf "%s\n" "${PYTHON_HELP}" | grep -q -- "--signal-start"; then
  echo "[ERROR] ${PYTHON_SCRIPT} is an old version and does not support --signal-start/--signal-length." >&2
  echo "[ERROR] Please copy the updated masked_reconstruction_stage1_stage2.py to the Linux server first." >&2
  exit 1
fi

if ! printf "%s\n" "${PYTHON_HELP}" | grep -q -- "--mask-percentage"; then
  echo "[ERROR] ${PYTHON_SCRIPT} is an old version and does not support --mask-percentage." >&2
  echo "[ERROR] Please copy the updated masked_reconstruction_stage1_stage2.py to the Linux server first." >&2
  exit 1
fi

# =========================
# 6. Run reconstruction
# =========================
echo "Running masked BERT reconstruction..."
echo "STAGE1_CKPT=${STAGE1_CKPT}"
echo "STAGE2_BERT=${STAGE2_BERT}"
echo "INPUT_NPY=${INPUT_NPY}"
echo "INPUT_INDEX=${INPUT_INDEX}"
echo "INPUT_MODE=${INPUT_MODE}"
echo "SIGNAL_START=${SIGNAL_START}"
echo "SIGNAL_LENGTH=${SIGNAL_LENGTH}"
echo "MASK_PERCENTAGE=${MASK_PERCENTAGE}"
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
  --mask-percentage "${MASK_PERCENTAGE}" \
  --plot-start "${PLOT_START}" \
  --plot-num-samples "${PLOT_NUM_SAMPLES}" \
  --seed "${SEED}"
