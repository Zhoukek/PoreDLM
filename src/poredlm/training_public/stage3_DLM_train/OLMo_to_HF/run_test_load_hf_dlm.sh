#!/bin/bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM}
STAGE3_ROOT=${PROJECT_ROOT}/src/poredlm/training_public/stage3_DLM_train
SCRIPT_DIR=${STAGE3_ROOT}/OLMo_to_HF
MODEL_DIR=${MODEL_DIR:-${STAGE3_ROOT}/runs/test/hf_dlm}

source ${PROJECT_ROOT}/src/poredlm/training/set_env.sh

# Keep imports on the public Stage3 path first so the generated HF model can find ELF.
export PYTHONPATH=${STAGE3_ROOT}/ELF-pytorch-port/src:${STAGE3_ROOT}/OLMo:${PROJECT_ROOT}/src:${PYTHONPATH:-}

# Useful on environments where torch/transformers import triggers torch._dynamo/triton backend discovery.
export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

TOKEN_IDS=${TOKEN_IDS:-"2,129,130,131,3"}
ODE_STEPS=${ODE_STEPS:-4}
ODE_START_T=${ODE_START_T:-0.85}
DEVICE=${DEVICE:-cuda}

cd "${SCRIPT_DIR}"

python3 test_load_hf_dlm.py \
  --model-dir "${MODEL_DIR}" \
  --token-ids "${TOKEN_IDS}" \
  --ode-steps "${ODE_STEPS}" \
  --ode-start-t "${ODE_START_T}" \
  --device "${DEVICE}"
