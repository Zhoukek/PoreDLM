#!/usr/bin/env bash
set -Eeuo pipefail

# V001 ELF-1 comparison on four A100-80GB GPUs, with W&B.
# Architecture: PoreDLM V001 ODE hidden (only the final ELF block trainable)
#               -> TCN -> state-4 CTC-CRF.
# For a fair V001/V003 comparison this uses the exact same stone corpus and
# file-level 90/5/5 split as run_v003_ctc_crf_elf1_4gpu.sh.
# Train records: 1,553,729; global batch: 4 x 64 = 256; steps/epoch: 6,070.

export WANDB_API_KEY=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


project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model_dir="${MODEL_DIR:-/mnt/zzbnew/poregpt/models/HF_VQE768C08A001_DNADLLM_V002/hf_dlm}"
data_root="${DATA_ROOT:-/mnt/zzbnew/poregpt/models/HF_VQE768C08A001_DNADLLM_V001/basecall/DNA_S1_HG00200_MIX_250F701901011_800000_chunks/basecall_data}"
result_root="${project_root}/01.result/HF_VQE768C08A001_DNADLLM_V002"
output_dir="${OUTPUT_DIR:-${result_root}/ctc_crf_state4_bos_eos_elf4_apple_2gpu_b64_e5}"
pretrained_ckpt="${PRETRAINED_CKPT:-}"

env \
  MODEL_DIR="${model_dir}" \
  DATA_ROOT="${data_root}" \
  OUTPUT_DIR="${output_dir}" \
  GPU_IDS="${GPU_IDS:-0,1}" \
  NPROC_PER_NODE="${NPROC_PER_NODE:-2}" \
  MASTER_PORT="${MASTER_PORT:-29834}" \
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-v002-ctc-crf-elf1-2gpu}" \
  GROUP_BY=file \
  TRAIN_RATIO=0.90 \
  VAL_RATIO=0.05 \
  TEST_RATIO=0.05 \
  SEED="${SEED:-42}" \
  PRETRAINED_CKPT="${pretrained_ckpt}" \
  PRETRAINED_STRICT=1 \
  AUTO_RESUME="${AUTO_RESUME:-1}" \
  USE_POREDLM_BOUNDARY_TOKENS=1 \
  POREDLM_BOUNDARY_MODE=bos_eos \
  TRAIN_REFERENCE_TRIM_BASES=0 \
  FEATURE_SOURCE=hidden \
  HIDDEN_LAYER=-1 \
  DLM_OUTPUT=ode \
  DLM_ODE_STEPS=2 \
  DLM_ODE_START_T=0.98 \
  DLM_ODE_SELF_COND_CFG_SCALE=0.0 \
  PRE_HEAD_TYPE=tcn \
  HEAD_TYPE=ctc_crf \
  TRAIN_DECODER=ctc_crf \
  CTC_CRF_STATE_LEN=4 \
  CTC_CRF_BLANK_SCORE=2.0 \
  CTC_CRF_DECODE_BLANK_SCORE="${CTC_CRF_DECODE_BLANK_SCORE:-1.86}" \
  CTC_CRF_MOVE_LOSS_WEIGHT=0 \
  POREDLM_UNFREEZE_CONTEXT_LAST_N_LAYERS=0 \
  POREDLM_UNFREEZE_ELF_LAST_N_BLOCKS=4 \
  POREDLM_MEMORY_EFFICIENT_ATTENTION=1 \
  BATCH_SIZE="${BATCH_SIZE:-64}" \
  NUM_WORKERS="${NUM_WORKERS:-8}" \
  EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-1}" \
  NUM_EPOCHS="${NUM_EPOCHS:-5}" \
  STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-6070}" \
  HEAD_LR="${HEAD_LR:-1e-4}" \
  PRE_HEAD_LR="${PRE_HEAD_LR:-1e-4}" \
  BACKBONE_LR="${BACKBONE_LR:-5e-7}" \
  WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}" \
  WARMUP_RATIO="${WARMUP_RATIO:-0.05}" \
  MIN_LR="${MIN_LR:-1e-6}" \
  USE_AMP=0 \
  EVAL_INTERVAL=0 \
  VAL_MAX_READS="${VAL_MAX_READS:-4096}" \
  TEST_MAX_READS="${TEST_MAX_READS:-4096}" \
  SAVE_EVERY=1 \
  SAVE_BEST=1 \
  LOG_INTERVAL="${LOG_INTERVAL:-25}" \
  USE_WANDB=1 \
  WANDB_MODE="${WANDB_MODE:-online}" \
  WANDB_PROJECT="${WANDB_PROJECT:-poredlm_basecall}" \
  WANDB_RUN_NAME="${WANDB_RUN_NAME:-V002_ctc_crf_s4_elf4_apple_2gpu_b64_e5_evalfix}" \
  BACKGROUND=0 \
  FOREGROUND_TEE="${FOREGROUND_TEE:-1}" \
  DRY_RUN="${DRY_RUN:-0}" \
  "${project_root}/run_ctc_crf_800k_4gpu.sh"
