#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM
OLMO_ROOT=${PROJECT_ROOT}/src/poredlm/training_public/stage3_DLM_train/OLMo
RUN_ROOT=${PROJECT_ROOT}/src/poredlm/training_public/stage3_DLM_train/runs/HF_RSQ741C12V523_DNABERT_V543_A800
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/config_150m.yaml}"
PYTHON_BIN="/mnt/zzbnew/rnamodel/zhoukexuan/miniconda3/envs/PoreDLM/bin/python"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29521}"

# mixed: conditional (prefix/infill) + unconditional joint training
# conditional: conditional-only training
# unconditional: pure unconditional generation training
DLM_TRAIN_MODE=${DLM_TRAIN_MODE:-mixed}
case "${DLM_TRAIN_MODE}" in
    mixed|conditional|unconditional) ;;
    *)
        echo "DLM_TRAIN_MODE must be mixed, conditional, or unconditional" >&2
        exit 2
        ;;
esac

# elf: original Stage-3 ELF objective; bert: same sampled target spans with
# Stage-2 BERT's own MLM head. BERT mode automatically unfreezes BERT.
DLM_OBJECTIVE=${DLM_OBJECTIVE:-elf}
case "${DLM_OBJECTIVE}" in
    elf) FREEZE_CONTEXT=true ;;
    bert) FREEZE_CONTEXT=false ;;
    *)
        echo "DLM_OBJECTIVE must be elf or bert" >&2
        exit 2
        ;;
esac

source "${PROJECT_ROOT}/src/poredlm/training/set_env.sh"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: PoreDLM Python not found: ${PYTHON_BIN}" >&2
    exit 1
fi

export PYTHONPATH=${OLMO_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eno1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

GPU_COUNT="$(${PYTHON_BIN} -c 'import torch; print(torch.cuda.device_count())')"
if [[ "${GPU_COUNT}" -lt "${NPROC_PER_NODE}" ]]; then
    echo "ERROR: PoreDLM environment sees ${GPU_COUNT} CUDA devices; ${NPROC_PER_NODE} are required." >&2
    exit 1
fi
export WANDB_API_KEY=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

cd "${OLMO_ROOT}"

nohup "${PYTHON_BIN}" -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="127.0.0.1:${MASTER_PORT}" \
    scripts/train_DLM.py "${CONFIG_PATH}" \
    --run_name="rsq741c12v523_${DLM_OBJECTIVE}_${DLM_TRAIN_MODE}" \
    --dlm.conditioning_mode="${DLM_TRAIN_MODE}" \
    --dlm.training_objective="${DLM_OBJECTIVE}" \
    --dlm.freeze_context_encoder="${FREEZE_CONTEXT}" \
    --save_folder="${RUN_ROOT}/model_${DLM_OBJECTIVE}_${DLM_TRAIN_MODE}" \
    > "${RUN_ROOT}/nohup_${DLM_OBJECTIVE}_${DLM_TRAIN_MODE}.out" 2>&1 &
