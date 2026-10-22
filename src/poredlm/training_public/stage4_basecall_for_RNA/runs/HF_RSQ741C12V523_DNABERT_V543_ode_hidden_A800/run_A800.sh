#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${RUN_DIR}/../../../../../.." && pwd)"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29515}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

ENV_SCRIPT="${ENV_SCRIPT:-${PROJECT_ROOT}/src/poredlm/training/set_env.sh}"
if [[ -f "${ENV_SCRIPT}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_SCRIPT}"
fi

PYTHON_BIN="${PYTHON_BIN:-/mnt/zzbnew/rnamodel/zhoukexuan/miniconda3/envs/PoreDLM/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: PoreDLM Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -c 'import torch' >/dev/null 2>&1; then
  echo "ERROR: PyTorch cannot be imported with ${PYTHON_BIN}" >&2
  exit 1
fi

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eno1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

GPU_COUNT="$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')"
if [[ "${GPU_COUNT}" -lt "${NPROC_PER_NODE}" ]]; then
  echo "ERROR: PoreDLM environment sees ${GPU_COUNT} CUDA devices; ${NPROC_PER_NODE} are required." >&2
  exit 1
fi

stage3_root="${PROJECT_ROOT}/src/poredlm/training_public/stage3_DLM_train"
stage4_root="${PROJECT_ROOT}/src/poredlm/training_public/stage4_basecall_for_RNA"

export NPROC_PER_NODE
export PYTHONPATH="${stage4_root}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1
export WANDB_API_KEY=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


# DDP 参数
ddp_backend="nccl"

# 模型参数
backbone_type="dlm"
add_cls_token=true
cls_token_id=5
batch_size=8
num_epochs=200
head_lr="1e-5"
backbone_lr="1e-5"
weight_decay="1e-5"
warmup_ratio="0.1"
min_lr="1e-6"

# 模型参数
# For feature_source=ode_hidden: -1 uses the final context layer, -2 starts ODE from the penultimate layer.
hidden_layer=-1
unfreeze_last_n_layers=0
unfreeze_target="auto"
unfreeze_context_last_n_layers=0
unfreeze_elf_last_n_layers=4
head_type="ctc"
train_decoder="ctc_viterbi"
pre_head_type="tcn"
# Set to box3 to apply a centered 3-point mean to ODE features before the TCN.
pre_head_smoothing="none"
feature_source="ode_hidden"
feature_l2_normalize=false
head_output_activation="tanh"
head_output_scale=5
train_blank_logit_bias="-0.0"
backbone_chunk_size=1540

# codebook/token-id feature fusion 参数：none / only / add / concat / gate
codebook_fusion="none"
codebook_fusion_dropout="0.1"
codebook_fusion_gate_bias="-2.0"
skip_backbone_for_codebook_only=false
# 留空时使用 HF context_encoder 的 token embedding；填 stage1 tokenizer encoder ckpt/.pt/.npy 时使用真正 VQ codebook。
codebook_feature_path="/mnt/zzbnew/poregpt/models/HF_VQE768C08A001_DNADLLM_V006/encoder"
codebook_feature_token_offset=128

# ODE 参数（feature_source="ode_hidden" 时生效）
elf_ode_steps=1
elf_ode_start_t="0.97"
elf_ode_self_cond_cfg_scale="0.5"

# SDE 参数（feature_source="sde_hidden" 时生效）
elf_sde_steps=4
elf_sde_start_t="0.85"
elf_sde_gamma="0.1"
elf_sde_self_cond_cfg_scale="1.0"
elf_sde_seed=6198

# 日志参数
use_wandb=true
wandb_project="stage4_basecall_public"
wandb_run_name="test_public_hf_dlm_basecall_100000_chunks_tcn_ctc_V543_DNA_ode_hidden_A800"

# 输入、模型和输出路径
base_model="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training_public/stage3_DLM_train/runs/HF_RSQ741C12V523_DNABERT_V543/hf_dlm_elf_mixed_with_cls_token_step65000-unsharded"
data_root="/mnt/zzbnew/poregpt/models/HF_RSQ741C12V523_DNADLLM_V011/basecall/basecall_30000_chunks"
outdir="${stage4_root}/runs/HF_RSQ741C12V523_DNABERT_V543_ode_hidden_A800"

mkdir -p "${outdir}"

wandb_args=()
if [[ "${use_wandb}" == "true" ]]; then
  wandb_args+=(
    --use_wandb
    --wandb_project "${wandb_project}"
    --wandb_run_name "${wandb_run_name}"
  )
fi

feature_norm_args=()
if [[ "${feature_l2_normalize}" == "true" ]]; then
  feature_norm_args+=(--feature_l2_normalize)
fi

codebook_feature_args=()
if [[ -n "${codebook_feature_path}" ]]; then
  codebook_feature_args+=(
    --codebook_feature_path "${codebook_feature_path}"
    --codebook_feature_token_offset "${codebook_feature_token_offset}"
  )
fi
if [[ "${skip_backbone_for_codebook_only}" == "true" ]]; then
  codebook_feature_args+=(--skip_backbone_for_codebook_only)
fi

cls_token_args=()
if [[ "${add_cls_token}" == "true" ]]; then
  cls_token_args+=(--add_cls_token --cls_token_id "${cls_token_id}")
fi

if [[ ! -d "${base_model}" ]]; then
  echo "ERROR: Base model directory not found: ${base_model}" >&2
  exit 1
fi
if [[ ! -d "${data_root}" ]]; then
  echo "ERROR: Data directory not found: ${data_root}" >&2
  exit 1
fi

echo "RUN_DIR=${RUN_DIR}"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "GPU_COUNT=${GPU_COUNT}"
echo "MASTER_PORT=${MASTER_PORT}"

cd "${RUN_DIR}"
nohup "${PYTHON_BIN}" -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="127.0.0.1:${MASTER_PORT}" \
  "${stage4_root}/Basecalling/basecaller_v8_0420/train_ddp_multifolder.py" \
  --jsonl_paths "${data_root}" \
  --model_name_or_path "${base_model}" \
  --backbone_type "${backbone_type}" \
  --tokenizer_type bwav \
  --tokenizer_token_offset 128 \
  "${cls_token_args[@]}" \
  --output_dir "${outdir}" \
  --batch_size "${batch_size}" \
  --num_epochs "${num_epochs}" \
  --lr "${head_lr}" \
  --head_lr "${head_lr}" \
  --backbone_lr "${backbone_lr}" \
  --weight_decay "${weight_decay}" \
  --warmup_ratio "${warmup_ratio}" \
  --min_lr "${min_lr}" \
  --group_by record \
  --freeze_backbone \
  --head_type "${head_type}" \
  --hidden-layer "${hidden_layer}" \
  --pre_head_type "${pre_head_type}" \
  --pre_head_smoothing "${pre_head_smoothing}" \
  --train_decoder "${train_decoder}" \
  --unfreeze_last_n_layers "${unfreeze_last_n_layers}" \
  --unfreeze_target "${unfreeze_target}" \
  --unfreeze_context_last_n_layers "${unfreeze_context_last_n_layers}" \
  --unfreeze_elf_last_n_layers "${unfreeze_elf_last_n_layers}" \
  --feature_source "${feature_source}" \
  "${feature_norm_args[@]}" \
  --train_blank_logit_bias "${train_blank_logit_bias}" \
  --codebook_fusion "${codebook_fusion}" \
  --codebook_fusion_dropout "${codebook_fusion_dropout}" \
  --codebook_fusion_gate_bias "${codebook_fusion_gate_bias}" \
  "${codebook_feature_args[@]}" \
  --head_output_activation "${head_output_activation}" \
  --head_output_scale "${head_output_scale}" \
  --backbone_chunk_size "${backbone_chunk_size}" \
  --elf_ode_steps "${elf_ode_steps}" \
  --elf_ode_start_t "${elf_ode_start_t}" \
  --elf_self_cond_cfg_scale "${elf_ode_self_cond_cfg_scale}" \
  --elf_sde_steps "${elf_sde_steps}" \
  --elf_sde_start_t "${elf_sde_start_t}" \
  --elf_sde_gamma "${elf_sde_gamma}" \
  --elf_sde_self_cond_cfg_scale "${elf_sde_self_cond_cfg_scale}" \
  --elf_sde_seed "${elf_sde_seed}" \
  --ddp_backend "${ddp_backend}" \
  --ddp_backend_fallback \
  --amp \
  --save_best \
  --log_interval 10 \
  --num_workers 8 \
  "${wandb_args[@]}" \
  > "${outdir}/nohup.out" 2>&1 &

echo "Training started. Log: ${outdir}/nohup.out"
