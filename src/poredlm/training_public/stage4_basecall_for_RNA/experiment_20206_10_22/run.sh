#!/bin/bash
set -euo pipefail

# 加载运行环境
run_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${run_dir}/../../../../../.." && pwd)"
env_script="${project_root}/src/poredlm/training/set_env.sh"
if [[ -f "${env_script}" ]]; then
  # shellcheck disable=SC1090
  source "${env_script}"
fi

stage3_root="${project_root}/src/poredlm/training_public/stage3_DLM_train_new"
stage4_root="${project_root}/src/poredlm/training_public/stage4_basecall_for_RNA"

export PYTHONPATH="${stage4_root}:${project_root}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0,1
export TORCHDYNAMO_DISABLE=1

# DDP 参数
nproc_per_node=2
master_port=29622
ddp_backend="nccl"

# 模型和训练参数
backbone_type="dlm"
add_cls_token=true
cls_token_id=5
batch_size=16
num_epochs=200
head_lr="1e-5"
backbone_lr="1e-5"
weight_decay="1e-5"
warmup_ratio="0.1"
min_lr="1e-6"

hidden_layer=-1
unfreeze_last_n_layers=0
unfreeze_target="auto"
unfreeze_context_last_n_layers=4
unfreeze_denoiser_last_n_layers=0
head_type="ctc_crf"
train_decoder="ctc_crf"
pre_head_type="tcn"
pre_head_smoothing="none"
feature_source="context_hidden"
feature_l2_normalize=false
head_output_activation="tanh"
head_output_scale=5
train_blank_logit_bias="-0.0"
backbone_chunk_size=1536

# ODE / SDE 参数
ode_steps=4
ode_start_t="0.85"
ode_self_cond_cfg_scale="1.0"
sde_steps=4
sde_start_t="0.85"
sde_gamma="0.1"
sde_self_cond_cfg_scale="1.0"
sde_seed=6198

# codebook/token-id feature fusion 参数
codebook_fusion="none"
codebook_fusion_dropout="0.1"
codebook_fusion_gate_bias="-2.0"
skip_backbone_for_codebook_only=false
codebook_feature_path=""
codebook_feature_token_offset=128

# 日志参数
use_wandb=true
wandb_project="stage4_basecall_public"
wandb_run_name="HF_RSQ741C12A529_DNABERT_V554_self_flow_context_hidden_ctc_crf"

# 输入、模型和输出路径
base_model="${stage3_root}/scripts/runs/HF_RSQ741C12A529_DNABERT_V554_self_flow/hf_dlm_step10000_new"
data_root="/mnt/zzbnew/poregpt/models/HF_RSQ741C12V523_DNADLLM_V011/basecall/basecall_30000_chunks"
outdir="${run_dir}/runs/HF_RSQ741C12A529_DNABERT_V554_self_flow_context_hidden_ctc_crf"
context_hidden_cache_dir=""

mkdir -p "${outdir}"

wandb_args=()
if [[ "${use_wandb}" == "true" ]]; then
  wandb_args+=(--use_wandb --wandb_project "${wandb_project}" --wandb_run_name "${wandb_run_name}")
fi

feature_norm_args=()
if [[ "${feature_l2_normalize}" == "true" ]]; then
  feature_norm_args+=(--feature_l2_normalize)
fi

codebook_feature_args=()
if [[ -n "${codebook_feature_path}" ]]; then
  codebook_feature_args+=(--codebook_feature_path "${codebook_feature_path}" --codebook_feature_token_offset "${codebook_feature_token_offset}")
fi
if [[ "${skip_backbone_for_codebook_only}" == "true" ]]; then
  codebook_feature_args+=(--skip_backbone_for_codebook_only)
fi

cls_token_args=()
if [[ "${add_cls_token}" == "true" ]]; then
  cls_token_args+=(--add_cls_token --cls_token_id "${cls_token_id}")
fi

context_hidden_cache_args=()
if [[ -n "${context_hidden_cache_dir}" ]]; then
  context_hidden_cache_args+=(--context_hidden_cache_dir "${context_hidden_cache_dir}")
fi

nohup torchrun --nproc_per_node="${nproc_per_node}" --nnodes=1 --master_port="${master_port}" \
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
  --lr "${head_lr}" --head_lr "${head_lr}" --backbone_lr "${backbone_lr}" \
  --weight_decay "${weight_decay}" --warmup_ratio "${warmup_ratio}" --min_lr "${min_lr}" \
  --group_by record --freeze_backbone \
  --head_type "${head_type}" \
  --hidden-layer "${hidden_layer}" \
  --pre_head_type "${pre_head_type}" \
  --pre_head_smoothing "${pre_head_smoothing}" \
  --train_decoder "${train_decoder}" \
  --unfreeze_last_n_layers "${unfreeze_last_n_layers}" \
  --unfreeze_target "${unfreeze_target}" \
  --unfreeze_context_last_n_layers "${unfreeze_context_last_n_layers}" \
  --unfreeze_denoiser_last_n_layers "${unfreeze_denoiser_last_n_layers}" \
  --feature_source "${feature_source}" \
  "${feature_norm_args[@]}" "${context_hidden_cache_args[@]}" \
  --train_blank_logit_bias "${train_blank_logit_bias}" \
  --codebook_fusion "${codebook_fusion}" \
  --codebook_fusion_dropout "${codebook_fusion_dropout}" \
  --codebook_fusion_gate_bias "${codebook_fusion_gate_bias}" \
  "${codebook_feature_args[@]}" \
  --head_output_activation "${head_output_activation}" \
  --head_output_scale "${head_output_scale}" \
  --backbone_chunk_size "${backbone_chunk_size}" \
  --ode_steps "${ode_steps}" --ode_start_t "${ode_start_t}" \
  --ode_self_cond_cfg_scale "${ode_self_cond_cfg_scale}" \
  --sde_steps "${sde_steps}" --sde_start_t "${sde_start_t}" \
  --sde_gamma "${sde_gamma}" --sde_self_cond_cfg_scale "${sde_self_cond_cfg_scale}" \
  --elf_sde_seed "${sde_seed}" \
  --ddp_backend "${ddp_backend}" --ddp_backend_fallback --amp --save_best \
  --log_interval 10 --num_workers 8 \
  "${wandb_args[@]}" \
  > "${outdir}/nohup.out" 2>&1 &

echo "Training started. Log: ${outdir}/nohup.out"
