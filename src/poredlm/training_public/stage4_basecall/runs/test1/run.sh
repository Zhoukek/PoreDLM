#!/bin/bash
set -euo pipefail

# 加载运行环境
project_root="/mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM"
source "${project_root}/src/poredlm/training/set_env.sh"

stage3_root="${project_root}/src/poredlm/training_public/stage3_DLM_train"
stage4_root="${project_root}/src/poredlm/training_public/stage4_basecall"

export PYTHONPATH="${stage4_root}:${project_root}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0,1
export TORCHDYNAMO_DISABLE=1
export WANDB_API_KEY=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


# DDP 参数
nproc_per_node=2
master_port=29519
ddp_backend="nccl"

# 训练参数
batch_size=8
num_epochs=100
head_lr="1e-4"
backbone_lr="1e-5"
weight_decay="1e-5"
warmup_ratio="0.1"
min_lr="1e-6"

# 模型参数
hidden_layer=-1
unfreeze_last_n_layers=0
unfreeze_target="auto"
unfreeze_context_last_n_layers=4
unfreeze_elf_last_n_layers=0
head_type="ctc"
train_decoder="ctc_viterbi"
pre_head_type="none"
feature_source="context_hidden"
head_output_activation="tanh"
head_output_scale=5
backbone_chunk_size=1540

# ODE 参数（feature_source="ode_hidden" 时生效）
elf_ode_steps=2
elf_ode_start_t="0.98"
elf_ode_self_cond_cfg_scale="0.0"

# SDE 参数（feature_source="sde_hidden" 时生效）
elf_sde_steps=4
elf_sde_start_t="0.85"
elf_sde_gamma="0.1"
elf_sde_self_cond_cfg_scale="1.0"
elf_sde_seed=6198

# 日志参数
use_wandb=true
wandb_project="stage4_basecall_public"
wandb_run_name="test_public_hf_dlm_basecall_BERT"

# 输入、模型和输出路径
base_model="${stage3_root}/runs/test/hf_dlm"
data_root="/mnt/si002562jbsc/poregpt/models/HF_VQE768C08A001_DNADLLM_V001/basecall/DNA_S1_HG00200_MIX_250F701901011_30000_chunks/basecall_test"
outdir="${stage4_root}/runs/test1"

mkdir -p "${outdir}"

wandb_args=()
if [[ "${use_wandb}" == "true" ]]; then
  wandb_args+=(
    --use_wandb
    --wandb_project "${wandb_project}"
    --wandb_run_name "${wandb_run_name}"
  )
fi

nohup torchrun --nproc_per_node="${nproc_per_node}" --nnodes=1 --master_port="${master_port}" \
  "${stage4_root}/Basecalling/basecaller_v8_0420/train_ddp_multifolder.py" \
  --jsonl_paths "${data_root}" \
  --model_name_or_path "${base_model}" \
  --tokenizer_type bwav \
  --tokenizer_token_offset 128 \
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
  --train_decoder "${train_decoder}" \
  --unfreeze_last_n_layers "${unfreeze_last_n_layers}" \
  --unfreeze_target "${unfreeze_target}" \
  --unfreeze_context_last_n_layers "${unfreeze_context_last_n_layers}" \
  --unfreeze_elf_last_n_layers "${unfreeze_elf_last_n_layers}" \
  --feature_source "${feature_source}" \
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
