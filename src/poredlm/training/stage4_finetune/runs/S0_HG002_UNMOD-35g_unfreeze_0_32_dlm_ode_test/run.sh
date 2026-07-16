#!/bin/bash 
set -euo pipefail

# 先加载MACA环境
source /mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/set_env.sh

export PYTHONPATH=/mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage4_finetune:/mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/src:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_API_KEY=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

nproc_per_node=2
batch_size=8
num_epochs=100
lr="1e-4"
weight_decay="1e-5"
warmup_ratio="0.1"
min_lr="1e-6"
hidden_layer=-1
unfreeze_last_n_layers=0
unfreeze_target="auto"
unfreeze_context_last_n_layers=0
unfreeze_elf_last_n_layers=4
head_type="ctc"
train_decode="ctc_viterbi"
pre_head_type="none"
feature_source="ode_hidden"
elf_ode_steps=2
elf_ode_start_t="0.95"
elf_self_cond_cfg_scale="0.0"
head_output_activation="tanh"
head_output_scale=5
backbone_chunk_size=1200
ddp_backend="nccl" 


wandb_project="stage4_finetune"
wandb_run_name="S0_HG002_UNMOD-35g_unfreeze_0_32_dlm_ode_test_mx_28000_chunks"

base_model="/mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage3_OLMo_DLM/runs/02_150m_no_cond_8k_vq_context_1200/hf_dlm"
data_root="/mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/S0_HG002_UNMOD-35g/stage4_finetune/temp"
outdir="/mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage4_finetune/runs/S0_HG002_UNMOD-35g_unfreeze_0_32_dlm_ode_test"

mkdir -p "${outdir}"

nohup torchrun --nproc_per_node="${nproc_per_node}" --nnodes=1 --master_port 29515 \
  /mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage4_finetune/Basecalling/basecaller_v8_0420/train_ddp_multifolder.py \
  --jsonl_paths "${data_root}" \
  --model_name_or_path "${base_model}" \
  --output_dir "${outdir}" \
  --batch_size "${batch_size}" \
  --num_epochs "${num_epochs}" \
  --lr "${lr}" \
  --weight_decay "${weight_decay}" \
  --warmup_ratio "${warmup_ratio}" \
  --min_lr "${min_lr}" \
  --group_by record \
  --freeze_backbone \
  --head_type "${head_type}" \
  --hidden-layer "${hidden_layer}" \
  --pre_head_type "${pre_head_type}" \
  --train_decoder "${train_decode}" \
  --unfreeze_last_n_layers "${unfreeze_last_n_layers}" \
  --unfreeze_target "${unfreeze_target}" \
  --unfreeze_context_last_n_layers "${unfreeze_context_last_n_layers}" \
  --unfreeze_elf_last_n_layers "${unfreeze_elf_last_n_layers}" \
  --feature_source "${feature_source}" \
  --elf_ode_steps "${elf_ode_steps}" \
  --elf_ode_start_t "${elf_ode_start_t}" \
  --elf_self_cond_cfg_scale "${elf_self_cond_cfg_scale}" \
  --head_output_activation "${head_output_activation}" \
  --head_output_scale "${head_output_scale}" \
  --backbone_chunk_size "${backbone_chunk_size}" \
  --ddp_backend nccl \
  --save_best \
  --use_wandb \
  --wandb_project "${wandb_project}" \
  --wandb_run_name "${wandb_run_name}" \
  --log_interval 1 \
  --num_workers 8 \
  > "${outdir}/nohup.out" 2>&1 &
