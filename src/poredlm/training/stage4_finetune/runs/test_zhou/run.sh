#!/bin/bash 

source /mnt/zzbnew/rnamodel/zhoukexuan/poregpt/poregpt/workflows/set_env.sh  # 你之前那个脚本

export PYTHONPATH=/mnt/zzbnew/rnamodel/zhoukexuan/poregpt/poregpt/dcbasecaller:/mnt/zzbnew/rnamodel/zhoukexuan/poregpt/poregpt/workflows/olmo_workflow/OLMo:/mnt/zzbnew/rnamodel/zhoukexuan/poregpt:${PYTHONPATH:-}

model_name="test-haojieshen-model-type26-cnn_type13_teacher_model_distill0.1_VQ_64k_lemon"
nproc_per_node=4
batch_size=32
num_epochs=500
lr="1e-3"
weight_decay="1e-4"
warmup_ratio="0.4"
min_lr="1e-5"
hidden_layer=-1
unfreeze_last_n_layers=4
head_type="ctc"
train_decode="ctc_viterbi"
pre_head_type="tcn"
feature_source="hidden"
head_output_activation="tanh"
head_output_scale=5
ddp_backend="nccl" 


wandb_project="basecall"
wandb_run_name="dna32g_${model_name}_unfreeze${unfreeze_last_n_layers}_bsz${batch_size}_for_codebook_aware_test-v0.1.1"

base_model="/mnt/zzbnew/rnamodel/shenhaojie/signalDNAmodel/${model_name}/base-v0.1"
data_root="/mnt/zzbnew/rnamodel/shenhaojie/signalDNAmodel/${model_name}/basecall"
outdir="/mnt/zzbnew/rnamodel/shenhaojie/signalDNAmodel/${model_name}/${wandb_run_name}"

mkdir -p "${outdir}"


export WANDB_API_KEY=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


nohup torchrun --nproc_per_node="${nproc_per_node}" --nnodes=1 \
  -m scripts.basecaller_v8_0420.train_ddp_multifolder \
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
  --feature_source "${feature_source}" \
  --head_output_activation "${head_output_activation}" \
  --head_output_scale "${head_output_scale}" \
  --ddp_backend nccl \
  --save_best \
  --use_wandb \
  --wandb_project "${wandb_project}" \
  --wandb_run_name "${wandb_run_name}" \
  --log_interval 100 \
  --num_workers 0 \
  > "${outdir}/nohup.out" 2>&1 &
