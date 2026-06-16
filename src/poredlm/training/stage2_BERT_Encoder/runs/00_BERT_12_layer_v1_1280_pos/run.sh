#!/usr/bin/env bash
set -euo pipefail

# Stage2 BERT v1 epoch-driven training.
# Original Stage2 BERT MLM model/masking + finite full-epoch train loop.

source /mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/set_env.sh

export PYTHONPATH=/mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage2_BERT_Encoder:/mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/src:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0,1

# If wandb.use_wandb is true in config.yaml, export WANDB_API_KEY before running.
# export WANDB_API_KEY="your_wandb_key"
export WANDB_API_KEY=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


nohup torchrun --nproc_per_node=2 --master_port 29517 \
  /mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage2_BERT_Encoder/stage2_bert_encoder_train_v1.py \
  --config config.yaml
