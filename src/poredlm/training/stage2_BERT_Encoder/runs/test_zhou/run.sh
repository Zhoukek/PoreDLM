#!/usr/bin/env bash
set -euo pipefail

# 先加载MACA环境
source /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/set_env.sh

export PYTHONPATH=/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage2_BERT_Encoder:/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_API_KEY=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

torchrun --nproc_per_node=2 --master_port 29501 \
  /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage2_BERT_Encoder/stage2_bert_encoder_train.py \
  --config config.yaml 2>&1 | tee run.log
