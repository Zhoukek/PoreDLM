#!/usr/bin/env bash
set -euo pipefail

# Stage2 BERT v8 epoch-driven symmetric-mask training:
# Plain BertForMaskedLM from bert_encoder_model.py.
# Each batch uses a 50% [MASK] view plus its complementary [MASK] view.

# source /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/set_env.sh

# export PYTHONPATH=/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage2_BERT_Encoder:/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src:${PYTHONPATH:-}
# export CUDA_VISIBLE_DEVICES=0,1
source /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/set_env.sh
export PYTHONPATH=/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage2_BERT_Encoder:/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0,1
# If wandb.use_wandb is true in config.yaml, export WANDB_API_KEY before running:
# export WANDB_API_KEY="your_wandb_key"
export WANDB_API_KEY=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


nohup torchrun --nproc_per_node=2 --master_port 29506 \
  /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage2_BERT_Encoder/stage2_bert_encoder_train_v8.py \
  --config config.yaml 2>&1 | tee run_v8.log
