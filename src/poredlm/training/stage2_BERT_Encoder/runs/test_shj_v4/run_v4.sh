#!/usr/bin/env bash
set -euo pipefail

# Stage2 BERT v4 training:
# V3 curriculum mask + codebook vector regression loss.
# Loss = CE_token + 0.1 * MSE(codebook_vec) + 0.05 * CosineLoss(codebook_vec).

# source /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/set_env.sh

# export PYTHONPATH=/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage2_BERT_Encoder:/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src:${PYTHONPATH:-}
# export CUDA_VISIBLE_DEVICES=0,1
source /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/set_env.sh
export PYTHONPATH=/mnt/zzbnew/rnamodel/shenhaojie/PoreDLM/src/poredlm/training/stage2_BERT_Encoder:/mnt/zzbnew/rnamodel/shenhaojie/PoreDLM/src:${PYTHONPATH:-}
# export PYTHONPATH=/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage2_BERT_Encoder:/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_API_KEY=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
# If wandb.use_wandb is true in config_v4.yaml, export WANDB_API_KEY before running:
# export WANDB_API_KEY="your_wandb_key"

torchrun --nproc_per_node=2 --master_port 29504 \
  /mnt/zzbnew/rnamodel/shenhaojie/PoreDLM/src/poredlm/training/stage2_BERT_Encoder/stage2_bert_encoder_train_v4.py \
  --config config_v4.yaml 2>&1 | tee run_v4.log
