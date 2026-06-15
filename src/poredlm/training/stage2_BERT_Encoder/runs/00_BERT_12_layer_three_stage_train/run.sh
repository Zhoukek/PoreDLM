#!/usr/bin/env bash
set -euo pipefail

# Stage2 BERT v5 training:
# V3/V4 curriculum mask + Stage1-initialized waveform decoder loss.
# Loss = CE_token + waveform_mse_weight * MSE(decoded_hidden_waveform, frozen_stage1_pred_token_waveform).

# source /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/set_env.sh

# export PYTHONPATH=/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage2_BERT_Encoder:/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src:${PYTHONPATH:-}
# export CUDA_VISIBLE_DEVICES=0,1
source /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/set_env.sh
# export PYTHONPATH=/mnt/zzbnew/rnamodel/shenhaojie/PoreDLM/src/poredlm/training/stage2_BERT_Encoder:/mnt/zzbnew/rnamodel/shenhaojie/PoreDLM/src:${PYTHONPATH:-}
export PYTHONPATH=/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage2_BERT_Encoder:/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0,1
# If wandb.use_wandb is true in config.yaml, export WANDB_API_KEY before running:
# export WANDB_API_KEY="your_wandb_key"
export WANDB_API_KEY=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


nohup torchrun --nproc_per_node=2 --master_port 29506 \
  /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage2_BERT_Encoder/stage2_bert_encoder_train.py \
  --config config.yaml
