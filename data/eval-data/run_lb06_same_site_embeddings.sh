#!/usr/bin/env bash
set -euo pipefail
#   --hf-dlm /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training_public/stage3_DLM_train_new/scripts/runs/HF_VQE768C08A001_DNADLLM_V007_self_flow/hf_dlm_step34000 \


python /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/eval-data/extract_same_site_embeddings_new.py \
  --input-dir /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/eval-data/LB06/after-filter \
  --output-dir /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/eval-data/LB06/embedding-space-self-flow-v006-2-098 \
  --encoder /mnt/zzbnew/poregpt/models/HF_VQE768C08A001_DNADLLM_V006/encoder \
  --hf-dlm /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training_public/stage3_DLM_train_new/scripts/runs/HF_VQE768C08A001_DNADLLM_V006_self_flow/hf_dlm_step96000 \
  --tokenizer-backend hf-codec \
  --token-offset 128 \
  --batch-size 64 \
  --device cuda \
  --dtype float32 \
  --sampling-method ode \
  --ode-steps 2 \
  --ode-start-t 0.98 \
  --sde-steps 2 \
  --sde-start-t 0.8 \
  --sde-gamma 0.1 \
  --sde-seed 42 \
  --self-cond-cfg-scale 1.0