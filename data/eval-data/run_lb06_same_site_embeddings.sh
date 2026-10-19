#!/usr/bin/env bash
set -euo pipefail
#   --hf-dlm /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training_public/stage3_DLM_train_new/scripts/runs/HF_VQE768C08A001_DNADLLM_V007_self_flow/hf_dlm_step34000 \


python /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/eval-data/extract_same_site_embeddings_new.py \
  --input-dir /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/eval-data/LB06/test \
  --output-dir /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/eval-data/LB06/embedding-space-base-flow-v011-1-050 \
  --encoder /mnt/zzbnew/poregpt/models/HF_RSQ741C12V523_DNADLLM_V011/encoder \
  --hf-dlm /mnt/zzbnew/poregpt/models/HF_RSQ741C12V523_DNADLLM_V011/hf_dlm_elf_mixed_step70000-unsharded \
  --tokenizer-backend hf-codec \
  --token-offset 128 \
  --batch-size 64 \
  --device cuda \
  --dtype float32 \
  --sampling-method ode \
  --ode-steps 1 \
  --ode-start-t 0.5 \
  --sde-steps 2 \
  --sde-start-t 0.8 \
  --sde-gamma 0.1 \
  --sde-seed 42 \
  --self-cond-cfg-scale 0.5