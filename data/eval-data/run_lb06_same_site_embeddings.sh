#!/usr/bin/env bash
set -euo pipefail

python /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/eval-data/extract_same_site_embeddings.py \
  --input-dir /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/eval-data/LB06/test \
  --output-dir /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/eval-data/LB06/embedding-space-1-005 \
  --encoder /mnt/zzbnew/poregpt/models/HF_VQE768C08A001_DNADLLM_V001/encoder \
  --hf-dlm /mnt/zzbnew/poregpt/models/HF_VQE768C08A001_DNADLLM_V001/hf_dlm \
  --tokenizer-backend hf-codec \
  --token-offset 128 \
  --batch-size 64 \
  --device cuda \
  --dtype float32 \
  --ode-steps 1 \
  --ode-start-t 0.5