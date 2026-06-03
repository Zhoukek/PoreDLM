#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM
OLMO_ROOT=${PROJECT_ROOT}/src/poredlm/training/stage3_OLMo_DLM/OLMo
RUN_ROOT=${PROJECT_ROOT}/src/poredlm/training/stage3_OLMo_DLM/runs/test_zhou

source ${PROJECT_ROOT}/src/poredlm/training/set_env.sh

export PYTHONPATH=${OLMO_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0,1


python3 /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/data/stage3_OLMo_DLM/test.py
