#!/usr/bin/env bash
set -euo pipefail

# 常用设置：换模型改 MODEL_DIR；策略选 apple / stone；多卡可写 GPUS=(0 1 2 3)。
MODEL_DIR="${MODEL_DIR:-/mnt/zzbnew/poregpt/models/HF_VQE768C08A001_DNADLLM_V007}"
STRATEGY="${STRATEGY:-apple}"
GPUS=(1)

# 本机 Python 和依赖；从任意目录启动，结果仍保存到当前测评目录下。
PYTHON="${PYTHON:-/root/miniconda3/envs/poregpt/bin/python}"
EVAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="/tmp/model_eval/deps${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2

# 三模式：完整 chunk 的中心碱基池化、整 5-mer 池化、裁出 5-mer 单独推理。
# 默认每域选四类、每类 100 条；全部 1024 类加 --all-kmers，其他参数也可在命令后追加。
# 辅助入口补齐路径后，实际执行 NanoRepDist 的 evaluate-model。
exec python "$EVAL_DIR/scripts/evaluate_model.py" \
  --model-dir "$MODEL_DIR" \
  --strategy "$STRATEGY" \
  --modes chunk_center chunk_5mer short_5mer \
  --samples-per-kmer 100 \
  --gpus "${GPUS[@]}" \
  "$@"
