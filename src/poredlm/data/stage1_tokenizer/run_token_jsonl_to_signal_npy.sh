#!/usr/bin/env bash
set -euo pipefail

# ======================== 参数配置区 ========================
# 输入可以是单个 .jsonl.gz 文件，也可以是包含多个 .jsonl.gz 的目录。
INPUT_PATH="/mnt/zzbnew/poregpt/models/HF_VQE768C08A001_DNADLLM_V003/basecall/DNA_S1_HG00200_MIX_250F701901011_validation_1_to_50_stone/basecall_data"

# 重建后的 .npy 文件保存目录。
OUTPUT_DIR="/mnt/zzbnew/poregpt/models/HF_VQE768C08A001_DNADLLM_V003/basecall/DNA_S1_HG00200_MIX_250F701901011_validation_1_to_50_stone/reconstructed_signal"

# 支持 HF codec 目录、Stage1 run 目录或旧版 accelerate checkpoint 目录。
MODEL_PATH="/mnt/zzbnew/poregpt/models/HF_VQE768C08A001_DNADLLM_V003/encoder"

# 可设置为 cuda、cuda:0、cuda:1 或 cpu。
DEVICE="cuda:0"

# 根据显存大小调整。显存不足时减小该值。
BATCH_SIZE=32

# 每条原始 chunk 有 6000 个采样点。decoder 原始卷积输出可能是 5997，脚本会在右侧补零到该长度。
# 如果设为空字符串，Python 脚本会按“token 数 × tokenizer stride”自动推断。
SIGNAL_LENGTH=6000

# reference 固定长度；不足部分在右侧补 0，超过该长度的部分会被截断。
REFERENCE_LENGTH=1000

# 设为 true 时覆盖已经存在的 .npy；设为 false 时遇到已有文件会停止。
OVERWRITE=false

# Python 命令，可按环境改为具体解释器路径，例如 /path/to/env/bin/python。
PYTHON_BIN="python"
# ===========================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/token_jsonl_to_signal_npy.py"

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "错误：找不到 Python 脚本：${PYTHON_SCRIPT}" >&2
    exit 1
fi

if [[ ! -e "${INPUT_PATH}" ]]; then
    echo "错误：输入路径不存在：${INPUT_PATH}" >&2
    exit 1
fi

if [[ ! -e "${MODEL_PATH}" ]]; then
    echo "错误：模型路径不存在：${MODEL_PATH}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

ARGS=(
    --input "${INPUT_PATH}"
    --output-dir "${OUTPUT_DIR}"
    --model "${MODEL_PATH}"
    --device "${DEVICE}"
    --batch-size "${BATCH_SIZE}"
    --reference-length "${REFERENCE_LENGTH}"
)

if [[ -n "${SIGNAL_LENGTH}" ]]; then
    ARGS+=(--signal-length "${SIGNAL_LENGTH}")
fi

if [[ "${OVERWRITE}" == "true" ]]; then
    ARGS+=(--overwrite)
elif [[ "${OVERWRITE}" != "false" ]]; then
    echo "错误：OVERWRITE 只能设置为 true 或 false，当前值为：${OVERWRITE}" >&2
    exit 1
fi

echo "输入路径：${INPUT_PATH}"
echo "输出目录：${OUTPUT_DIR}"
echo "模型路径：${MODEL_PATH}"
echo "计算设备：${DEVICE}"
echo "批次大小：${BATCH_SIZE}"
echo "信号长度：${SIGNAL_LENGTH:-自动推断}"
echo "Reference 长度：${REFERENCE_LENGTH}"

"${PYTHON_BIN}" "${PYTHON_SCRIPT}" "${ARGS[@]}" "$@"
