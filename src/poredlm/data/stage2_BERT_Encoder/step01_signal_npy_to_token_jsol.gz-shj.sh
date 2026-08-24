#!/bin/bash

set -u

# ============================================================
# 环境配置
# ============================================================

export PYTHONPATH="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src:/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm:${PYTHONPATH:-}"

# 减少 PyTorch CUDA 显存碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


# ============================================================
# 数据路径
# ============================================================

INPUT_DIR="/mnt/zzbnew/rnamodel/shenhaojie/data/ONT-R9-data"

OUTPUT_DIR="/mnt/zzbnew/rnamodel/shenhaojie/data/ONT-R9-data"


# ============================================================
# Tokenizer 模型
# ============================================================

MODEL_CHECKPOINT="/mnt/zzbnew/poregpt/models/HF_VQE768C08A001_DNADLLM_V001/encoder"

MODEL_TYPE=1


# ============================================================
# GPU 设置
# ============================================================

# 固定使用物理 GPU 1
TARGET_GPU=1

# 每次 tokenizer inference 的 batch size
BATCH_SIZE=8

# 因为固定使用一张 GPU，只同时运行一个任务
MAX_CONCURRENT=1


# ============================================================
# Python tokenizer 脚本
# ============================================================

TOKENIZER_SCRIPT="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/data/stage2_BERT_Encoder/step01_signal_npy_to_token_jsol.gz.py"


# ============================================================
# 启动信息
# ============================================================

echo "============================================================"
echo "Starting signal tokenization"
echo "============================================================"
echo "Input Directory : $INPUT_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Model Checkpoint: $MODEL_CHECKPOINT"
echo "Target GPU      : cuda:$TARGET_GPU"
echo "Batch Size      : $BATCH_SIZE"
echo "Max Concurrent  : $MAX_CONCURRENT"
echo "============================================================"


# ============================================================
# 基础检查
# ============================================================

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "❌ ERROR: Input directory does not exist:"
    echo "$INPUT_DIR"
    exit 1
fi

if [[ ! -d "$MODEL_CHECKPOINT" ]]; then
    echo "❌ ERROR: Model checkpoint does not exist:"
    echo "$MODEL_CHECKPOINT"
    exit 1
fi

if [[ ! -f "$TOKENIZER_SCRIPT" ]]; then
    echo "❌ ERROR: Tokenizer script does not exist:"
    echo "$TOKENIZER_SCRIPT"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"


# ============================================================
# 检查 GPU
# ============================================================

echo ""
echo "🔍 Checking GPU..."
nvidia-smi -i "$TARGET_GPU" || {
    echo "❌ ERROR: GPU $TARGET_GPU is unavailable."
    exit 1
}

echo ""


# ============================================================
# 查找所有 *chunks.npy
# ============================================================

echo "🔍 Finding *chunks.npy files..."

mapfile -d '' all_files < <(
    find "$INPUT_DIR" \
        -type f \
        -name "*chunks.npy" \
        -print0
)

if [[ ${#all_files[@]} -eq 0 ]]; then
    echo "❌ No *chunks.npy files found in:"
    echo "$INPUT_DIR"
    exit 1
fi

total=${#all_files[@]}

echo "✅ Found $total .npy file(s)."
echo ""


# ============================================================
# 统计变量
# ============================================================

processed=0
skipped=0
failed=0
success=0

declare -a PIDS
declare -a NAMES


# ============================================================
# 处理每个 chunks.npy
# ============================================================

for npy_file in "${all_files[@]}"; do

    [[ -z "$npy_file" ]] && continue

    # --------------------------------------------------------
    # 计算相对路径
    # --------------------------------------------------------

    rel_path="${npy_file#$INPUT_DIR/}"

    output_subpath="${rel_path%.npy}.jsonl.gz"

    output_file="$OUTPUT_DIR/$output_subpath"

    output_dir="$(dirname "$output_file")"

    mkdir -p "$output_dir"


    # --------------------------------------------------------
    # lock / log
    # --------------------------------------------------------

    base_name="${rel_path%.npy}"

    lock_file="$output_dir/${base_name}.lock"

    log_file="$output_dir/${base_name}.tokenize.log"


    # --------------------------------------------------------
    # 如果已经有输出，直接跳过
    # --------------------------------------------------------

    if [[ -s "$output_file" ]]; then

        echo "⏭️  SKIP:"
        echo "    $npy_file"
        echo "    output already exists:"
        echo "    $output_file"

        skipped=$((skipped + 1))
        continue
    fi


    # --------------------------------------------------------
    # 如果存在旧的空/损坏输出，删除
    # --------------------------------------------------------

    if [[ -f "$output_file" && ! -s "$output_file" ]]; then
        echo "⚠️ Removing empty output:"
        echo "   $output_file"

        rm -f "$output_file"
    fi


    # --------------------------------------------------------
    # 创建 lock
    # --------------------------------------------------------

    touch "$lock_file"


    echo ""
    echo "============================================================"
    echo "➡️ Starting tokenization"
    echo "============================================================"
    echo "Input : $npy_file"
    echo "Output: $output_file"
    echo "GPU   : cuda:$TARGET_GPU"
    echo "Batch : $BATCH_SIZE"
    echo "Log   : $log_file"
    echo "============================================================"


    # --------------------------------------------------------
    # 启动 tokenizer
    # --------------------------------------------------------

    python "$TOKENIZER_SCRIPT" \
        -i "$npy_file" \
        -o "$output_file" \
        --model-type "$MODEL_TYPE" \
        --model-ckpt "$MODEL_CHECKPOINT" \
        --device "cuda:$TARGET_GPU" \
        --batch-size "$BATCH_SIZE" \
        > "$log_file" 2>&1 &

    pid=$!

    PIDS+=("$pid")
    NAMES+=("$npy_file")

    processed=$((processed + 1))


    echo "PID = $pid"


    # --------------------------------------------------------
    # 固定一张卡，因此达到并发限制时等待
    # --------------------------------------------------------

    while (( ${#PIDS[@]} >= MAX_CONCURRENT )); do

        pid_to_wait="${PIDS[0]}"
        file_to_wait="${NAMES[0]}"

        echo ""
        echo "⏳ Waiting for PID $pid_to_wait ..."
        echo "   $file_to_wait"

        if wait "$pid_to_wait"; then

            # 检查输出是否真实存在且非空
            rel_wait="${file_to_wait#$INPUT_DIR/}"
            output_wait="$OUTPUT_DIR/${rel_wait%.npy}.jsonl.gz"
            lock_wait="$(dirname "$output_wait")/${rel_wait%.npy}.lock"

            if [[ -s "$output_wait" ]]; then
                echo "✅ SUCCESS:"
                echo "   $file_to_wait"
                echo "   -> $output_wait"

                success=$((success + 1))
            else
                echo "❌ FAILED: process exited but output is missing/empty"
                echo "   $file_to_wait"

                failed=$((failed + 1))
            fi

            rm -f "$lock_wait"

        else

            rel_wait="${file_to_wait#$INPUT_DIR/}"
            output_wait="$OUTPUT_DIR/${rel_wait%.npy}.jsonl.gz"
            lock_wait="$(dirname "$output_wait")/${rel_wait%.npy}.lock"

            echo "❌ FAILED:"
            echo "   $file_to_wait"

            failed=$((failed + 1))

            rm -f "$lock_wait"
            rm -f "$output_wait"
        fi


        # 删除已经 wait 的第一个任务
        PIDS=("${PIDS[@]:1}")
        NAMES=("${NAMES[@]:1}")

    done

done


# ============================================================
# 等待剩余任务
# ============================================================

for idx in "${!PIDS[@]}"; do

    pid="${PIDS[$idx]}"
    npy_file="${NAMES[$idx]}"

    echo ""
    echo "⏳ Waiting for remaining PID $pid ..."

    rel_path="${npy_file#$INPUT_DIR/}"

    output_file="$OUTPUT_DIR/${rel_path%.npy}.jsonl.gz"

    lock_file="$(dirname "$output_file")/${rel_path%.npy}.lock"

    if wait "$pid"; then

        if [[ -s "$output_file" ]]; then

            echo "✅ SUCCESS:"
            echo "   $npy_file"
            echo "   -> $output_file"

            success=$((success + 1))

        else

            echo "❌ FAILED: output missing or empty"
            echo "   $npy_file"

            failed=$((failed + 1))
        fi

    else

        echo "❌ FAILED:"
        echo "   $npy_file"

        failed=$((failed + 1))

        rm -f "$output_file"
    fi

    rm -f "$lock_file"

done


# ============================================================
# 最终统计
# ============================================================

echo ""
echo "============================================================"
echo "Tokenization Summary"
echo "============================================================"
echo "Total files found : $total"
echo "Submitted         : $processed"
echo "Success           : $success"
echo "Failed            : $failed"
echo "Skipped           : $skipped"
echo "============================================================"

if (( failed == 0 )); then

    echo "🎉 All tokenization tasks completed successfully."

else

    echo "⚠️ Some tokenization tasks failed."
    echo "Please inspect *.tokenize.log files."

fi