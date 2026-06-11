# 或指定进程数（例如 8）
# 或指定进程数（例如 8）

#!/bin/bash

# --- Configuration ---
# Change this variable to switch between different VQE model outputs (e.g., vqe25s57500, vqe43s67000, etc.)
# 我们用595G数据训练的vqe90s160000来对human_dna_032g进行
VQE_VERSION="vqe64k"

# Split parameters
# Split parameters
WORKERS=64
MIN_CHUNK_TOKEN_COUNT=300
CHUNK_WINDOW_SIZE=600
CHUNK_OVERLAP_SIZE=0


BASE_INPUT_DIR="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/S0_HG002_UNMOD-35g/stage2_BERT/01_S0_HG002_UNMOD_35g_model_type_1_cnn_type_0_distill_0.1_8k_vq"
BASE_OUTPUT_DIR="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/S0_HG002_UNMOD-35g/stage2_BERT/01_S0_HG002_UNMOD_35g_model_type_1_cnn_type_0_distill_0.1_8k_vq_split_600"


# --- End of Configuration ---

echo "Starting split process for VQE version: $VQE_VERSION"
echo "Base Input Directory: $BASE_INPUT_DIR"
echo "Base Output Directory: $BASE_OUTPUT_DIR"
echo "Workers: $WORKERS"
echo "Min Chunk Token Count: $MIN_CHUNK_TOKEN_COUNT"
echo "Window Size: $CHUNK_WINDOW_SIZE"
echo "Overlap Size: $CHUNK_OVERLAP_SIZE"
echo "----------------------------------------"

# Define subdirectories
SUBDIRS=("train" "validation")
# SUBDIRS=("validation")

# Loop through each subdirectory
for subdir in "${SUBDIRS[@]}"; do
    INPUT_DIR="${BASE_INPUT_DIR}/${subdir}"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${subdir}"

    # Create the output directory if it doesn't exist
    echo "📁 Ensuring output directory exists: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"

    # Run the python command
    echo "🚀 Executing split for '$subdir': $INPUT_DIR -> $OUTPUT_DIR"
    python /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/data/stage2_BERT_Encoder/step02_split_jsol.gz.py \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --workers "$WORKERS" \
        --min_chunk_token_count "$MIN_CHUNK_TOKEN_COUNT" \
        --chunk_window_size "$CHUNK_WINDOW_SIZE" \
        --chunk_overlap_size "$CHUNK_OVERLAP_SIZE"
    
    # Check the exit code of the python command
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "❌ Python command failed for '$subdir' with exit code: $exit_code" >&2
        exit $exit_code # Exit the script with the same error code
    fi

    echo "✅ Completed split for '$subdir'."
    echo "----------------------------------------"
done

echo "🎉 All splits completed successfully!"
