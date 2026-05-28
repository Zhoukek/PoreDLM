#!/bin/bash
# 先加载MACA环境
source /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/set_env.sh  # 你之前那个脚本

export PYTHONPATH=/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src:/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm:${PYTHONPATH:-}
# --- Configuration Section ---
# Modify these variables according to your setup

# Input directory containing .npy files
# INPUT_DIR="/mnt/si003067jezr/default/poregpt/dataset/human_dna_595g/memap_mongoq30/trank"
INPUT_DIR="/mnt/si003067jezr/default/poregpt/dataset/human_dna_032g/memap_lemonq0"
# Output directory for .jsonl.gz files
OUTPUT_DIR="/mnt/zzbnew/rnamodel/zhoukexuan/poregpt/dataset/human_dna_032g_baseline"


#!/bin/bash

# ==============================================================================
# ⚙️ 配置区域
# ==============================================================================


# 模型检查点路径
MODEL_CHECKPOINT="/mnt/zzbnew/rnamodel/shenhaojie/signalDNAmodel/test-haojieshen-model-type26-cnn_type13-baseline_model-lemon/encoder"

# Path to your trained VQ tokenizer model checkpoint (.pth file)

MODEL_TYPE=25

# Number of GPUs to use
NUM_GPUS=8 # <--- CHANGE THIS to the number of GPUs you want to use

# Batch size for tokenization (adjust based on your GPU memory)
BATCH_SIZE=8 # Adjust as needed

# Maximum number of concurrent tasks (recommended to match NUM_GPUS)
MAX_CONCURRENT=8 # Usually best to keep this equal to NUM_GPUS

# --- End of Configuration ---

# Validate NUM_GPUS

echo "Starting distributed tokenization process..."
echo "Input Directory: $INPUT_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Model Checkpoint: $MODEL_CHECKPOINT"
echo "Number of GPUs: $NUM_GPUS"
echo "Max Concurrent Tasks: $MAX_CONCURRENT"
echo "Batch Size: $BATCH_SIZE"
echo "----------------------------------------"

# Check if the input directory exists
if [[ ! -d "$INPUT_DIR" ]]; then # Fixed: Added space before closing bracket
  echo "Error: Input directory '$INPUT_DIR' does not exist!"
  exit 1
fi

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Get all .npy files recursively using mapfile and find with null separator
echo "🔍 Finding .npy files..."
mapfile -d '' all_files < <(find "$INPUT_DIR" -name "*.npy" -type f -print0) # Fixed: Added spaces around -name and -type

if [ ${#all_files[@]} -eq 0 ]; then
    echo "❌ No .npy files found in $INPUT_DIR." >&2
    exit 1
fi

total=${#all_files[@]} # Fixed: Removed space after =
echo "🔍 Found $total .npy files."

# Initialize counters
task_count=0 # Fixed: Removed space after =
processed=0  # Fixed: Removed space after =
skipped=0    # Fixed: Removed space after =

# Loop through each .npy file
for npy_file in "${all_files[@]}"; do
    # Skip if the file path is empty (shouldn't happen with mapfile -d '', but just in case)
    if [ -z "$npy_file" ]; then # Fixed: Removed space after =
        continue
    fi

    # Calculate relative path to maintain directory structure in output
    rel_path="${npy_file#$INPUT_DIR/}" # Fixed: Removed space after =, before $
    # Replace .npy extension with .jsonl.gz
    output_subpath="${rel_path%.npy}.jsonl.gz"
    # Construct full output path
    output_file="$OUTPUT_DIR/$output_subpath" # Fixed: Removed space after =
    # Get the target directory for the output file
    output_dir="$(dirname "$output_file")"

    # Construct lock file path (same location as output file, but with .lock extension)
    lock_file="$output_dir/${rel_path%.npy}.lock"

    # ✅ Check if output file already exists, skip if it does
    if [ -f "$output_file" ]; then
        # echo "Skipping $npy_file (output file already exists: $output_file)" # Uncomment if you want to see skipped files
        # ((skipped++))
        skipped=$((skipped + 1))
        continue
    fi

    # ✅ Check if lock file already exists, skip if it does (after checking output file)
    #if [ -f "$lock_file" ]; then
    #    # echo "Skipping $npy_file (lock file exists: $lock_file)" # Uncomment if you want to see skipped files due to locks
    #    ((skipped++))
    #    continue
    #fi

    # Ensure the output subdirectory exists
    mkdir -p "$output_dir"

    # Create the lock file *before* starting the task
    #touch "$lock_file"

    # Determine which GPU to use for this task
    gpu_id=$(( task_count % NUM_GPUS )) # Fixed: Removed space after =

    # Control concurrency: wait for any one task to finish if we've hit the limit
    if (( task_count >= MAX_CONCURRENT )); then
        # echo "Waiting for a task slot (max $MAX_CONCURRENT reached)..." # Optional debug info
        wait -n # Wait for the *next* background job to finish
    fi

    # Log the task submission
    echo "➡️  Submitting $(basename "$npy_file") to GPU $gpu_id (output: $output_file, lock: $lock_file)" >&2

    
    # Start the tokenization command in the background
    # Construct the device string and pass it via --device
    DEVICE_ARG="cuda:$gpu_id" # Fixed: Removed space after :
    echo "Executing: poredlm-vqe-tokenize-trank -i '$npy_file' -o '$output_file' --model-ckpt '$MODEL_CHECKPOINT' --device '$DEVICE_ARG' --batch-size $BATCH_SIZE"
    poregpt-vqe-tokenize-trank \
         -i "$npy_file" \
         -o "$output_file" \
	 --model-type $MODEL_TYPE \
         --model-ckpt "$MODEL_CHECKPOINT" \
         --device "$DEVICE_ARG" \
         --batch-size $BATCH_SIZE &

    # Increment task counter
    # ((task_count++))
    task_count=$((task_count + 1))
    # Increment processed counter (only for files that were actually submitted)
    # ((processed++))
    processed=$((processed + 1))
    

done

# Wait for *all* remaining background tasks to complete
echo "⏳ Waiting for all $processed tasks to finish..."
wait

# Print final summary
final_total=$((processed + skipped)) # Fixed: Removed space after =
echo "----------------------------------------"
echo "🎉 Tokenization finished!"
echo "  Total Files Found: $total"
echo "  Submitted for Processing: $processed"
echo "  Skipped (Already Existed or Locked): $skipped"
echo "  Final Count (P+S): $final_total"
if [ $final_total -eq $total ]; then
  echo "✅ Counts match. Process completed."
else
  echo "⚠️  Counts don't match total found. Unexpected state."
fi
