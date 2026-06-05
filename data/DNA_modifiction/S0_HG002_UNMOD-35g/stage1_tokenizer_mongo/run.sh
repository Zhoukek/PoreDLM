#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# =============================================================================
# Pipeline switches
# =============================================================================

RUN_MERGE="true"
RUN_SPLIT="true"
RUN_MOVE_REFERENCES="true"
RUN_GENERATE_SHARDS="true"

# =============================================================================
# Paths
# =============================================================================

MERGE_PY="${script_dir}/merge_chunks_references.py"
SPLIT_PY="${script_dir}/split_chunks_references.py"
MOVE_REFERENCE_PY="${script_dir}/move_reference_files.py"
GENERATE_SHARDS_PY="${script_dir}/generate_shards_json.py"

ALL_DATA_DIR="${script_dir}/all_data"
TRAIN_DIR="${script_dir}/train"
TEST_DIR="${script_dir}/test"
VAL_DIR="${script_dir}/validation"

LOG_DIR="${script_dir}/logs"
mkdir -p "${LOG_DIR}"

# =============================================================================
# Merge settings
# =============================================================================

INPUT_ROOT_BASE="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/S0_HG002_UNMOD-35g/basecall_chunk"

MERGE_BATCH_IDS=(
    "250F601844011_0_0_0_0"
    "250F601844011_0_0_0_1"
    "250F601844011_0_0_0_2"
    "250F601844011_0_0_0_3"
    "250F601844011_0_0_1_0"
    "250F601844011_0_0_1_1"
    "250F601844011_0_0_1_2"
    "250F601844011_0_0_2_0"
)

MERGE_OVERWRITE="true"
MERGE_CHUNKS_FILENAME="chunks.npy"
MERGE_REFERENCES_FILENAME="references.npy"
REFERENCE_PAD_VALUE="0"

# =============================================================================
# Split settings
# =============================================================================

TRAIN_RATIO="0.8"
TEST_RATIO="0.1"
VAL_RATIO="0.1"
SEED="42"
BATCH_SIZE="4096"

SPLIT_OVERWRITE="false"
CHECK_CHUNK_LEN="true"
EXPECTED_CHUNK_LEN="6000"

# =============================================================================
# Reference move / shards settings
# =============================================================================

MOVE_OVERWRITE="false"
MOVE_DRY_RUN="false"

SHARDS_OVERWRITE="true"
SHARDS_OUTPUT_JSON="shards.json"
SHARDS_FILE_PATTERN="*_chunks.npy"
CHECK_EXPECTED_CHUNK_SIZE="true"
EXPECTED_CHUNK_SIZE="6000"


run_logged() {
    local step_name="$1"
    shift

    local log_file="${LOG_DIR}/${step_name}_$(date +%Y%m%d_%H%M%S).log"

    echo "================================================================================"
    echo "[RUN] ${step_name}"
    echo "[LOG] ${log_file}"
    echo "[CMD] $*"
    echo "================================================================================"

    "$@" 2>&1 | tee "${log_file}"

    echo "================================================================================"
    echo "[DONE] ${step_name}"
    echo "================================================================================"
}


check_file() {
    local path="$1"
    if [ ! -f "${path}" ]; then
        echo "[ERROR] File not found: ${path}"
        exit 1
    fi
}


check_dir() {
    local path="$1"
    if [ ! -d "${path}" ]; then
        echo "[ERROR] Directory not found: ${path}"
        exit 1
    fi
}


check_file "${MERGE_PY}"
check_file "${SPLIT_PY}"
check_file "${MOVE_REFERENCE_PY}"
check_file "${GENERATE_SHARDS_PY}"

if [ "${RUN_MERGE}" = "true" ]; then
    if [ "${#MERGE_BATCH_IDS[@]}" -eq 0 ]; then
        echo "[ERROR] MERGE_BATCH_IDS is empty"
        exit 1
    fi

    check_dir "${INPUT_ROOT_BASE}"

    for batch_id in "${MERGE_BATCH_IDS[@]}"; do
        input_root_dir="${INPUT_ROOT_BASE}/${batch_id}"
        output_prefix="${batch_id}"

        check_dir "${input_root_dir}"

        run_logged "step1_merge_${output_prefix}" \
            python "${MERGE_PY}" \
                --input_root_dir "${input_root_dir}" \
                --output_dir "${ALL_DATA_DIR}" \
                --output_prefix "${output_prefix}" \
                --overwrite "${MERGE_OVERWRITE}" \
                --chunks_filename "${MERGE_CHUNKS_FILENAME}" \
                --references_filename "${MERGE_REFERENCES_FILENAME}" \
                --reference_pad_value "${REFERENCE_PAD_VALUE}"
    done
else
    echo "[SKIP] step1 merge"
fi

if [ "${RUN_SPLIT}" = "true" ]; then
    run_logged "step2_split_train_test_validation" \
        python "${SPLIT_PY}" \
            --input_dir "${ALL_DATA_DIR}" \
            --train_dir "${TRAIN_DIR}" \
            --test_dir "${TEST_DIR}" \
            --validation_dir "${VAL_DIR}" \
            --train_ratio "${TRAIN_RATIO}" \
            --test_ratio "${TEST_RATIO}" \
            --val_ratio "${VAL_RATIO}" \
            --seed "${SEED}" \
            --batch_size "${BATCH_SIZE}" \
            --overwrite "${SPLIT_OVERWRITE}" \
            --check_chunk_len "${CHECK_CHUNK_LEN}" \
            --expected_chunk_len "${EXPECTED_CHUNK_LEN}"
else
    echo "[SKIP] split train/test/validation"
fi

if [ "${RUN_MOVE_REFERENCES}" = "true" ]; then
    for split_name in train validation test; do
        case "${split_name}" in
            train)
                split_dir="${TRAIN_DIR}"
                ;;
            validation)
                split_dir="${VAL_DIR}"
                ;;
            test)
                split_dir="${TEST_DIR}"
                ;;
        esac

        run_logged "step3_move_reference_${split_name}" \
            python "${MOVE_REFERENCE_PY}" \
                --source_dir "${split_dir}" \
                --target_dir "${split_dir}/reference" \
                --overwrite "${MOVE_OVERWRITE}" \
                --dry_run "${MOVE_DRY_RUN}"
    done
else
    echo "[SKIP] move reference files"
fi

if [ "${RUN_GENERATE_SHARDS}" = "true" ]; then
    for split_name in train validation test; do
        case "${split_name}" in
            train)
                split_dir="${TRAIN_DIR}"
                ;;
            validation)
                split_dir="${VAL_DIR}"
                ;;
            test)
                split_dir="${TEST_DIR}"
                ;;
        esac

        run_logged "step4_generate_shards_${split_name}" \
            python "${GENERATE_SHARDS_PY}" \
                --input_dir "${split_dir}" \
                --output_json "${SHARDS_OUTPUT_JSON}" \
                --file_pattern "${SHARDS_FILE_PATTERN}" \
                --overwrite "${SHARDS_OVERWRITE}" \
                --check_expected_chunk_size "${CHECK_EXPECTED_CHUNK_SIZE}" \
                --expected_chunk_size "${EXPECTED_CHUNK_SIZE}"
    done
else
    echo "[SKIP] generate shards.json"
fi

echo "================================================================================"
echo "[ALL DONE] Stage1 tokenizer pipeline finished"
echo "[Root] ${script_dir}"
echo "[Train] ${TRAIN_DIR}"
echo "[Validation] ${VAL_DIR}"
echo "[Test] ${TEST_DIR}"
echo "================================================================================"
