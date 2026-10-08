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

INPUT_ROOT_BASE="/mnt/zzbnew/rnamodel/data/DNA_data/ONT/ONT-R10-HG002-5K/PBA15131/test/data3/data3"

MERGE_BATCH_IDS=(
    "eval_00001"
    # "eval_00002"
    # "eval_00003"
    # "eval_00004"
    # "eval_00005"
    # "eval_00006"
    # "eval_00007"
    # "eval_00008"
    # "eval_00009"
    # "eval_00010"
    # "eval_00011"
    # "eval_00012"
    # "eval_00013"
    # "eval_00014"
    # "eval_00015"
    # "eval_00016"
    # "eval_00017"
    # "eval_00018"
    # "eval_00019"
    # "eval_00020"
    # "eval_00021"
    # "eval_00022"
    # "eval_00023"
    # "eval_00024"
    # "eval_00025"
    # "eval_00026"
    # "eval_00027"
    # "eval_00028"
    # "eval_00029"
    # "eval_00030"
    # "eval_00031"
    # "eval_00032"
    # "eval_00033"
    # "eval_00034"
    # "eval_00035"
    # "eval_00036"
    # "eval_00037"
    # "eval_00038"
    # "eval_00039"
    # "eval_00040"
    # "eval_00041"
    # "eval_00042"
    # "eval_00043"
    # "eval_00044"
    # "eval_00045"
    # "eval_00046"
    # "eval_00047"
    # "eval_00048"
    # "eval_00049"
    # "eval_00050"
    # "eval_00051"
    # "eval_00052"
    # "eval_00053"
    # "eval_00054"
    # "eval_00055"
    # "eval_00056"
    # "eval_00057"
    # "eval_00058"
    # "eval_00059"
    # "eval_00060"
    # "eval_00061"
    # "eval_00062"
    # "eval_00063"
    # "eval_00064"
    # "eval_00065"
    # "eval_00066"
    # "eval_00067"
    # "eval_00068"
    # "eval_00069"
    # "eval_00070"
    # "eval_00071"
    # "eval_00072"
    # "eval_00073"
    # "eval_00074"
    # "eval_00075"
    # "eval_00076"
    # "eval_00077"
    # "eval_00078"
    # "eval_00079"
    # "eval_00080"
    # "eval_00081"
    # "eval_00082"
    # "eval_00083"
    # "eval_00084"
    # "eval_00085"
    # "eval_00086"
    # "eval_00087"
    # "eval_00088"
    # "eval_00089"
    # "eval_00090"
    # "eval_00091"
    # "eval_00092"
    # "eval_00093"
    # "eval_00094"
    # "eval_00095"
    # "eval_00096"
    # "eval_00097"
    # "eval_00098"
    # "eval_00099"
    # "eval_00100"
    # "eval_00101"
    # "eval_00102"
    # "eval_00103"
    # "eval_00104"
    # "eval_00105"
    # "eval_00106"
    # "eval_00107"
    # "eval_00108"
    # "eval_00109"
    # "eval_00110"
    # "eval_00111"
    # "eval_00112"
    # "eval_00113"
    # "eval_00114"
    # "eval_00115"
    # "eval_00116"
    # "eval_00117"
    # "eval_00118"
    # "eval_00119"
    # "eval_00120"
    # "eval_00121"
    # "eval_00122"
    # "eval_00123"
    # "eval_00124"
    # "eval_00125"
    # "eval_00126"
    # "eval_00127"
    # "eval_00128"
    # "eval_00129"
    # "eval_00130"
    # "eval_00131"
    # "eval_00132"
    # "eval_00133"
    # "eval_00134"
    # "eval_00135"
    # "eval_00136"
    # "eval_00137"
    # "eval_00138"
    # "eval_00139"
    # "eval_00140"
    # "eval_00141"
    # "eval_00142"
    # "eval_00143"
    # "eval_00144"
    # "eval_00145"
    # "eval_00146"
    # "eval_00147"
    # "eval_00148"
    # "eval_00149"
    # "eval_00150"
    # "eval_00151"
    # "eval_00152"
    # "eval_00153"
    # "eval_00154"
    # "eval_00155"
    # "eval_00156"
    # "eval_00157"
    # "eval_00158"
    # "eval_00159"
    # "eval_00160"
    # "eval_00161"
    # "eval_00162"
    # "eval_00163"
    # "eval_00164"
    # "eval_00165"
    # "eval_00166"
    # "eval_00167"
    # "eval_00168"
    # "eval_00169"
    # "eval_00170"
    # "eval_00171"
    # "eval_00172"
    # "eval_00173"
    # "eval_00174"
    # "eval_00175"
    # "eval_00176"
    # "eval_00177"
    # "eval_00178"
    # "eval_00179"
    # "eval_00180"
    # "eval_00181"
    # "eval_00182"
    # "eval_00183"
    # "eval_00184"
    # "eval_00185"
    # "eval_00186"
    # "eval_00187"
    # "eval_00188"
    # "eval_00189"
    # "eval_00190"
    # "eval_00191"
    # "eval_00192"
    # "eval_00193"
    # "eval_00194"
    # "eval_00195"
    # "eval_00196"
    # "eval_00197"
    # "eval_00198"
    # "eval_00199"
    # "eval_00200"
    # "eval_00201"
    # "eval_00202"
    # "eval_00203"
    # "eval_00204"
    # "eval_00205"
    # "eval_00206"
    # "eval_00207"
    # "eval_00208"
    # "eval_00209"
    # "eval_00210"
    # "eval_00211"
    # "eval_00212"
    # "eval_00213"
    # "eval_00214"
    # "eval_00215"
    # "eval_00216"
    # "eval_00217"
    # "eval_00218"
    # "eval_00219"
    # "eval_00220"
    # "eval_00221"
    # "eval_00222"
    # "eval_00223"
    # "eval_00224"
    # "eval_00225"
    # "eval_00226"
    # "eval_00227"
    # "eval_00228"
    # "eval_00229"
    # "eval_00230"
    # "eval_00231"
    # "eval_00232"
    # "eval_00233"
    # "eval_00234"
    # "eval_00235"
    # "eval_00236"
    # "eval_00237"
    # "eval_00238"
    # "eval_00239"
    # "eval_00240"
    # "eval_00241"
    # "eval_00242"
    # "eval_00243"
    # "eval_00244"
)

MERGE_OVERWRITE="false"
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
