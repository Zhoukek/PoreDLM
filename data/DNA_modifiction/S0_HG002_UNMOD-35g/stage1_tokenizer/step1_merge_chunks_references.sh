#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# 只需要修改这里
# =============================================================================

PY_SCRIPT="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/S0_HG002_UNMOD-35g/stage1_tokenizer/merge_chunks_references.py"

INPUT_ROOT_DIR="/mnt/zzbnew/rnamodel/wangxue/data/DNA_data/S0_HG002_UNMOD/250F601844011/basecall_chunk/250F601844011_0_2_1_2"

OUTPUT_DIR="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/S0_HG002_UNMOD-35g/stage1_tokenizer/all_data"

OUTPUT_PREFIX="250F601844011_0_2_1_2"

OVERWRITE="false"

# =============================================================================
# 日志
# =============================================================================

LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/merge_${OUTPUT_PREFIX}_$(date +%Y%m%d_%H%M%S).log"

echo "================================================================================"
echo "Run merge_chunks_references.py"
echo "================================================================================"
echo "[Python script] ${PY_SCRIPT}"
echo "[Input root]    ${INPUT_ROOT_DIR}"
echo "[Output dir]    ${OUTPUT_DIR}"
echo "[Output prefix] ${OUTPUT_PREFIX}"
echo "[Overwrite]     ${OVERWRITE}"
echo "[Log file]      ${LOG_FILE}"
echo "================================================================================"

if [ ! -f "${PY_SCRIPT}" ]; then
    echo "[ERROR] Python script not found: ${PY_SCRIPT}"
    exit 1
fi

python "${PY_SCRIPT}" \
    --input_root_dir "${INPUT_ROOT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --output_prefix "${OUTPUT_PREFIX}" \
    --overwrite "${OVERWRITE}" \
    2>&1 | tee "${LOG_FILE}"

echo "================================================================================"
echo "[Done] Merge finished"
echo "[Log file] ${LOG_FILE}"
echo "================================================================================"