#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Directory that already contains:
#   chr16_split.parquet / chr19_split.parquet
#   embeddings/chr16_v600_full.npy, chr16_v610_full.npy, chr16_v003_full.npy
#   embeddings/chr19_v600_full.npy, chr19_v610_full.npy, chr19_v003_full.npy
out_dir="${1:-${OUT_DIR:-${script_dir}/output/v610_l2_200x}}"

chrom="${CHROM:-chr16}"
compare_role="${COMPARE_ROLE:-test}"
reference_role="${REFERENCE_ROLE:-baseline}"
output_prefix="${OUTPUT_PREFIX:-embedding_distance}"
min_reference_per_7mer="${MIN_REFERENCE_PER_7MER:-3}"
min_class_per_7mer="${MIN_CLASS_PER_7MER:-3}"
max_strip_points="${MAX_STRIP_POINTS:-4000}"

# Space-separated model names, for example:
#   MODELS="V003_Stone"
#   MODELS="V610_Apple V003_Stone"
models="${MODELS:-V610_Apple}"

args=(
  "--out-dir" "${out_dir}"
  "--chrom" "${chrom}"
  "--compare-role" "${compare_role}"
  "--reference-role" "${reference_role}"
  "--output-prefix" "${output_prefix}"
  "--min-reference-per-7mer" "${min_reference_per_7mer}"
  "--min-class-per-7mer" "${min_class_per_7mer}"
  "--max-strip-points" "${max_strip_points}"
  "--models"
)

read -r -a model_array <<< "${models}"
args+=("${model_array[@]}")

if [[ "${L2_NORMALIZE_EMBEDDING:-0}" == "1" ]]; then
  args+=("--l2-normalize-embedding")
fi

python "${script_dir}/08_plot_embedding_distances_s18.py" "${args[@]}"
