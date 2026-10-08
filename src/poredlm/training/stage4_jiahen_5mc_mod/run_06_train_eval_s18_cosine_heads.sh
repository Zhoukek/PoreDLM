#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Directory that already contains:
#   chr19_split.parquet
#   chr16_split.parquet
#   embeddings/chr19_v600_full.npy
#   embeddings/chr19_v610_full.npy
#   embeddings/chr19_v003_full.npy
#   embeddings/chr16_v600_full.npy
#   embeddings/chr16_v610_full.npy
#   embeddings/chr16_v003_full.npy
# source_out_dir="${1:-${script_dir}/output/v610_l2_200x}"
# source_out_dir="${1:-${script_dir}/output/v003_ode_l2_s2_t098}"
source_out_dir="${1:-${script_dir}/output/v007_apple_200x}"


# Parent directory for cosine-head results.
result_root="${2:-${source_out_dir}_cosine_heads}"

# The current 06 scripts require CUDA_VISIBLE_DEVICES=0 and --device cuda:0.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
device="${DEVICE:-cuda:0}"
epochs="${EPOCHS:-50}"
batch_size="${BATCH_SIZE:-1024}"
patience="${PATIENCE:-5}"
bootstrap="${BOOTSTRAP:-1000}"

# required_files=(
#   "chr19_split.parquet"
#   "chr16_split.parquet"
#   "embeddings/chr19_v003_ode_l2_s2_t098_full.npy"
#   "embeddings/chr16_v003_ode_l2_s2_t098_full.npy"
# )

required_files=(
  "chr19_split.parquet"
  "chr16_split.parquet"
  "embeddings/chr19_v007_full.npy"
  "embeddings/chr16_v007_full.npy"
)


for rel_path in "${required_files[@]}"; do
  if [[ ! -e "${source_out_dir}/${rel_path}" ]]; then
    echo "Missing required input: ${source_out_dir}/${rel_path}" >&2
    exit 1
  fi
done

prepare_out_dir() {
  local out_dir="$1"
  mkdir -p "${out_dir}"
  ln -sfn "${source_out_dir}/chr19_split.parquet" "${out_dir}/chr19_split.parquet"
  ln -sfn "${source_out_dir}/chr16_split.parquet" "${out_dir}/chr16_split.parquet"
  ln -sfn "${source_out_dir}/embeddings" "${out_dir}/embeddings"
  if [[ -e "${out_dir}/metrics.csv" ]]; then
    echo "Refusing to overwrite existing result: ${out_dir}/metrics.csv" >&2
    exit 1
  fi
}

run_head() {
  local name="$1"
  local py_file="$2"
  local out_dir="${result_root}/${name}"
  prepare_out_dir "${out_dir}"
  echo "Running ${name}"
  python "${script_dir}/${py_file}" \
    --out-dir "${out_dir}" \
    --device "${device}" \
    --epochs "${epochs}" \
    --batch-size "${batch_size}" \
    --patience "${patience}" \
    --bootstrap "${bootstrap}"
}

mkdir -p "${result_root}"

# run_head "cosine_residual1536" "06_train_eval_s18_cosine_residual1536.py"
# run_head "cosine_residual" "06_train_eval_s18_cosine_residual.py"
# run_head "cosine_raw" "06_train_eval_s18_cosine_raw.py"
# run_head "mlp_plus_cosine_residual" "06_train_eval_s18_mlp_plus_cosine_residual.py"
# run_head "mlp_plus_cosine_aux_residual" "06_train_eval_s18_mlp_plus_cosine_aux_residual.py" \
# run_head "mlp_plus_cosine_proto_residual" "06_train_eval_s18_mlp_plus_cosine_proto_residual.py" \

# run_head "metric_proto" "06_train_eval_s18_metric_proto.py"
# run_head "rbf_kernel" "06_train_eval_s18_rbf_kernel.py" 

# run_head "multi_proto" "06_train_eval_s18_multi_proto.py" \

# run_head "mlp_plus_cosine_whitened_residual" "06_train_eval_s18_mlp_plus_cosine_whitened_residual.py" \

# run_head "mlp_residual" "06_train_eval_s18.py" \
# run_head "l2_distance_proto" "06_train_eval_s18_l2_distance_proto.py"

run_head "mlp_plus_cosine_whitened_embedding" "06_train_eval_s18_mlp_plus_cosine_whitened_embedding.py" \

echo "All cosine-head runs finished under: ${result_root}"
