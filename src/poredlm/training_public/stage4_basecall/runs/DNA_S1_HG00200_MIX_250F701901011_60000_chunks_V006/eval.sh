#!/bin/bash
set -euo pipefail

# Usage:
#   bash eval.sh /path/to/basecall_data [/path/to/eval_out]
#
# INPUT_TYPE=jsonl (default) uses --jsonl_paths.
# INPUT_TYPE=npy uses --npy_paths for tokens_*.npy/reference_*.npy pairs.

project_root="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM"
source "${project_root}/src/poredlm/training/set_env.sh"

stage3_root="${project_root}/src/poredlm/training_public/stage3_DLM_train"
stage4_root="${project_root}/src/poredlm/training_public/stage4_basecall"

export PYTHONPATH="${stage4_root}:${project_root}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export TORCHDYNAMO_DISABLE=1

base_model="/mnt/zzbnew/poregpt/models/HF_VQE768C08A001_DNADLLM_V006/hf_dlm"
run_dir="${stage4_root}/runs/DNA_S1_HG00200_MIX_250F701901011_60000_chunks_V006"

ckpt="${CKPT:-${run_dir}/ckpt_best.pt}"
if [[ ! -f "${ckpt}" ]]; then
  ckpt="${run_dir}/ckpt_last.pt"
fi
if [[ ! -f "${ckpt}" ]]; then
  echo "ERROR: checkpoint not found. Expected ${run_dir}/ckpt_best.pt or ckpt_last.pt."
  echo "       You can also set CKPT=/path/to/checkpoint.pt before running this script."
  exit 1
fi

input_path="/mnt/zzbnew/poregpt/models/HF_VQE768C08A001_DNADLLM_V006/basecall/test/test_for_dlm/eval_00001_chunks.jsonl.gz"
out_dir="${2:-${run_dir}/eval_out_cyclone_s1_test}"
# out_dir="${2:-${run_dir}/eval_outont_r10}"

input_type="${INPUT_TYPE:-jsonl}"

if [[ -z "${input_path}" ]]; then
  echo "Usage: bash $0 /path/to/basecall_data [/path/to/eval_out]"
  echo "       Set INPUT_TYPE=npy for tokens/reference npy inputs."
  exit 1
fi
if [[ ! -e "${input_path}" ]]; then
  echo "ERROR: input path not found: ${input_path}"
  exit 1
fi
if [[ "${input_type}" != "jsonl" && "${input_type}" != "npy" ]]; then
  echo "ERROR: INPUT_TYPE must be 'jsonl' or 'npy', got: ${input_type}"
  exit 1
fi

mkdir -p "${out_dir}"

input_args=()
if [[ "${input_type}" == "npy" ]]; then
  input_args=(--npy_paths "${input_path}")
else
  input_args=(--jsonl_paths "${input_path}")
fi

python -m Basecalling.basecaller_v8_0420.eval \
  "${input_args[@]}" \
  --recursive \
  --ckpt "${ckpt}" \
  --model_name_or_path "${base_model}" \
  --out_dir "${out_dir}" \
  --fastq_out "${out_dir}/eval.fastq" \
  --decoder ctc_viterbi \
  --pre_head_type tcn \
  --feature_source ode_hidden \
  --hidden_layer -1 \
  --elf_ode_steps 2 \
  --elf_ode_start_t 0.98 \
  --elf_self_cond_cfg_scale 0.5 \
  --ctc_crf_blank_score 2.0 \
  --token_offset "${TOKEN_OFFSET:-0}" \
  --batch_size "${BATCH_SIZE:-64}" \
  --num_workers "${NUM_WORKERS:-0}" \
  --num_visualize "${NUM_VISUALIZE:-100}" \
  --max_len "${MAX_LEN:-200}"

echo "Evaluation done."
echo "Metrics: ${out_dir}/metrics.json"
echo "FASTQ:   ${out_dir}/eval.fastq"
