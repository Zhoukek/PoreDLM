#!/bin/bash
set -euo pipefail

# Usage:
#   bash eval.sh [/path/to/eval.jsonl.gz_or_data_dir] [/path/to/eval_out]
#
# INPUT_TYPE=jsonl (default) uses --jsonl_paths.
# INPUT_TYPE=npy uses --npy_paths for tokens_*.npy/reference_*.npy pairs.

project_root="${PROJECT_ROOT:-/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM}"
source "${project_root}/src/poredlm/training/set_env.sh"

stage4_root="${project_root}/src/poredlm/training_public/stage4_basecall"

export PYTHONPATH="${stage4_root}:${project_root}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export TORCHDYNAMO_DISABLE=1

# Keep these model settings aligned with run.sh for the checkpoint being evaluated.
base_model="${BASE_MODEL:-/mnt/zzbnew/poregpt/models/HF_VQE768C08A001_DNADLLM_V006/hf_dlm}"
run_dir="${stage4_root}/runs/DNA_S1_HG00200_MIX_250F701901011_60000_chunks_V006_base"

pre_head_type="tcn"
feature_source="ode_hidden"
head_output_activation="tanh"
head_output_scale="5"
backbone_chunk_size="1540"

# codebook/token-id feature fusion 参数：none / only / add / concat / gate
codebook_fusion="none"
codebook_fusion_dropout="${CODEBOOK_FUSION_DROPOUT:-0.1}"
codebook_fusion_gate_bias="${CODEBOOK_FUSION_GATE_BIAS:--2.0}"
skip_backbone_for_codebook_only="${SKIP_BACKBONE_FOR_CODEBOOK_ONLY:-false}"
codebook_feature_path="/mnt/zzbnew/poregpt/models/HF_VQE768C08A001_DNADLLM_V006/encoder"
codebook_feature_token_offset="${CODEBOOK_FEATURE_TOKEN_OFFSET:-128}"

# ODE 参数（feature_source="ode_hidden" 且未跳过 backbone 时生效）
elf_ode_steps="1"
elf_ode_start_t="0.97"
elf_ode_self_cond_cfg_scale="0.5"

ckpt="${CKPT:-${run_dir}/ckpt_best.pt}"
if [[ ! -f "${ckpt}" ]]; then
  ckpt="${run_dir}/ckpt_last.pt"
fi
if [[ ! -f "${ckpt}" ]]; then
  echo "ERROR: checkpoint not found. Expected ${run_dir}/ckpt_best.pt or ckpt_last.pt."
  echo "       You can also set CKPT=/path/to/checkpoint.pt before running this script."
  exit 1
fi

input_path="/mnt/zzbnew/poregpt/models/HF_VQE768C08A001_DNADLLM_V006/basecall_ONT/test/test_for_dlm"
# input_path="/mnt/zzbnew/poregpt/models/HF_VQE768C08A001_DNADLLM_V006/basecall/test/test_for_dlm"
out_dir="${2:-${run_dir}/eval_out_${codebook_fusion}_ont_v1}"
input_type="${INPUT_TYPE:-jsonl}"
ctc_decode_beamsize="${CTC_DECODE_BEAMSIZE:-1}"
ctc_blank_logit_bias="-0.0"
ctc_nonblank_logit_bias="0.0"
ctc_logit_temperature="${CTC_LOGIT_TEMPERATURE:-1.0}"

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
if [[ "${skip_backbone_for_codebook_only}" == "true" && "${codebook_fusion}" != "only" ]]; then
  echo "ERROR: SKIP_BACKBONE_FOR_CODEBOOK_ONLY=true requires CODEBOOK_FUSION=only."
  exit 1
fi

mkdir -p "${out_dir}"

input_args=()
if [[ "${input_type}" == "npy" ]]; then
  input_args=(--npy_paths "${input_path}")
else
  input_args=(--jsonl_paths "${input_path}")
fi

codebook_args=(
  --codebook_fusion "${codebook_fusion}"
  --codebook_fusion_dropout "${codebook_fusion_dropout}"
  --codebook_fusion_gate_bias "${codebook_fusion_gate_bias}"
)
if [[ -n "${codebook_feature_path}" ]]; then
  codebook_args+=(
    --codebook_feature_path "${codebook_feature_path}"
    --codebook_feature_token_offset "${codebook_feature_token_offset}"
  )
fi
if [[ "${skip_backbone_for_codebook_only}" == "true" ]]; then
  codebook_args+=(--skip_backbone_for_codebook_only)
fi

python -m Basecalling.basecaller_v8_0420.eval \
  "${input_args[@]}" \
  --recursive \
  --ckpt "${ckpt}" \
  --model_name_or_path "${base_model}" \
  --out_dir "${out_dir}" \
  --fastq_out "${out_dir}/eval.fastq" \
  --decoder ctc_viterbi \
  --ctc_decode_beamsize "${ctc_decode_beamsize}" \
  --ctc_blank_logit_bias "${ctc_blank_logit_bias}" \
  --ctc_nonblank_logit_bias "${ctc_nonblank_logit_bias}" \
  --ctc_logit_temperature "${ctc_logit_temperature}" \
  --pre_head_type tcn \
  --feature_source ode_hidden \
  --hidden_layer -1 \
  --head_output_activation tanh \
  --head_output_scale 5 \
  --backbone_chunk_size 1540 \
  --elf_ode_steps 1 \
  --elf_ode_start_t 0.97 \
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

