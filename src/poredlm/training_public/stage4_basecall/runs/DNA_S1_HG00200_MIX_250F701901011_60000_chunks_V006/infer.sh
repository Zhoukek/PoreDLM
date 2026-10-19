#!/bin/bash
set -euo pipefail

# Usage:
#   bash infer.sh /path/to/input.jsonl.gz /path/to/output.fastq
#
# The input JSONL(.gz) should contain reads with a "text" field made of
# <|bwav:ID|> tokens and a "read_id" or "id" field.

project_root="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM"
source "${project_root}/src/poredlm/training/set_env.sh"

stage3_root="${project_root}/src/poredlm/training_public/stage3_DLM_train"
stage4_root="${project_root}/src/poredlm/training_public/stage4_basecall"

export PYTHONPATH="${stage4_root}:${project_root}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
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

jsonl_gz="/mnt/zzbnew/poregpt/models/HF_VQE768C08A001_DNADLLM_V006/basecall/test/test_for_dlm/eval_00001_chunks.jsonl.gz"
out_fastq="${2:-${run_dir}/infer.fastq}"

if [[ -z "${jsonl_gz}" ]]; then
  echo "Usage: bash $0 /path/to/input.jsonl.gz [/path/to/output.fastq]"
  exit 1
fi
if [[ ! -f "${jsonl_gz}" ]]; then
  echo "ERROR: input JSONL file not found: ${jsonl_gz}"
  exit 1
fi

mkdir -p "$(dirname "${out_fastq}")"

python -m Basecalling.basecaller_v8_0420.eval \
  --ckpt "${ckpt}" \
  --model_name_or_path "${base_model}" \
  --jsonl_gz "${jsonl_gz}" \
  --out "${out_fastq}" \
  --device cuda \
  --amp \
  --decoder auto \
  --pre_head_type auto \
  --feature_source ode_hidden \
  --hidden_layer -1 \
  --elf_ode_steps 2 \
  --elf_ode_start_t 0.98 \
  --elf_self_cond_cfg_scale 0.5 \
  --ctc_crf_blank_score 2.0 \
  --token_offset "${TOKEN_OFFSET:-0}" \
  --batch_size "${BATCH_SIZE:-4}" \
  --max_tokens "${MAX_TOKENS:-2048}" \
  --overlap "${OVERLAP:-128}"

echo "Inference done. FASTQ: ${out_fastq}"
