#!/usr/bin/env bash
set -Eeuo pipefail

# PoreDLM ODE + TCN + CTC-CRF training on the 800k-chunk corpus.
# Every commonly changed setting can be overridden from the environment, e.g.:
#   NUM_EPOCHS=1 STEPS_PER_EPOCH=20 BATCH_SIZE=2 BACKGROUND=0 ./run_ctc_crf_800k_4gpu.sh

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
conda_env="${CONDA_ENV:-poregpt}"
model_dir="${MODEL_DIR:-/mnt/zzbnew/poregpt/models/HF_VQE768C08A001_DNADLLM_V001/hf_dlm}"
data_root="${DATA_ROOT:-/mnt/zzbnew/poregpt/models/HF_VQE768C08A001_DNADLLM_V001/basecall/DNA_S1_HG00200_MIX_250F701901011_800000_chunks/basecall_data}"

# CTC-CRF settings. The reference script implicitly used state_len=5 and
# blank_score=2.0; make both explicit so the checkpoint is reproducible.
ctc_crf_state_len="${CTC_CRF_STATE_LEN:-5}"
ctc_crf_blank_score="${CTC_CRF_BLANK_SCORE:-2.0}"
ctc_crf_decode_blank_score="${CTC_CRF_DECODE_BLANK_SCORE:-${ctc_crf_blank_score}}"
ctc_decode_blank_bias="${CTC_DECODE_BLANK_BIAS:-0.0}"
ctc_beam_width="${CTC_BEAM_WIDTH:-16}"
ctc_beam_cut_threshold="${CTC_BEAM_CUT_THRESHOLD:-0.001}"
ctc_crf_move_loss_weight="${CTC_CRF_MOVE_LOSS_WEIGHT:-0}"
ctc_crf_move_target_offset="${CTC_CRF_MOVE_TARGET_OFFSET:-0}"
ctc_crf_move_smooth_l1_beta="${CTC_CRF_MOVE_SMOOTH_L1_BETA:-0.05}"
train_decoder="${TRAIN_DECODER:-ctc_crf}"
koi_blank_score="${KOI_BLANK_SCORE:-2.0}"
head_type="${HEAD_TYPE:-ctc_crf}"
head_output_activation="${HEAD_OUTPUT_ACTIVATION:-tanh}"
head_output_scale="${HEAD_OUTPUT_SCALE:-5}"
pretrained_strict="${PRETRAINED_STRICT:-1}"
context_unfreeze_last_n="${POREDLM_UNFREEZE_CONTEXT_LAST_N_LAYERS:-0}"
elf_unfreeze_last_n="${POREDLM_UNFREEZE_ELF_LAST_N_BLOCKS:-4}"
poredlm_memory_efficient_attention="${POREDLM_MEMORY_EFFICIENT_ATTENTION:-0}"
feature_source="${FEATURE_SOURCE:-hidden}"
hidden_layer="${HIDDEN_LAYER:--1}"
dlm_output="${DLM_OUTPUT:-ode}"
dlm_ode_steps="${DLM_ODE_STEPS:-2}"
dlm_ode_start_t="${DLM_ODE_START_T:-0.98}"
dlm_ode_self_cond_cfg_scale="${DLM_ODE_SELF_COND_CFG_SCALE:-0.0}"
pre_head_type="${PRE_HEAD_TYPE:-tcn}"
output_dir="${OUTPUT_DIR:-${project_root}/01.result/HF_VQE768C08A001_DNADLLM_V001/dlm_ode_tcn_ctc_crf_s${ctc_crf_state_len}_800k_4gpu}"

# Distributed runtime. GPU_IDS and NPROC_PER_NODE must describe the same number of GPUs.
gpu_ids="${GPU_IDS:-0,1,2,3}"
nproc_per_node="${NPROC_PER_NODE:-4}"
master_addr="${MASTER_ADDR:-127.0.0.1}"
master_port="${MASTER_PORT:-29512}"
ddp_backend="${DDP_BACKEND:-nccl}"

# The defaults mirror the referenced A100 run. For a smoke test, override
# NUM_EPOCHS/STEPS_PER_EPOCH/BATCH_SIZE as shown above.
batch_size="${BATCH_SIZE:-16}"
num_epochs="${NUM_EPOCHS:-500}"
# 12,250 optimizer steps is approximately one 98% pass over 800k reads with
# four GPUs and per-GPU batch size 16. Override when changing the global batch.
steps_per_epoch="${STEPS_PER_EPOCH:-12250}"
num_workers="${NUM_WORKERS:-8}"
eval_num_workers="${EVAL_NUM_WORKERS:-}"
head_lr="${HEAD_LR:-1e-5}"
backbone_lr="${BACKBONE_LR:-1e-5}"
pre_head_lr="${PRE_HEAD_LR:-}"
adapter_lr="${ADAPTER_LR:-}"
weight_decay="${WEIGHT_DECAY:-1e-5}"
warmup_ratio="${WARMUP_RATIO:-0.1}"
min_lr="${MIN_LR:-1e-6}"
seed="${SEED:-42}"
record_split_key_root="${RECORD_SPLIT_KEY_ROOT:-}"
train_reference_trim_bases="${TRAIN_REFERENCE_TRIM_BASES:-0}"
group_by="${GROUP_BY:-record}"
train_ratio="${TRAIN_RATIO:-0.98}"
val_ratio="${VAL_RATIO:-0.01}"
test_ratio="${TEST_RATIO:-0.01}"

# Evaluation/logging controls.
log_interval="${LOG_INTERVAL:-10}"
eval_interval="${EVAL_INTERVAL:-0}"
val_max_reads="${VAL_MAX_READS:-0}"
test_max_reads="${TEST_MAX_READS:-0}"
save_every="${SAVE_EVERY:-1}"
save_best="${SAVE_BEST:-1}"
use_amp="${USE_AMP:-1}"
use_wandb="${USE_WANDB:-0}"
wandb_project="${WANDB_PROJECT:-stage4_basecall_public}"
wandb_run_name="${WANDB_RUN_NAME:-dlm_ode_tcn_ctc_crf_s${ctc_crf_state_len}_800k_4gpu}"

# Execution controls. BACKGROUND=1 reproduces the nohup behavior of the source
# script. AUTO_RESUME only resumes a checkpoint created in this output folder.
background="${BACKGROUND:-1}"
foreground_tee="${FOREGROUND_TEE:-1}"
auto_resume="${AUTO_RESUME:-1}"
resume_ckpt="${RESUME_CKPT:-}"
pretrained_ckpt="${PRETRAINED_CKPT:-}"
train_adapter_only="${TRAIN_ADAPTER_ONLY:-0}"
use_poredlm_boundary_tokens="${USE_POREDLM_BOUNDARY_TOKENS:-0}"
dry_run="${DRY_RUN:-0}"

export CUDA_VISIBLE_DEVICES="${gpu_ids}"
export PYTHONPATH="${project_root}/script/dcbasecaller:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-basecall-ctc-crf}"
export HF_HOME="${HF_HOME:-/tmp/hf-basecall-dlm}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export POREDLM_BOUNDARY_MODE="${POREDLM_BOUNDARY_MODE:-bos_eos}"

if [[ ! -s "${model_dir}/config.json" || ! -s "${model_dir}/model.safetensors" ]]; then
  echo "[CTC-CRF][Error] incomplete PoreDLM model: ${model_dir}" >&2
  exit 2
fi

shopt -s nullglob
data_files=("${data_root}"/*.jsonl.gz)
shopt -u nullglob
if (( ${#data_files[@]} == 0 )); then
  echo "[CTC-CRF][Error] no .jsonl.gz files found in: ${data_root}" >&2
  exit 2
fi

IFS=',' read -r -a gpu_id_list <<<"${gpu_ids}"
if [[ "${#gpu_id_list[@]}" -ne "${nproc_per_node}" ]]; then
  echo "[CTC-CRF][Error] GPU_IDS has ${#gpu_id_list[@]} entries but NPROC_PER_NODE=${nproc_per_node}" >&2
  exit 2
fi

if [[ -n "${resume_ckpt}" ]]; then
  if [[ ! -s "${resume_ckpt}" ]]; then
    echo "[CTC-CRF][Error] RESUME_CKPT does not exist: ${resume_ckpt}" >&2
    exit 2
  fi
elif [[ "${auto_resume}" == "1" && -s "${output_dir}/ckpt_last.pt" ]]; then
  resume_ckpt="${output_dir}/ckpt_last.pt"
fi

if [[ -n "${pretrained_ckpt}" && ! -s "${pretrained_ckpt}" ]]; then
  echo "[CTC-CRF][Error] PRETRAINED_CKPT does not exist: ${pretrained_ckpt}" >&2
  exit 2
fi
if [[ -n "${resume_ckpt}" && -n "${pretrained_ckpt}" ]]; then
  echo "[CTC-CRF][Error] RESUME_CKPT/AUTO_RESUME and PRETRAINED_CKPT are mutually exclusive" >&2
  exit 2
fi

args=(
  --jsonl_paths "${data_root}"
  --group_by "${group_by}"
  --train_ratio "${train_ratio}"
  --val_ratio "${val_ratio}"
  --test_ratio "${test_ratio}"
  --split_seed "${seed}"
  --streaming
  --shuffle_buffer_size 20000
  --token_offset 0
  --train_reference_trim_bases "${train_reference_trim_bases}"
  --model_name_or_path "${model_dir}"
  --output_dir "${output_dir}"
  --batch_size "${batch_size}"
  --num_epochs "${num_epochs}"
  --steps_per_epoch "${steps_per_epoch}"
  --num_workers "${num_workers}"
  --lr "${head_lr}"
  --backbone_lr "${backbone_lr}"
  --weight_decay "${weight_decay}"
  --warmup_steps -1
  --warmup_ratio "${warmup_ratio}"
  --lr_scheduler warmup_stable_decay
  --lr_stable_ratio 0.8
  --min_lr "${min_lr}"
  --seed "${seed}"
  --feature_source "${feature_source}"
  --hidden-layer "${hidden_layer}"
  --dlm_output "${dlm_output}"
  --dlm_ode_steps "${dlm_ode_steps}"
  --dlm_ode_start_t "${dlm_ode_start_t}"
  --dlm_ode_self_cond_cfg_scale "${dlm_ode_self_cond_cfg_scale}"
  --freeze_backbone
  --poredlm_unfreeze_context_last_n_layers "${context_unfreeze_last_n}"
  --poredlm_unfreeze_elf_last_n_blocks "${elf_unfreeze_last_n}"
  --pre_head_type "${pre_head_type}"
  --head_type "${head_type}"
  --train_decoder "${train_decoder}"
  --ctc_crf_state_len "${ctc_crf_state_len}"
  --ctc_crf_blank_score "${ctc_crf_blank_score}"
  --ctc_crf_decode_blank_score "${ctc_crf_decode_blank_score}"
  --ctc_decode_blank_bias "${ctc_decode_blank_bias}"
  --ctc_beam_width "${ctc_beam_width}"
  --ctc_beam_cut_threshold "${ctc_beam_cut_threshold}"
  --ctc_crf_move_loss_weight "${ctc_crf_move_loss_weight}"
  --ctc_crf_move_target_offset "${ctc_crf_move_target_offset}"
  --ctc_crf_move_smooth_l1_beta "${ctc_crf_move_smooth_l1_beta}"
  --koi_blank_score "${koi_blank_score}"
  --head_output_activation "${head_output_activation}"
  --head_output_scale "${head_output_scale}"
  --ddp_backend "${ddp_backend}"
  --ddp_backend_fallback
  --clip_grad_norm 2.0
  --log_interval "${log_interval}"
  --eval_interval "${eval_interval}"
  --val_max_reads "${val_max_reads}"
  --test_max_reads "${test_max_reads}"
  --acc_min_coverage 0.5
  --save_every "${save_every}"
)

if [[ -n "${record_split_key_root}" ]]; then
  args+=(--record_split_key_root "${record_split_key_root}")
fi

if [[ "${poredlm_memory_efficient_attention}" == "1" ]]; then
  args+=(--poredlm_memory_efficient_attention)
fi

if [[ "${save_best}" == "1" ]]; then
  args+=(--save_best)
fi
if [[ -n "${pre_head_lr}" ]]; then
  args+=(--pre_head_lr "${pre_head_lr}")
fi
if [[ -n "${eval_num_workers}" ]]; then
  args+=(--eval_num_workers "${eval_num_workers}")
fi
if [[ -n "${adapter_lr}" ]]; then
  args+=(--adapter_lr "${adapter_lr}")
fi
if [[ "${train_adapter_only}" == "1" ]]; then
  args+=(--train_adapter_only)
fi

if [[ "${use_amp}" == "1" ]]; then
  args+=(--amp)
fi
if [[ -n "${resume_ckpt}" ]]; then
  args+=(--resume_ckpt "${resume_ckpt}")
fi
if [[ -n "${pretrained_ckpt}" ]]; then
  args+=(--pretrained_ckpt "${pretrained_ckpt}")
  if [[ "${pretrained_strict}" == "1" ]]; then
    args+=(--pretrained_strict)
  fi
fi
if [[ "${use_wandb}" == "1" ]]; then
  args+=(
    --use_wandb
    --wandb_project "${wandb_project}"
    --wandb_run_name "${wandb_run_name}"
  )
fi

if [[ "${use_poredlm_boundary_tokens}" == "1" ]]; then
  trainer_entrypoint=("${project_root}/script/dcbasecaller/scripts/train_poredlm_boundary.py")
else
  trainer_entrypoint=(-m basecall.train_ddp_multifolder)
fi

if command -v conda >/dev/null 2>&1; then
  command=(
    conda run --no-capture-output -n "${conda_env}"
    torchrun
  )
else
  torchrun_bin="${TORCHRUN_BIN:-}"
  if [[ -z "${torchrun_bin}" ]]; then
    torchrun_bin="$(command -v torchrun || true)"
  fi
  if [[ -n "${torchrun_bin}" ]]; then
    command=("${torchrun_bin}")
  else
    python_bin="${PYTHON_BIN:-python}"
    command=("${python_bin}" -m torch.distributed.run)
  fi
fi
command+=(
  --master_addr="${master_addr}"
  --master_port="${master_port}"
  --nnodes=1
  --nproc_per_node="${nproc_per_node}"
  "${trainer_entrypoint[@]}"
  "${args[@]}"
)

echo "[CTC-CRF] model=${model_dir}"
echo "[CTC-CRF] data=${data_root} files=${#data_files[@]} split=${group_by}:${train_ratio}/${val_ratio}/${test_ratio} split_key_root=${record_split_key_root:-physical} token_offset=0 train_reference_trim_bases=${train_reference_trim_bases}"
echo "[CTC-CRF] output=${output_dir}"
echo "[CTC-CRF] GPUs=${gpu_ids} world_size=${nproc_per_node} per_gpu_batch=${batch_size} global_batch=$((nproc_per_node * batch_size))"
echo "[CTC-CRF] epochs=${num_epochs} steps_per_epoch=${steps_per_epoch} head_lr=${head_lr} backbone_lr=${backbone_lr}"
echo "[CTC-CRF] representation=${feature_source}/${dlm_output} ode_steps=${dlm_ode_steps} ode_start_t=${dlm_ode_start_t} ode_cfg=${dlm_ode_self_cond_cfg_scale} pre_head=${pre_head_type} head=${head_type} context_unfreeze=${context_unfreeze_last_n} ELF_unfreeze=${elf_unfreeze_last_n} memory_efficient_attention=${poredlm_memory_efficient_attention} state_len=${ctc_crf_state_len} blank=${ctc_crf_blank_score} decode_blank=${ctc_crf_decode_blank_score} decoder=${train_decoder} move_loss=${ctc_crf_move_loss_weight}"
echo "[CTC-CRF] poredlm_boundary_tokens=${use_poredlm_boundary_tokens} boundary_mode=${POREDLM_BOUNDARY_MODE}"
if [[ -n "${resume_ckpt}" ]]; then
  echo "[CTC-CRF] resume=${resume_ckpt}"
elif [[ -n "${pretrained_ckpt}" ]]; then
  echo "[CTC-CRF] pretrained=${pretrained_ckpt} (weights only; fresh optimizer/scheduler)"
else
  echo "[CTC-CRF] resume=disabled (fresh CTC-CRF head)"
fi

if [[ "${dry_run}" == "1" ]]; then
  printf '[CTC-CRF] command:'
  printf ' %q' "${command[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${output_dir}" "${MPLCONFIGDIR}" "${HF_HOME}"

if [[ "${background}" == "1" ]]; then
  log_path="${output_dir}/nohup.out"
  nohup "${command[@]}" >"${log_path}" 2>&1 &
  launcher_pid=$!
  printf '%s\n' "${launcher_pid}" >"${output_dir}/launcher.pid"
  echo "[CTC-CRF] started pid=${launcher_pid} log=${log_path}"
else
  if [[ "${foreground_tee}" == "1" ]]; then
    "${command[@]}" 2>&1 | tee -a "${output_dir}/console.log"
  else
    "${command[@]}"
  fi
fi
