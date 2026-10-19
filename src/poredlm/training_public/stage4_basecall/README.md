# Public Stage 4 basecalling

This directory is a public copy of the Stage 4 basecaller, adapted to load the
Stage 3 Hugging Face DLM model from `stage3_DLM_train/runs/test/hf_dlm`.

## Run

The default launcher is `runs/test/run.sh`. Paths and common launch settings can
be overridden without editing the script:

```bash
DATA_ROOT=/path/to/jsonl_dir \
MODEL_DIR=/path/to/hf_dlm \
OUTPUT_DIR=/path/to/output \
CUDA_VISIBLE_DEVICES=0,1 \
bash runs/test/run.sh
```

The public Stage 2/3 vocabulary uses raw codebook id + 128, with BOS/EOS ids
2/3. Therefore the launcher selects the built-in `bwav` tokenizer. If an HF
tokenizer is packaged with the model instead, pass
`--tokenizer_type auto --tokenizer_name_or_path /path/to/tokenizer` after the
script name.

The default feature is `denoised_hidden` (`last_hidden_state` from the DLM). For
experiments, append `--feature_source context_hidden` or
`--feature_source ode_hidden --elf_ode_steps 2 --elf_ode_start_t 0.98`.
SDE features are selected with `--feature_source sde_hidden`; the corresponding
step, start-time, gamma, CFG-scale, and seed settings are exposed in `run.sh`.

The HF model directory is expected to contain its own
`ELF-pytorch-port/src` directory. The launcher does not add a separate Stage 3
ELF source tree to `PYTHONPATH`.

W&B is disabled by default. Enable it by appending `--use_wandb` and the desired
W&B arguments; credentials should be provided by the environment rather than
stored in this repository.

On a MACA node, the launcher automatically sources `src/poredlm/training/set_env.sh`.
Set `SOURCE_TRAINING_ENV=1` to force this on a node where `/opt/maca` is mounted
later by the runtime.
