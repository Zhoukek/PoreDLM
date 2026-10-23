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

`--backbone_type gpt` supports decoder-only Hugging Face models that expose
standard `last_hidden_state`/`hidden_states` outputs (including compatible
`AutoModel` implementations). Use `--feature_source hidden`; DLM-specific
context/ELF features are not available for GPT backbones. The `auto` setting
recognizes common causal-LM architectures from their HF config, including
Olmo/Olmo2. For `bwav` input, all emitted token IDs must fit the model's
embedding vocabulary; this is checked by the tokenizer at data load time.

The default feature is `denoised_hidden` (`last_hidden_state` from the DLM). For
experiments, append `--feature_source context_hidden` or
`--feature_source ode_hidden --elf_ode_steps 2 --elf_ode_start_t 0.98`.
SDE features can be selected for inference or evaluation with
`--feature_source sde_hidden --sde_steps 4 --sde_start_t 0.85 --sde_gamma 0.1`.
Use `--sde_seed N` to make the SDE noise repeatable. The ODE-trained
basecalling head can be evaluated with these SDE features, but this changes its
input feature distribution; compare against the ODE baseline on validation data.
For `--feature_source context_hidden`, select the context encoder layer with
`--hidden-layer`: `-1` is the final layer and `-2` is the penultimate layer.
The same selection is supported for `--feature_source ode_hidden`; in that
case ODE refinement starts from the selected context layer. Non-final layer
selection runs directly through the context encoder, so it is not compatible
with a final-layer-only `--context_hidden_cache_dir` cache.

For a self-flow DLM checkpoint, keep its packaged `flow_matching_core/src`
directory beside `modeling_poredlm.py`; the checkpoint adds that source tree
when its custom HF model is loaded. Stage 4 supports both the newer
`flow_denoiser` interface and older `elf_denoiser` checkpoints. Use
`--unfreeze_denoiser_last_n_layers` and `--ode_*` for new runs. The prior
`--unfreeze_elf_last_n_layers` and `--elf_ode_*` option names remain aliases for
existing launch scripts.

W&B is disabled by default. Enable it by appending `--use_wandb` and the desired
W&B arguments; credentials should be provided by the environment rather than
stored in this repository.

On a MACA node, the launcher automatically sources `src/poredlm/training/set_env.sh`.
Set `SOURCE_TRAINING_ENV=1` to force this on a node where `/opt/maca` is mounted
later by the runtime.
