# PoreLM Stage-3 Training

This directory contains the publishable Stage-3 training code for PoreLM. It
trains a transformer denoiser in the latent space of a Stage-2 signal encoder.
The current objective is conditional flow matching.

## Layout

```text
configs/                  experiment configuration
scripts/train.sh          local or multi-GPU launcher
scripts/train_porelm.py   stable training entrypoint
scripts/convert_porelm_to_hf.py
                          stage-3 checkpoint to HuggingFace export
src/porelm/
  config.py               training configuration schema
  train.py                stable trainer base: checkpoint, eval, logging
  trainer_flow_matching.py
                          flow-matching microbatch objective
  checkpoint.py           sharded and unsharded checkpoint support
  optim.py                optimizer, gradient clipping, LR schedules
  data/                   resumable iterable dataset infrastructure
  data_flow_matching.py   token dataset and condition masks
  model_porelm.py         Stage-2 context encoder plus denoiser model
  flow_matching_core/     PyTorch denoiser blocks and sampling helpers
src/hf_porelm/            lightweight HF causal-model compatibility package
src/porelm_training/      objective-registry prototype for future objectives
```

The default launcher uses the stable `porelm` training runtime so FSDP,
optimizer state, sharded/unsharded checkpoints, dataloader progress, evaluator
hooks, and logging behavior stay close to the original large-scale training
path. The `porelm_training/objectives` package is kept as the cleaner extension
point for later self-flow work, but it is not the default launcher yet.

## Installation

From this directory:

```bash
python -m pip install -e .
```

Install experiment tracking support only when needed:

```bash
python -m pip install -e '.[tracking]'
```

## Training

Edit `configs/porelm_flow_matching.yaml` to set the Stage-2 encoder and token data
paths. Launch one process per GPU:

```bash
NUM_PROCESSES=4 bash scripts/train.sh configs/porelm_flow_matching.yaml
```

Configuration values can be overridden from the command line:

```bash
NUM_PROCESSES=4 bash scripts/train.sh configs/porelm_flow_matching.yaml \
  run_name=ablation-01 dlm.conditioning_mode=unconditional
```

Checkpoints contain model, optimizer, scheduler, trainer progress, dataloader
state, and random-number generator states. Set `load_path` to a checkpoint
directory to resume a run, and use `reset_optimizer_state` or
`reset_trainer_state` only when you intentionally want a partial restore.

## HuggingFace Export

After training, export a stage-3 checkpoint with:

```bash
PYTHONPATH=src python scripts/convert_porelm_to_hf.py /path/to/checkpoint --output-dir /path/to/hf_model
```
