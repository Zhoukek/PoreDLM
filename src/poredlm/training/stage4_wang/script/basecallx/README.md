# BasecallX

`basecallx` is a cleaner orchestration layer for the existing `dcbasecaller` code.
It keeps the mature algorithm modules from `script/dcbasecaller/basecall` and
rebuilds the outer training pipeline with clearer ownership:

- `config.py`: typed experiment configuration.
- `data.py`: JSONL discovery, split policy, streaming/eager dataset construction.
- `modeling.py`: model/head construction and delayed unfreeze.
- `optim.py`: optimizer and scheduler.
- `checkpointing.py`: save/resume model, optimizer, scheduler, and `model_config`.
- `train.py`: the training loop only.

## Install

From this repository root:

```bash
export PYTHONPATH="$PWD/script/dcbasecaller:$PWD/script/basecallx:${PYTHONPATH:-}"
```

Or install editable packages:

```bash
pip install -e script/dcbasecaller
pip install -e script/basecallx
```

## Train

```bash
accelerate launch --num_processes 4 -m basecallx.train \
  --jsonl_paths /path/to/jsonl_dir \
  --model_name_or_path /path/to/backbone \
  --output_dir outputs/run1 \
  --head_type ctc \
  --train_decoder ctc_viterbi \
  --group_by record \
  --streaming \
  --max_steps_per_epoch 1000
```

Important differences from the legacy trainer:

- `--streaming` defaults to on.
- `--group_by record` or `record_per_file` without streaming is blocked by default,
  because it can eagerly load very large JSONL datasets.
- `--max_steps_per_epoch` really stops an epoch after N batches.
- `--steps_per_epoch` is only for scheduler sizing when you do not want to auto-estimate.

Stage3 PoreDLM backbones are supported through `--dlm_output last|context|ode`.
For the exported ODE protocol, use `--dlm_output ode`,
`--dlm_ode_steps 2`, `--dlm_ode_start_t 0.98`, and
`--dlm_ode_self_cond_cfg_scale 0.0`. Keep `--hidden_layer -1` and start
with the backbone frozen.

## Eval / Infer

The eval and inference entry points currently forward to the compatible legacy
commands so saved checkpoints stay usable:

```bash
basecallx-eval --ckpt ckpt_best.pt --jsonl_paths /path/to/val --model_name_or_path /path/to/backbone
basecallx-infer --ckpt ckpt_best.pt --jsonl_gz reads.jsonl.gz --out preds.fastq
```
