# Stage 2: Continuous BERT

Stage 2 trains a BERT-style contextual model on continuous embeddings produced
online by the frozen Stage 1 CNN. It does not read or write `features.npy` and
there are no token IDs or vocabulary lookup.

Set `cnn.checkpoint` to the Stage 1 HF checkpoint directory, and set
`data.train.paths` and `data.valid.paths` to the same raw signal paths used by
Stage 1. The model receives `[B, length, 768]` continuous embeddings from the
frozen CNN, masks feature positions, and predicts the original continuous
vectors.

Configure the CNN checkpoint and raw signal paths in
`runs/continuous_bert/config.yaml`:

```yaml
cnn:
  checkpoint: /path/to/stage1/save/continuous_cnn/latest
data:
  train:
    paths:
      - /path/to/raw/train
  valid:
    paths:
      - /path/to/raw/valid
```

The BERT checkpoint is also saved in Hugging Face format under the configured
output directory.

Checkpoints contain `config.json`, `model.safetensors`,
`modeling_continuous_bert.py`, `training_config.yaml`, and `trainer_state.pt`
inside `step_xxx`, `final`, and `latest` directories. They can be loaded with:

```python
from modeling_continuous_bert import ContinuousBert

model = ContinuousBert.from_pretrained("save/continuous_bert/latest")
```

W&B logging is enabled by default in `runs/continuous_bert/config.yaml`.
Authenticate with `wandb login` before training. Only the main DDP process
creates the W&B run. Set `wandb.enabled: false` to disable it, or use
`WANDB_MODE=offline` for offline logging.

```bash
bash runs/continuous_bert/run_train.sh
```

The launcher uses `torchrun`; set `CUDA_VISIBLE_DEVICES` and
`NPROC_PER_NODE` to match the available GPUs.
