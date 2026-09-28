# Stage 2: Continuous BERT

Stage 2 trains a BERT-style contextual model on continuous embeddings produced
online by the frozen Stage 1 CNN. It does not read or write `features.npy` and
there are no token IDs, discrete codebook, tokenizer vocabulary, or token
embedding lookup.

The training route is:

```text
raw signal [B, signal_length]
  -> Stage 1 ContinuousSignalCNN
  -> continuous embeddings [B, sequence_length, feature_dim]
  -> mask a subset of positions
  -> ContinuousBert encoder
  -> prediction head reconstructs the masked CNN embeddings
```

The BERT encoder output and the reconstruction prediction are intentionally
different tensors. The encoder output is the general contextual representation
for downstream tasks; the prediction head output is only used by the Stage 2
reconstruction objective.

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

The BERT checkpoint is saved in Hugging Face format under the configured output
directory. A typical output directory is:

```text
save/continuous_bert/
  step_5000/
    config.json
    model.safetensors
    modeling_continuous_bert.py
    training_config.yaml
    trainer_state.pt
  final/
  latest/
```

`latest` is a copy of the most recently saved checkpoint. You can also load a
specific `step_xxx` or `final` directory.

Checkpoints contain `config.json`, `model.safetensors`,
`modeling_continuous_bert.py`, `training_config.yaml`, and `trainer_state.pt`
inside `step_xxx`, `final`, and `latest` directories. They can be loaded with:

```python
from pathlib import Path
import sys
import torch

stage2_root = Path("/path/to/PoreDLM/src/poredlm/training_another_way/stage2_bert_train")
sys.path.insert(0, str(stage2_root))

from modeling_continuous_bert import ContinuousBert

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ContinuousBert.from_pretrained(
    "/path/to/save/continuous_bert/latest"
).to(device)
model.eval()
```

The directory must contain `config.json`, `model.safetensors`, and the local
`modeling_continuous_bert.py` file. Loading with `ContinuousBert.from_pretrained`
restores both the architecture and learned weights; the original YAML training
configuration is kept separately as `training_config.yaml` for reference.

## Calling the Encoder

The main inference API is `encode`. It returns the final Transformer hidden
state before the continuous reconstruction prediction head:

```python
from poredlm.training_another_way.stage1_cnn_train.modeling_continuous_cnn import (
    ContinuousSignalCNN,
)

cnn = ContinuousSignalCNN.from_pretrained(
    "/path/to/save/continuous_cnn/latest"
).to(device)
cnn.eval()

# signals: [batch, signal_length], for example [B, 6000]
signals = signals.to(device)
signal_lengths = torch.full(
    (signals.shape[0],), signals.shape[1], dtype=torch.long, device=device
)

with torch.no_grad():
    cnn_features = cnn.encode(signals).transpose(1, 2)

    # The CNN currently has stride 5.
    encoded_lengths = (signal_lengths + cnn.stride - 1) // cnn.stride
    attention_mask = torch.zeros(
        cnn_features.shape[:2], dtype=torch.long, device=device
    )
    positions = torch.arange(
        cnn_features.shape[1], device=device
    ).unsqueeze(0)
    attention_mask[positions < encoded_lengths.unsqueeze(1)] = 1

    hidden = model.encode(cnn_features, attention_mask)

print(hidden.shape)
# [batch, sequence_length, model.config.hidden_size]
```

For the current configuration, `feature_dim=768` and `hidden_size=768`, so the
CNN feature dimension and BERT hidden dimension are both 768. If they differ,
`input_projection` maps the CNN feature dimension to `hidden_size` internally.

`attention_mask` uses `1` for valid positions and `0` for padding. The sequence
length after the current CNN is approximately `signal_length / 5`. The model
raises an error if the resulting sequence is longer than
`config.max_position_embeddings`.

### Variable-Length Batches

The training dataset currently uses `chunk_size: 6000`, so each training
sample is cropped or padded to 6000 raw signal points before entering the CNN.
The model itself can process a batch of shorter, variable-length signals. In
that case, pad the raw signals to the longest signal in the batch, keep the
original lengths, and construct the BERT attention mask after the CNN:

```python
# Padded raw signals: [B, max_signal_length]
signals = signals.to(device)
signal_lengths = signal_lengths.to(device)  # [B], before padding

with torch.no_grad():
    cnn_features = cnn.encode(signals).transpose(1, 2)

encoded_lengths = (signal_lengths + cnn.stride - 1) // cnn.stride
attention_mask = torch.zeros(
    cnn_features.shape[:2], dtype=torch.long, device=device
)
positions = torch.arange(
    cnn_features.shape[1], device=device
).unsqueeze(0)
attention_mask[positions < encoded_lengths.unsqueeze(1)] = 1

hidden = model.encode(cnn_features, attention_mask)
```

For example, raw signal lengths `[6000, 5200, 4100, 3000]` can be padded to a
single `[4, 6000]` tensor. With the current CNN stride, their valid BERT
lengths are approximately `[1200, 1040, 820, 600]`. Padding positions are
ignored by BERT through `attention_mask`. The maximum raw length of 6000 is
compatible with the default `max_position_embeddings: 1536` because the CNN
reduces it to approximately 1200 positions.

When constructing batches manually, keep the original `signal_lengths`; do not
replace them with the padded tensor length. The CNN convolution can affect a
small number of boundary positions near padding, so downstream code should
only use positions marked valid by `attention_mask`.

## Calling the Full Forward Method

`forward` returns a `ContinuousBertOutput` object:

```python
with torch.no_grad():
    output = model(
        cnn_features,
        attention_mask,
        target_embeddings=cnn_features,
        mask_positions=None,
    )

hidden = output.last_hidden_state
prediction = output.predictions
loss = output.loss
```

The fields are:

```text
output.last_hidden_state  [B, L, hidden_size]
  Final BERT representation before prediction_head. Use this for downstream
  tasks and Stage 4 basecalling.

output.predictions        [B, L, feature_dim]
  Output of prediction_head. It is trained to reconstruct the masked CNN
  embedding and should not normally be used as the general representation.

output.loss               scalar or None
  SmoothL1 reconstruction loss plus 0.1 times cosine loss. It is only computed
  when target_embeddings and mask_positions are provided.
```

To reproduce the masked reconstruction call used during training:

```python
mask_positions = (
    (torch.rand(attention_mask.shape, device=device) < 0.15)
    & attention_mask.bool()
)

with torch.no_grad():
    output = model(
        cnn_features,
        attention_mask,
        target_embeddings=cnn_features,
        mask_positions=mask_positions,
    )

print(output.loss.item())
```

During normal downstream inference, do not mask the input and use:

```python
hidden = model.encode(cnn_features, attention_mask)
```

## Using It in Stage 4

Stage 4 loads the CNN and BERT checkpoints together and calls the same
`encode` method internally. The Stage 4 configuration should point to the HF
directories, not to a `.pt` trainer checkpoint:

```yaml
backbone:
  cnn_checkpoint: /path/to/save/continuous_cnn/latest
  bert_checkpoint: /path/to/save/continuous_bert/latest
  freeze_cnn: true
  freeze_bert: true
```

The Stage 4 basecalling head receives `last_hidden_state`. It does not receive
`output.predictions` from the BERT reconstruction head.

W&B logging is enabled by default in `runs/continuous_bert/config.yaml`.
Authenticate with `wandb login` before training. Only the main DDP process
creates the W&B run. Set `wandb.enabled: false` to disable it, or use
`WANDB_MODE=offline` for offline logging.

```bash
bash runs/continuous_bert/run_train.sh
```

The launcher uses `torchrun`; set `CUDA_VISIBLE_DEVICES` and
`NPROC_PER_NODE` to match the available GPUs.
