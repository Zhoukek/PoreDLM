# DLM waveform decoder training

This stage freezes both pretrained models and trains only a copy of the Stage 1
convolutional decoder:

1. Load shifted token sequences from the same headerless `.npy` + `.csv.gz`
   shards as Stage 2/3.
2. Run the frozen DLM and select a configured hidden state (ODE by default).
3. Remove BOS/EOS and the token offset, then call the frozen Stage 1 tokenizer's
   `decode_token()` to create the waveform label online.
4. Train only `WaveformDecoder`; padded waveform samples are excluded from loss.

Before running, edit `config.yaml`. Both `model.dlm_path` and
`model.tokenizer_path` must be complete Hugging Face `save_pretrained`
directories containing a config and model weights.

```bash
bash run.sh
```

The final and periodic `checkpoint.pt` files contain only the trainable decoder
state plus optimizer/scheduler state. The intended inference input shape is
`[batch, token_count, 768]`. With the Stage 1 convolution parameters, the exact
output length is `token_count * 5 - 3`.
