# Conditional token generation to waveform

This tool selects indexed sequences from the Stage 3 eval token shards, keeps a
configurable content-token prefix, generates the following tokens with the HF
DLM, and compares three aligned waveforms in one figure:

1. Reference token IDs → Stage-1 codebook → Stage-1 tokenizer decoder.
2. Generated token IDs → Stage-1 codebook → Stage-1 tokenizer decoder.
3. Generated token IDs → DLM hidden states → trained waveform decoder.

Edit `config.yaml`, especially `models.waveform_decoder_checkpoint`, then run:

```bash
bash run.sh
```

The common 1000-condition / 200-generation experiment can be overridden without
editing YAML:

```bash
bash run.sh \
  --condition-length 1000 \
  --generation-length 200 \
  --sample-indices 0,10,25 \
  --output-dir outputs/cond1000_gen200
```

`condition_length` and `generation_length` count content/codebook tokens; BOS and
EOS are handled by the script. Output includes one PNG and NPZ per sample plus a
`summary.json` containing token accuracy, waveform MSE, and Pearson correlation.

The DLM vocabulary uses an offset of 128 for waveform-codebook tokens. By
default, generation stops with an error if the DLM emits a special or otherwise
non-codebook token. `generation.invalid_token_policy: clip` exists only for
diagnostic plotting and should not be used for quantitative results.
