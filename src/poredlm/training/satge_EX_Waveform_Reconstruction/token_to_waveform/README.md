# Conditional token generation to waveform

This tool selects indexed sequences from the Stage 3 eval token shards and
repairs a configurable interval with one selected generator per run:

1. `generator_type: dlm`: conditional DLM infill using both sides, plus the
   embedded Stage-2 BERT baseline.
2. `generator_type: gpt`: causal, left-to-right prediction of the interval.

All repaired sequences use the original Stage-1 codebook and tokenizer decoder;
no separately trained waveform decoder is needed. Set the model paths in
`config.yaml`, then run:

```bash
bash run.sh
```

Select the generator from YAML or the command line. Separate output directories
make it easy to retain both experiments:

```bash
bash run.sh --generator-type dlm --output-dir outputs/dlm
bash run.sh --generator-type gpt --output-dir outputs/gpt
```

For example, the following repairs content-token positions 1001 through 1100
(CLI coordinates are zero-based, so `mask-start=1000`):

```bash
bash run.sh \
  --total-length 1200 \
  --mask-start 1000 \
  --mask-length 100 \
  --sample-indices 0,10,25 \
  --output-dir outputs/infill_1001_1100
```

`total_length`, `mask_start`, and `mask_length` count content/codebook tokens;
BOS is handled by the script. For DLM, tokens before and after the interval are
conditions. For GPT, only BOS and tokens before the interval are visible during
generation; the reference suffix is stitched back afterward. Output includes one
generator-prefixed PNG and NPZ per sample plus `summary_dlm.json` or
`summary_gpt.json`.

The DLM PNG contains three panels: full reference versus DLM, masked-region
reference versus DLM, and masked-region reference versus BERT. The GPT PNG is
separate and contains full-sequence and masked-region reference-versus-GPT panels.
MSE is calculated over the waveform interval corresponding to the predicted tokens.

Each summary entry stores `reference_region_token_ids` and
`predicted_region_token_ids`; DLM runs also store
`bert_predicted_region_token_ids`.

The codec loader supports both `modeling_pore_vq_codec.py` and
`modeling_pore_codec.py`. The latter is the packed residual-RSQ/FSQ codec used by
`HF_RSQ742C12A511_MIXOLMO_V606`; decoding uses its public `decode_token()` method.
Set `models.gpt_tokenizer_path` to that model's `encoder` directory. Unless
`data.gpt_codebook_size` is explicitly set, the packed GPT token vocabulary is
inferred as `product(fsq_levels) ** codebook_nqtz`. Set
`data.gpt_token_offset` to the offset used when the GPT training tokens were built.
The DLM layout is configured independently with `data.dlm_token_offset` and
`data.dlm_codebook_size`; its default codebook size is 65536. If an explicit
`gpt_codebook_size` disagrees with the loaded RSQ encoder, the script stops.
The current default `data.eval_path` contains VQE tokens, so GPT/RSQ experiments
must set `data.gpt_eval_path` to token shards created with the matching RSQ codec;
VQE token IDs and packed RSQ token IDs are not interchangeable.

The DLM vocabulary uses an offset of 128 for waveform-codebook tokens. By
default, `restrict_to_codebook: true` masks BOS/EOS/PAD at generated positions
and selects only IDs from the waveform codebook. The additional validation still
stops on invalid IDs. `generation.invalid_token_policy: clip` exists only for
diagnostic plotting and should not be used for quantitative results.
