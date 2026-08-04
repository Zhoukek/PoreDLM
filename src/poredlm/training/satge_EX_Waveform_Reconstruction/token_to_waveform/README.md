# Conditional token generation to waveform

This tool selects indexed sequences from the Stage 3 eval token shards, masks a
configurable middle region, repairs it independently with the HF DLM and its
embedded Stage-2 BERT, then decodes all token sequences with the same Stage-1
codebook and tokenizer decoder:

1. Reference token IDs → Stage-1 codebook → Stage-1 tokenizer decoder.
2. DLM-repaired token IDs → Stage-1 codebook → Stage-1 tokenizer decoder.
3. BERT-repaired token IDs → Stage-1 codebook → Stage-1 tokenizer decoder.
Both paths use the original Stage-1 codebook and tokenizer decoder; no separately
trained waveform decoder is needed. Edit `config.yaml`, then run:

```bash
bash run.sh
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
BOS is handled by the script. Tokens before and after the masked interval remain
fixed conditions. Output includes one PNG and NPZ per sample plus a `summary.json`
containing separate DLM/BERT token accuracy and masked-region waveform MSE.

The PNG contains three panels: full reference versus DLM, masked-region reference
versus DLM, and masked-region reference versus BERT. MSE is calculated only over
the waveform interval corresponding to the masked tokens.

The DLM vocabulary uses an offset of 128 for waveform-codebook tokens. By
default, `restrict_to_codebook: true` masks BOS/EOS/PAD at generated positions
and selects only IDs from the waveform codebook. The additional validation still
stops on invalid IDs. `generation.invalid_token_policy: clip` exists only for
diagnostic plotting and should not be used for quantitative results.
