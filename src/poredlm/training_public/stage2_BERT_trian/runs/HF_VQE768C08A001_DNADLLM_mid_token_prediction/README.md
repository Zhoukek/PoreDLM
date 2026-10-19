# RF-aware middle-token prediction

This run extends the normal MLM objective with configurable span masking.
The anchor token is sampled using `model.mask_probability`. The input span
around each anchor is replaced by `[MASK]`, while loss is computed only on the
configured target interval.

For the old Stage1 CNN tokenizer (`RF=27`, `stride=5`), the default settings
mask `i-3 ... i+3` and compute loss only at `i`:

```yaml
model:
  masking:
    use_rf_aware_masking: true
    mask_left_context: 3
    mask_right_context: 3
    center_loss_token_count: 3
    loss_left_context: 0
    loss_right_context: 0
    force_mask_context: true
```

Set `use_rf_aware_masking: false` to use the original BERT masking behavior
(single-token sampling with the existing 80/10/10 replacement rule). The
range fields are ignored when the switch is false.

To train on a wider target interval, for example `i-1 ... i+1`, set
`loss_left_context: 1` and `loss_right_context: 1`. The mask interval should
be at least as wide as the loss interval.

`force_mask_context: true` is intentional: it prevents the 80/10/10 MLM rule
from leaving an expanded-context token unchanged, which would reintroduce the
information leak this run is designed to remove.

`loss_left_context` and `loss_right_context` values instead.
`center_loss_token_count` controls the number of supervised tokens centered on
the anchor. In this mode, `mask_left_context` and `mask_right_context` are the
number of additional non-loss tokens on each side. Therefore, with context 3
on both sides and `center_loss_token_count: 3`, the input masks `i-4...i+4`
(9 tokens) and the loss is computed on `i-1...i+1` (3 tokens). Set it to `0`
to use the explicit `loss_left_context` and `loss_right_context` values
instead.
`loss_left_context` and `loss_right_context` values instead.
