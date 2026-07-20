# MOD/UNMOD 5mC signal embedding test

## Data and site selection

- BED source: `HG002_250F500490011.bed.gz`, chr19 sites with `percent_modified=100.00`.
- Corpus constraints: MOD and UNMOD each have at least 5 reads whose complete 10-nt window is aligned without an indel.
- Window definition: `[site_pos0-5, site_pos0+5)`, ten reference bases in total.
- Sites were required to be at least 1 Mb apart to prevent the same long read from appearing at multiple sites.
- Final data: 5 sites, 50 MOD reads and 106 UNMOD reads; all 156 read IDs are unique.
- Every extracted record is marked `normal_mode=apple` and `signal_base_shift=-4`.

## Model processing

- Model: `HF_RSQ742C12A511_DNAOLMO_V602`.
- The corpus signal was passed directly to the encoder; Apple normalization was not applied again.
- Encoder output: first residual-quantizer layer, raw token range 124-1983, followed by the model's documented `+128` vocabulary offset.
- Embedding: mean of the final hidden states from the 24-layer OLMo2 base model, producing one 1024-dimensional vector per read-window.

## Separation result

Evaluation used leave-one-site-out cross-validation: each site was predicted using a classifier trained only on the other four sites.

| Feature | Pooled AUROC | Balanced accuracy |
|---|---:|---:|
| Signal statistics | 0.583 | 0.545 |
| Token frequency | 0.582 | 0.555 |
| OLMo embedding | 0.508 | 0.514 |

The OLMo embedding did not reliably separate MOD from UNMOD. Its pooled AUROC and balanced accuracy were near random, site-specific AUROC varied from 0.348 to 0.693, and the within-site label permutation test was not significant (`P=0.400`, 1,000 permutations).

This result does not show that the model is generally insensitive to 5mC. It only shows that this specific 10-nt window representation and mean-pooling test did not generalize across these five sites. MOD and UNMOD also come from separate sequencing runs, so any observed dataset separation would remain confounded by run-level technical differences rather than proving a 5mC-specific effect.
