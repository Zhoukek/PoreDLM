# VQE-DLM ODE multi-read site-level evaluation

## Model processing

- Model: Stage-3 DLM under `HF_VQE768C08A001_DNADLLM_V001/hf_dlm`.
- The existing 64K VQE token sequences were reused for all 156 10-nt signal windows.
- Inputs use raw VQE ID + 128, with BOS=2, EOS=3, and PAD=1.
- Following the model README, inference used `return_context=True`, `return_ode_hidden=True`, `ode_steps=2`, `ode_start_t=0.95`, and `ode_self_cond_cfg_scale=0.0`.
- Mean pooling of `ode_hidden_state` over signal-token positions only produced one 768-dimensional embedding per read; BOS, EOS, and padding were excluded.

## Multi-read aggregation

Reads were not classified individually. Within every site-condition group, positive-strand and negative-strand embeddings were averaged separately, followed by an equal-weight average of the two strand centroids. This produced ten primary site-condition aggregates from 156 unique reads.

## Results

| Aggregated feature | Site-level AUROC | Balanced accuracy | Accuracy |
|---|---:|---:|---:|
| Signal statistics | 0.520 | 0.600 | 0.600 |
| 64K VQE token frequency | 0.200 | 0.500 | 0.500 |
| VQE-DLM ODE embedding | 0.400 | 0.300 | 0.300 |

The VQE-DLM ODE aggregate embedding did not distinguish MOD from UNMOD at unseen genomic sites. The exact 32-assignment site-label swap test gave `P=0.750`.

Only five genomic sites are included, and MOD/UNMOD originate from separate sequencing runs. The result therefore cannot establish or exclude a 5mC-specific representation mechanism.
