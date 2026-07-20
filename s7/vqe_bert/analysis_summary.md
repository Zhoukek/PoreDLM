# VQE-BERT multi-read site-level evaluation

## Model processing

- Codec: `/mnt/zzbnew/poregpt/models/HF_VQE768C08A001_DNADLLM_V001/encoder`
- BERT: `/mnt/zzbnew/poregpt/models/HF_VQE768C08A001_DNADLLM_V001/HF_BERT_part`
- The 156 existing 10-nt signal windows were used directly; all are marked `normal_mode=apple` and `signal_base_shift=-4`.
- Apple normalization was not applied again.
- The codec generated 13-100 tokens per read from a 65,536-entry codebook.
- Following the model README, token IDs were shifted by 128 and surrounded by BOS/EOS.
- Mean pooling of final BERT signal-token hidden states produced one 768-dimensional embedding per read.

## Multi-read aggregation

Reads were not classified individually. Within every site-condition group, positive-strand reads and negative-strand reads were averaged separately, followed by an equal-weight average of the two strand centroids. This produced ten primary site-condition aggregates from 156 unique reads.

## Results

| Aggregated feature | Site-level AUROC | Balanced accuracy | Accuracy |
|---|---:|---:|---:|
| Signal statistics | 0.520 | 0.600 | 0.600 |
| 64K VQE token frequency | 0.200 | 0.500 | 0.500 |
| VQE-BERT embedding | 0.520 | 0.600 | 0.600 |

The VQE-BERT aggregate embedding did not reliably distinguish MOD from UNMOD at unseen genomic sites. The exact 32-assignment site-label swap test gave `P=0.500`. Relative to the previous OLMo aggregate result (`AUROC=0.400`), VQE-BERT is numerically closer to random rather than demonstrating significant improvement.

Only five genomic sites are included, and MOD/UNMOD originate from separate sequencing runs. The result therefore cannot establish or exclude a 5mC-specific representation mechanism.
