# Multi-read site-level MOD/UNMOD analysis

## Aggregation

This revision does not classify individual reads. For each site and dataset, all aligned 10-nt read-window embeddings were aggregated as follows:

1. Mean the positive-strand read embeddings.
2. Mean the negative-strand read embeddings.
3. Average the two strand centroids with equal weight.

The primary statistical unit is therefore one site-condition aggregate. Five sites and two conditions produce ten primary observations. The 156 underlying reads are used only to calculate these aggregates and their bootstrap uncertainty.

## Validation

- Leave-one-site-out validation: both MOD and UNMOD aggregates from one site are held out together.
- Training uses only the other four genomic sites.
- Read bootstrap is performed separately within each strand while preserving each strand's observed read count.
- Bootstrap replicates provide 95% intervals; they are not counted as independent observations.
- The OLMo significance test enumerates all 32 possible within-site MOD/UNMOD label assignments.

## Results

| Aggregated feature | Site-level AUROC | Balanced accuracy | Accuracy |
|---|---:|---:|---:|
| Signal statistics | 0.520 | 0.600 | 0.600 |
| Token frequency | 0.640 | 0.700 | 0.700 |
| OLMo embedding | 0.400 | 0.400 | 0.400 |

The all-read OLMo aggregate did not separate MOD from UNMOD across unseen sites. Its exact site-label swap test was not significant (`P=0.625`). Aggregated token frequencies were somewhat better (`AUROC=0.64`), but this is based on only five sites and is not sufficient evidence of robust modification detection.

MOD and UNMOD are from separate sequencing runs. Even a stronger classification result would need matched-run controls before it could be attributed specifically to 5mC.
