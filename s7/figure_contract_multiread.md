# Multi-read site-level figure contract

- Core conclusion: test whether strand-balanced aggregation of all reads at a genomic site enables the supplied PoreGPT representation to distinguish MOD from UNMOD at the site level.
- Statistical unit: one site-condition aggregate, not one read. There are ten primary observations: five sites times two conditions.
- Aggregation: average read features within each strand, then average the positive- and negative-strand centroids with equal weight.
- Validation: leave one genomic site out. The two held-out site-condition aggregates are predicted using only the other four sites.
- Uncertainty: stratified read bootstrap within each strand, with the original strand-specific read count retained. Bootstrap replicates are intervals, not independent primary samples.
- Panel a: PCA of ten all-read site-condition OLMo aggregate embeddings, with paired MOD/UNMOD points connected by site.
- Panel b: held-out-site MOD probabilities for the ten primary aggregates, with 95% read-bootstrap intervals.
- Panel c: site-level AUROC and balanced accuracy for aggregate signal statistics, token frequencies, and OLMo embeddings.
- Panel d: number of reads contributing to each aggregate and the positive/negative strand composition.
- Main limitation: only five genomic sites are available, and MOD/UNMOD remain confounded with sequencing run.
- Output: editable SVG, PDF, 600-dpi TIFF, and 300-dpi PNG with opaque white background.
