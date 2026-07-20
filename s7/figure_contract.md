# Figure contract

- Core conclusion: determine whether the supplied PoreGPT representation separates MOD and UNMOD read windows at five spatially separated, fully modified chr19 sites shared at >=5x depth.
- Evidence chain: embedding PCA -> held-out-site prediction distributions -> held-out-site performance against technical and token baselines -> selected-site coverage and sample counts.
- Archetype: quantitative four-panel grid.
- Panel a: unsupervised PCA of mean-pooled final OLMo hidden states; color encodes dataset and marker encodes held-out genomic site.
- Panel b: out-of-fold OLMo MOD probabilities by site and dataset; every point is one unique read.
- Panel c: leave-one-site-out AUROC and balanced accuracy for technical signal statistics, token frequencies, and OLMo embeddings; small points show individual test sites and large points show pooled predictions.
- Panel d: complete-window depths for the five selected sites, confirming all selection constraints.
- Statistical unit: one unique read-window; 50 MOD and 106 UNMOD reads across five sites.
- Validation: each site's reads are predicted by a classifier trained on the other four sites. Labels are never used for PCA fitting in the classifier pipeline, except through logistic regression.
- Primary caveat: MOD and UNMOD come from separate sequencing runs, so separation cannot be attributed uniquely to 5mC without matched-run controls or per-read modification labels.
- Output: editable SVG, PDF, 600-dpi TIFF, and 300-dpi PNG with white background and embedded/selectable text where supported.
