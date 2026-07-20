# VQE-BERT multi-read figure contract

- Model: 64K `VQE768C08A001` signal codec plus `HF_BERT_part` 12-layer bidirectional Transformer.
- Input invariant: existing corpus signal is Apple-normalized and uses signal-base shift -4; no feature extractor is called again.
- Token protocol: raw VQE ID + 128, with BOS=2 and EOS=3 following the model README.
- Read embedding: mean final BERT hidden state over signal-token positions only; BOS, EOS, and padding are excluded.
- Primary unit: one strand-balanced all-read site-condition aggregate, giving ten observations across five sites.
- Validation: leave one site out, 1,000 strand-stratified read bootstraps for intervals, and all 32 within-site label assignments for the exact embedding test.
- Figure panels: aggregate PCA, held-out aggregate probabilities, site-level metrics, and contributing read/strand support.
- Output: PNG, PDF, SVG, and 600-dpi TIFF under the parent `s7/plot` directory.
