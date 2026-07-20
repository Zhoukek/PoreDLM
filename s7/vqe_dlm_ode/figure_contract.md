# VQE-DLM ODE multi-read figure contract

- Core conclusion: test whether ODE-refined Stage-3 DLM signal representations separate strand-balanced MOD and UNMOD site aggregates at unseen genomic sites.
- Archetype: four-panel quantitative grid.
- Input protocol: 64K VQE token ID + 128, with BOS=2, EOS=3, and PAD=1.
- DLM protocol: README parameters `ode_steps=2`, `ode_start_t=0.95`, and `ode_self_cond_cfg_scale=0.0`.
- Read embedding: mean `ode_hidden_state` over signal-token positions only; BOS, EOS, and padding are excluded.
- Primary unit: one strand-balanced all-read site-condition aggregate, giving ten observations across five sites.
- Validation: leave one site out, 1,000 strand-stratified read bootstraps for intervals, and all 32 within-site label assignments for the exact embedding test.
- Panel a: PCA of the ten site-condition aggregate embeddings, with each MOD/UNMOD pair connected by site.
- Panel b: held-out aggregate MOD probabilities and 95% read-bootstrap intervals.
- Panel c: site-level AUROC and balanced accuracy against signal-statistic and token-frequency baselines.
- Panel d: contributing read counts and positive/negative strand composition.
- Reviewer risk: only five sites are available, and condition remains confounded with sequencing run.
- Output: Python/matplotlib PNG, PDF, editable-text SVG, and 600-dpi TIFF under the parent `s7/plot` directory.
