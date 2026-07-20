# QA notes

- All scripts compile under the `torch2d1` Python environment.
- Site coverage is CIGAR-derived and requires a complete gap-free 10-nt reference window.
- Extracted windows: 156 expected, 156 found; all read IDs are unique.
- Normalization invariant: all records report `normal_mode=apple`; no feature extractor was called during tokenization.
- Alignment invariant: all records report `signal_base_shift=-4`.
- Encoder token IDs fit the 2,401-token first-layer codebook; shifted OLMo IDs remain within the 2,560-token vocabulary.
- Figure visually inspected at original PNG resolution after revision; no panel-title or label overlap remains.
- PNG is 2160 x 1950 with opaque white background; TIFF is 600 dpi; PDF is one page; SVG contains editable text elements.

## Multi-read site-level revision

- Primary units: 10 strand-balanced site-condition aggregates, built from 156 unique reads.
- Every site-condition group contains both positive- and negative-strand reads.
- No individual read receives a primary MOD/UNMOD classification.
- Bootstrap intervals preserve each group's observed strand-specific read count and are not treated as independent samples.
- Leave-one-site-out validation holds out both conditions and every read from the test site.
- The revised PNG is 2160 x 1950 with an opaque white background; matching PDF, SVG, and 600-dpi TIFF files were exported.
