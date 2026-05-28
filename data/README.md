# Data Layout

Large datasets are intentionally ignored by Git.

Expected local layout:

```text
data/
  raw/          # original nanopore signal files
  interim/      # temporary converted files
  processed/    # model-ready tensors, embeddings, and labels
  external/     # third-party references or annotations
```
