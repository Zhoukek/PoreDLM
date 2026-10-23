# Context hidden cache for Stage 4 basecalling

The cache stores the frozen DLM `context_encoder` output for each distinct tokenized input. It is intended for `backbone_type=dlm` with `feature_source=ode_hidden` or `context_hidden`; with `ode_hidden`, basecall training still runs the trainable ELF ODE steps for every batch.

## 1. Build the cache

Use the same model, tokenizer settings, raw input data, and `token_offset` as the basecall run. Include every input split used for train, validation, and test. For example:

```bash
python src/poredlm/training_public/stage4_basecall_for_RNA/data/precompute_context_hidden.py \
  --jsonl_paths /path/to/all_labeled_jsonl_or_folder \
  --model_name_or_path /path/to/HF_DLM_checkpoint \
  --tokenizer_type bwav \
  --tokenizer_token_offset 128 \
  --token_offset 0 \
  --add_cls_token --cls_token_id 5 \
  --batch_size 8 \
  --output_dir /path/to/context_hidden_cache
```

For `tokens_*.npy` / `reference_*.npy` data, use `--npy_paths` in place of `--jsonl_paths`. The output is an fp16 flat feature file plus a SQLite index and metadata manifest. Required storage is approximately `unique_token_count × hidden_size × 2` bytes, plus a small index.

## 2. Use it in basecall training

Keep `feature_source=ode_hidden` and add this argument to the training command:

```bash
--context_hidden_cache_dir /path/to/context_hidden_cache
```

The trainer checks model/tokenizer metadata and looks up each input by its token IDs, so dataset order and train/validation splitting can differ from cache-generation order. The cache requires a frozen context encoder and `input_mask_ratio=0`; it can be reused while ELF layers train. Do not reuse it after changing the DLM checkpoint or tokenizer configuration.
