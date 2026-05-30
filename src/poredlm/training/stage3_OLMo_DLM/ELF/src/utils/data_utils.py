import json
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
from datasets import DatasetDict, load_dataset as hf_load_dataset, load_from_disk
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.encoder_utils import build_self_attn_cond_masks
from utils.logging_utils import log_for_0

PRNGKey = jax.random.PRNGKey


def get_pad_token_id(tokenizer, pad_token="pad"):
    """Resolve the token id used for padding, optionally using EOS as pad."""
    token_id = tokenizer.eos_token_id if pad_token == "eos" else tokenizer.pad_token_id
    if token_id is None:
        raise ValueError(
            f"Tokenizer has no pad_token_id or eos_token_id."
        )
    return token_id


def prepare_batch(batch: Dict, config, rng: PRNGKey):
    """Convert numpy batch to JAX arrays and sample label-drop decisions."""
    result = {k: jnp.array(v) if isinstance(v, np.ndarray) else v for k, v in batch.items()}
    input_ids = jnp.array(batch["input_ids"])
    batch_size = input_ids.shape[0]

    label_drop_mask = jnp.zeros((batch_size,), dtype=jnp.bool_)
    if config.label_drop_prob > 0:
        rng, drop_rng = jax.random.split(rng)
        label_drop_mask = jax.random.uniform(drop_rng, (batch_size,)) < config.label_drop_prob
    result["label_drop_mask"] = label_drop_mask
    return result


def pad_and_truncate(ids_list, target_len, pad_token_id):
    """Pad or truncate sequences to target_len, return stacked array and lengths."""
    padded, lengths = [], []
    for ids in ids_list:
        orig_len = min(len(ids), target_len)
        ids = ids[:target_len]
        if orig_len < target_len:
            ids = np.concatenate([ids, np.full(target_len - orig_len, pad_token_id, dtype=ids.dtype)])
        padded.append(ids)
        lengths.append(orig_len)
    return np.stack(padded), np.array(lengths)


def get_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
    max_seq_length: int = 512,
    pad_token_id: int = 0,
    max_input_seq_length: Optional[int] = None,
    distributed: bool = True,
):
    """Create a DataLoader."""

    def collate_fn(batch_list):
        input_ids_list = [np.array(item["input_ids"]) for item in batch_list]

        if "condition_input_ids" in batch_list[0]:
            seq_list, cond_lens = [], []
            for item in batch_list:
                cond = np.array(item["condition_input_ids"])[:max_input_seq_length]
                inp = np.array(item["input_ids"])
                seq_list.append(np.concatenate([cond, inp]))
                cond_lens.append(len(cond))
            cond_lens = np.array(cond_lens)
        else:
            seq_list = input_ids_list
            cond_lens = np.zeros(len(input_ids_list), dtype=np.int32)

        ids, total_lens = pad_and_truncate(seq_list, max_seq_length, pad_token_id)
        pos = np.arange(max_seq_length)[None, :]
        is_cond = pos < cond_lens[:, None]
        is_valid = pos < total_lens[:, None]
        encoder_attn, attn, pred = build_self_attn_cond_masks(is_cond, is_valid, xp=np)
        result = {
            "input_ids": ids,
            "encoder_attention_mask": encoder_attn,
            "attention_mask": attn,
            "cond_seq_mask": pred,
        }
        for key in ("index", "input", "target"):
            if key in batch_list[0]:
                result[key] = [item[key] for item in batch_list]
        return result

    common = dict(
        batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn,
        drop_last=drop_last, persistent_workers=num_workers > 0,
    )
    if distributed:
        sampler = DistributedSampler(
            dataset, num_replicas=jax.process_count(), rank=jax.process_index(),
            shuffle=shuffle, drop_last=drop_last,
        )
        return DataLoader(dataset, sampler=sampler, **common)
    return DataLoader(dataset, shuffle=shuffle, **common)


def load_jsonl_dataset(path, tokenizer, input_key="input", output_key="output"):
    """Load a JSONL eval set (one `{input, output}` example per line).

    Triggered by `eval.py` whenever `config.eval_data_path` ends with `.jsonl`;
    otherwise the standard `datasets.load_from_disk` Arrow path is used.
    """
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            examples.append({
                "index": i,
                "input": data[input_key],
                "target": data[output_key],
                "condition_input_ids": tokenizer(data[input_key], add_special_tokens=False)["input_ids"],
                "input_ids": tokenizer(data[output_key], add_special_tokens=False)["input_ids"],
            })
    return examples


# ============================================
# Dataset loading
# ============================================

def _looks_like_save_to_disk_arrow(ds) -> bool:
    """HF datasets uploaded via `save_to_disk` get loaded as a fake 1-row dataset
    where the columns are internal metadata fields like `_data_files`, `_fingerprint`,
    etc. Detect that here so we can fall back to `load_from_disk`."""
    return (
        len(ds) == 1
        and any(c.startswith("_") for c in ds.column_names)
        and not any(not c.startswith("_") for c in ds.column_names)
    )


def load_dataset_split(path: str, dataset_cache_dir=None):
    """Load a dataset. Tries HuggingFace Hub first; falls back to local on-disk Arrow.

    For HF repos uploaded via `dataset.save_to_disk()` instead of `push_to_hub()`,
    `load_dataset` silently returns a 1-row dataset of internal metadata. We detect
    that and re-download the repo, then load it via `load_from_disk`.
    """
    ds = None
    try:
        ds = hf_load_dataset(path, cache_dir=dataset_cache_dir)
    except Exception:
        ds = load_from_disk(path)

    # Normalize DatasetDict -> single split before detection.
    if isinstance(ds, DatasetDict):
        splits = list(ds.keys())
        if len(splits) != 1:
            raise ValueError(
                f"Expected dataset at {path!r} to have a single split, got {splits}."
            )
        ds = ds[splits[0]]

    # save_to_disk-style HF repo: fall back via snapshot_download + load_from_disk.
    if _looks_like_save_to_disk_arrow(ds):
        from huggingface_hub import snapshot_download
        log_for_0(
            f"Dataset at {path!r} looks like a save_to_disk-format HF repo; "
            f"re-downloading via snapshot_download + load_from_disk."
        )
        local_dir = snapshot_download(
            repo_id=path, repo_type="dataset", cache_dir=dataset_cache_dir,
        )
        ds = load_from_disk(local_dir)
        if isinstance(ds, DatasetDict):
            splits = list(ds.keys())
            if len(splits) != 1:
                raise ValueError(
                    f"Expected dataset at {path!r} to have a single split, got {splits}."
                )
            ds = ds[splits[0]]

    ds.set_format(type="numpy", columns=ds.column_names)
    return ds


def load_dataset(config, dataset_cache_dir=None):
    """Resolve config.data_path / config.eval_data_path into train/eval datasets."""
    log_for_0(f"Loading dataset from {config.data_path}...")
    train_dataset = load_dataset_split(config.data_path, dataset_cache_dir)
    log_for_0(f"Train size: {len(train_dataset)}")

    eval_dataset = None
    if config.eval_data_path:
        eval_dataset = load_dataset_split(config.eval_data_path, dataset_cache_dir)
        log_for_0(f"Eval size: {len(eval_dataset)}")
    else:
        log_for_0("No eval dataset")

    return train_dataset, eval_dataset
