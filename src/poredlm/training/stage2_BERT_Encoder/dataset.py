"""Datasets and collators for BERT encoder pretraining."""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Stage2Batch:
    """Token-id batch consumed by the Stage 2 BERT model."""

    input_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None


class Stage2TokenShardDataset(Dataset):
    """Memmap shard dataset for VQ token ids.

    Expected layout:

    ```text
    stage2_dir/
      shards.json
      shard_000.npy
      shard_001.npy
    ```

    `shards.json` should contain `{"shards": [{"path": "...", "num_samples": N}, ...]}`.
    Each npy shard should have shape `[num_samples, seq_len]` and integer dtype.
    """

    def __init__(self, shards_dir: str, max_cache_size: int = 32) -> None:
        self.shards_dir = shards_dir
        self.max_cache_size = max_cache_size
        meta_path = os.path.join(shards_dir, "shards.json")
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)

        self.shard_info = meta["shards"]
        self.offsets = [0]
        for info in self.shard_info:
            self.offsets.append(self.offsets[-1] + int(info["num_samples"]))

        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return self.offsets[-1]

    def _get_memmap(self, shard_path: str) -> np.ndarray:
        if shard_path in self._cache:
            self._cache.move_to_end(shard_path)
            return self._cache[shard_path]

        arr = np.load(shard_path, mmap_mode="r")
        if len(self._cache) >= self.max_cache_size:
            self._cache.popitem(last=False)
        self._cache[shard_path] = arr
        return arr

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range [0, {len(self)})")

        for shard_id in range(len(self.offsets) - 1):
            if self.offsets[shard_id] <= index < self.offsets[shard_id + 1]:
                return shard_id, index - self.offsets[shard_id]

        raise IndexError(f"Index {index} was not found in any shard.")

    def __getitem__(self, index: int) -> torch.Tensor:
        shard_id, local_index = self._locate(index)
        shard_path = os.path.join(self.shards_dir, self.shard_info[shard_id]["path"])
        sample = self._get_memmap(shard_path)[local_index]

        if not np.issubdtype(sample.dtype, np.integer):
            raise TypeError(
                "Stage 2 BERT encoder currently expects integer VQ token ids. "
                f"Got dtype {sample.dtype} from {shard_path}."
            )
        return torch.from_numpy(np.asarray(sample, dtype=np.int64))


class Stage2Collator:
    """Pad token-id samples into one batch."""

    def __init__(self, pad_token_id: int = 0) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, samples: list[torch.Tensor]) -> Stage2Batch:
        max_len = max(sample.shape[0] for sample in samples)
        input_ids = torch.full(
            (len(samples), max_len),
            fill_value=self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((len(samples), max_len), dtype=torch.long)
        for i, sample in enumerate(samples):
            if sample.ndim != 1:
                raise ValueError(f"Token samples must have shape [seq_len]. Got {tuple(sample.shape)}.")
            length = sample.shape[0]
            input_ids[i, :length] = sample.long()
            attention_mask[i, :length] = 1
        return Stage2Batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
