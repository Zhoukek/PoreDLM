"""Streaming loader for frozen continuous feature memmaps."""

from __future__ import annotations

import csv
import gzip
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


@dataclass(frozen=True)
class FeatureShard:
    data_path: Path
    index_path: Path


class ContinuousFeatureDataset(IterableDataset):
    def __init__(self, data_dir: str, feature_dim: int, max_length: int,
                 pattern: str = "features*.npy", shuffle_files: bool = True,
                 repeat: bool = True, seed: int = 42, dtype: str = "float16") -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.feature_dim = int(feature_dim)
        self.max_length = int(max_length)
        self.shuffle_files = bool(shuffle_files)
        self.repeat = bool(repeat)
        self.seed = int(seed)
        self.dtype = np.dtype(dtype)
        self.shards = self._discover(pattern)
        if not self.shards:
            raise FileNotFoundError(f"No continuous feature shards under {data_dir!r}.")

    def _discover(self, pattern: str) -> list[FeatureShard]:
        paths = sorted(self.data_dir.rglob(pattern))
        result = []
        for path in paths:
            index = path.with_name(path.name.removesuffix(path.suffix) + ".csv.gz")
            if index.exists(): result.append(FeatureShard(path, index))
        return result

    def _iter_shard(self, shard: FeatureShard, global_id: int, global_count: int):
        feature_stream = np.memmap(shard.data_path, dtype=self.dtype, mode="r")
        with gzip.open(shard.index_path, "rt", encoding="utf-8", newline="") as handle:
            for row_number, row in enumerate(csv.reader(handle)):
                if row_number % global_count != global_id or len(row) < 2: continue
                start, end = int(row[0]), int(row[1])
                if start < 0 or end <= start or end > feature_stream.shape[0]: continue
                flat = np.asarray(feature_stream[start:end]).copy()
                if flat.size % self.feature_dim != 0:
                    raise ValueError("Feature span is not divisible by feature_dim.")
                sample = torch.from_numpy(flat.reshape(-1, self.feature_dim)).float()
                sample = sample[: self.max_length]
                length = sample.shape[0]
                embeddings = torch.zeros(self.max_length, self.feature_dim)
                attention = torch.zeros(self.max_length, dtype=torch.long)
                embeddings[:length] = sample
                attention[:length] = 1
                sample_id = int(row[2]) if len(row) > 2 else row_number
                yield {"embeddings": embeddings, "attention_mask": attention,
                       "id": torch.tensor(sample_id, dtype=torch.long)}

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        worker_count = worker.num_workers if worker else 1
        rank = int(os.environ.get("RANK", "0"))
        world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
        global_id = rank * worker_count + worker_id
        global_count = world_size * worker_count
        epoch = 0
        while True:
            shards = list(self.shards)
            if self.shuffle_files: random.Random(self.seed + epoch).shuffle(shards)
            for shard in shards:
                yield from self._iter_shard(shard, global_id, global_count)
            epoch += 1
            if not self.repeat: break
