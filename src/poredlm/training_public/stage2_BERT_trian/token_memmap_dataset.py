"""Dataset utilities for Stage 2 BERT training on infer.py token outputs."""

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
class TokenShard:
    token_path: Path
    index_path: Path


class TokenMemmapIterableDataset(IterableDataset):
    """Stream raw token memmap shards plus ``csv.gz`` span metadata.

    The expected format is exactly what ``token_dataset/infer.py`` writes:
    ``<prefix>.npy`` is a raw ``tofile`` token stream, and ``<prefix>.csv.gz``
    stores rows ``start_pos,end_pos,read_id`` without a header.
    """

    def __init__(
        self,
        data_dir: str | os.PathLike[str],
        pattern: str = "*.npy",
        token_dtype: str = "uint32",
        max_length: int = 1280,
        pad_token_id: int = 0,
        shuffle_files: bool = False,
        seed: int = 42,
        repeat: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.pattern = pattern
        self.token_dtype = np.dtype(token_dtype)
        self.max_length = int(max_length)
        self.pad_token_id = int(pad_token_id)
        self.shuffle_files = bool(shuffle_files)
        self.seed = int(seed)
        self.repeat = bool(repeat)
        self.shards = self._discover_shards()
        if not self.shards:
            raise FileNotFoundError(f"No token shards matching {pattern!r} under {self.data_dir}.")

    def _discover_shards(self) -> list[TokenShard]:
        shards: list[TokenShard] = []
        for token_path in sorted(self.data_dir.rglob(self.pattern)):
            index_path = token_path.with_suffix(".csv.gz")
            if index_path.exists():
                shards.append(TokenShard(token_path=token_path, index_path=index_path))
        return shards

    def _iter_rows(
        self,
        shard: TokenShard,
        global_worker_id: int,
        global_worker_count: int,
    ) -> Iterator[dict[str, torch.Tensor | str]]:
        token_stream = np.memmap(shard.token_path, dtype=self.token_dtype, mode="r")
        with gzip.open(shard.index_path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            for row_number, row in enumerate(reader, start=1):
                if (row_number - 1) % global_worker_count != global_worker_id:
                    continue
                if len(row) < 2:
                    continue
                start = int(row[0])
                end = int(row[1])
                if start < 0 or end < start or end > token_stream.shape[0]:
                    raise ValueError(
                        f"Invalid span in {shard.index_path} row {row_number}: "
                        f"start={start}, end={end}, token_count={token_stream.shape[0]}."
                    )
                sample = torch.as_tensor(np.asarray(token_stream[start:end]), dtype=torch.long)
                if sample.numel() > self.max_length:
                    sample = sample[: self.max_length]

                input_ids = torch.full((self.max_length,), self.pad_token_id, dtype=torch.long)
                attention_mask = torch.zeros((self.max_length,), dtype=torch.long)
                length = int(sample.numel())
                if length:
                    input_ids[:length] = sample
                    attention_mask[:length] = 1

                yield {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "read_id": row[2] if len(row) > 2 else "",
                }

    def __iter__(self) -> Iterator[dict[str, torch.Tensor | str]]:
        rank = int(os.environ.get("RANK", "0"))
        world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        global_worker_count = world_size * num_workers
        global_worker_id = rank * num_workers + worker_id

        epoch = 0
        while True:
            shards = list(self.shards)
            if self.shuffle_files:
                rng = random.Random(self.seed + epoch)
                rng.shuffle(shards)

            for shard in shards:
                yield from self._iter_rows(shard, global_worker_id, global_worker_count)

            epoch += 1
            if not self.repeat:
                break
