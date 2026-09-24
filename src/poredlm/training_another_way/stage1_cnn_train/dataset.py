"""Streaming raw-signal dataset compatible with the public training layout."""

from __future__ import annotations

import csv
import gzip
import glob
import os
import random
from typing import Iterator, List, Union

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


class PoreSignalDataset(IterableDataset):
    """Read ``.npy`` signal memmaps with matching ``.csv.gz`` span indexes."""

    def __init__(
        self,
        shard_paths: Union[str, List[str]],
        chunk_size: int = 6000,
        memmap_dtype: str = "float32",
        buffer_size: int = 20000,
        shuffle_buffer: bool = True,
        repeat: bool = True,
        seed: int = 6198,
    ) -> None:
        super().__init__()
        paths = [shard_paths] if isinstance(shard_paths, str) else list(shard_paths)
        files: list[str] = []
        for path in paths:
            if os.path.isdir(path):
                files.extend(glob.glob(os.path.join(path, "**", "*.npy"), recursive=True))
            elif os.path.isfile(path) and path.endswith(".npy"):
                files.append(path)
        self.shards = []
        for signal_path in sorted(set(files)):
            index_path = signal_path.replace(".npy", ".csv.gz")
            if os.path.exists(index_path):
                self.shards.append((signal_path, index_path))
        if not self.shards:
            raise FileNotFoundError("No matching signal .npy/.csv.gz shard pairs found.")

        self.chunk_size = int(chunk_size)
        self.dtype = np.dtype(memmap_dtype)
        self.buffer_size = int(buffer_size)
        self.shuffle_buffer = bool(shuffle_buffer)
        self.repeat = bool(repeat)
        self.seed = int(seed)

    @staticmethod
    def _row_span(row: list[str]) -> tuple[int, int, int]:
        if len(row) < 5:
            raise ValueError("Signal index rows must contain at least five columns.")
        return int(row[0]), int(row[1]), int(row[4])

    def _iter_shard(self, signal_path: str, index_path: str) -> Iterator[dict[str, torch.Tensor]]:
        signal = np.memmap(signal_path, dtype=self.dtype, mode="r")
        with gzip.open(index_path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                start, end, sample_id = self._row_span(row)
                if start < 0 or end <= start or end > signal.shape[0]:
                    continue
                values = torch.from_numpy(np.asarray(signal[start:end]).copy()).float()
                if values.numel() < self.chunk_size:
                    values = torch.nn.functional.pad(values, (0, self.chunk_size - values.numel()))
                else:
                    values = values[: self.chunk_size]
                yield {"signal": values, "id": torch.tensor(sample_id, dtype=torch.long)}

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        worker_count = worker.num_workers if worker else 1
        rank = int(os.environ.get("RANK", "0"))
        world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
        global_worker_id = rank * worker_count + worker_id
        global_worker_count = world_size * worker_count
        epoch = 0
        while True:
            shards = list(self.shards)
            rng = random.Random(self.seed + epoch)
            if self.shuffle_buffer:
                rng.shuffle(shards)
            shards = [s for i, s in enumerate(shards) if i % global_worker_count == global_worker_id]
            buffer: list[dict[str, torch.Tensor]] = []
            for signal_path, index_path in shards:
                for sample in self._iter_shard(signal_path, index_path):
                    if self.shuffle_buffer and self.buffer_size > 0:
                        if len(buffer) < self.buffer_size:
                            buffer.append(sample)
                        else:
                            index = rng.randrange(len(buffer))
                            yield buffer[index]
                            buffer[index] = sample
                    else:
                        yield sample
            if buffer:
                rng.shuffle(buffer)
                yield from buffer
            epoch += 1
            if not self.repeat:
                break
