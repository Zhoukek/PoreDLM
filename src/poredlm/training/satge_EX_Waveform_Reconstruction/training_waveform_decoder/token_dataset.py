"""Streaming reader and collator for Stage 2/3 token shards."""

from __future__ import annotations

import csv
import gzip
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


@dataclass(frozen=True)
class TokenShard:
    data: Path
    index: Path


def resolve_shards(paths: Sequence[str], pattern: str = "*.npy") -> list[TokenShard]:
    shards: list[TokenShard] = []
    for raw_path in paths:
        path = Path(raw_path)
        candidates = [path] if path.is_file() else path.rglob(pattern)
        for data_path in candidates:
            index_path = data_path.with_suffix(".csv.gz")
            if index_path.is_file():
                shards.append(TokenShard(data_path, index_path))
    return sorted(set(shards), key=lambda item: str(item.data))


class TokenSequenceDataset(IterableDataset):
    """Read headerless token memmaps indexed by either 3- or 6-column CSV rows."""

    def __init__(
        self,
        paths: Sequence[str],
        *,
        pattern: str = "*.npy",
        dtype: str = "uint32",
        shuffle_files: bool = False,
        repeat: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.shards = resolve_shards(paths, pattern)
        if not self.shards:
            raise FileNotFoundError(f"No .npy/.csv.gz token shard pairs found under {list(paths)}.")
        self.dtype = np.dtype(dtype)
        self.shuffle_files = bool(shuffle_files)
        self.repeat = bool(repeat)
        self.seed = int(seed)

    def __iter__(self) -> Iterator[dict[str, object]]:
        rank = int(os.environ.get("RANK", "0"))
        world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        workers = worker.num_workers if worker else 1
        global_worker = rank * workers + worker_id
        global_workers = world_size * workers
        epoch = 0

        while True:
            shards = list(self.shards)
            if self.shuffle_files:
                random.Random(self.seed + epoch).shuffle(shards)
            row_index = 0
            for shard in shards:
                token_stream = np.memmap(shard.data, dtype=self.dtype, mode="r")
                with gzip.open(shard.index, "rt", encoding="utf-8", newline="") as handle:
                    for row in csv.reader(handle):
                        take = row_index % global_workers == global_worker
                        row_index += 1
                        if not take or len(row) < 2:
                            continue
                        start, end = int(row[0]), int(row[1])
                        if start < 0 or end <= start or end > token_stream.size:
                            raise ValueError(
                                f"Bad span [{start}, {end}) in {shard.index}; "
                                f"stream contains {token_stream.size} tokens."
                            )
                        tokens = np.asarray(token_stream[start:end]).astype(np.int64, copy=True)
                        sample_id = row[4] if len(row) >= 5 else (row[2] if len(row) >= 3 else "")
                        yield {"tokens": torch.from_numpy(tokens), "id": sample_id}
            epoch += 1
            if not self.repeat:
                return


class WaveformTokenCollator:
    def __init__(
        self,
        *,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        token_offset: int,
        codebook_size: int,
        max_length: int,
        strict_boundaries: bool = True,
    ) -> None:
        self.pad_token_id = int(pad_token_id)
        self.bos_token_id = int(bos_token_id)
        self.eos_token_id = int(eos_token_id)
        self.token_offset = int(token_offset)
        self.codebook_size = int(codebook_size)
        self.max_length = int(max_length)
        self.strict_boundaries = bool(strict_boundaries)

    def __call__(self, samples: list[dict[str, object]]) -> dict[str, object]:
        dlm_sequences: list[torch.Tensor] = []
        codec_sequences: list[torch.Tensor] = []
        ids: list[str] = []
        max_content_length = self.max_length - 2
        for sample in samples:
            raw = torch.as_tensor(sample["tokens"], dtype=torch.long).flatten()
            if raw.numel() < 2:
                raise ValueError("A token sample must contain at least BOS and EOS.")
            if self.strict_boundaries and (
                int(raw[0]) != self.bos_token_id or int(raw[-1]) != self.eos_token_id
            ):
                raise ValueError(
                    f"Expected BOS/EOS={self.bos_token_id}/{self.eos_token_id}, "
                    f"got {int(raw[0])}/{int(raw[-1])}."
                )
            content = raw[1:-1][:max_content_length]
            if content.numel() == 0:
                raise ValueError("A token sample must contain at least one content token.")
            codec_ids = content - self.token_offset
            if codec_ids.numel() and (
                int(codec_ids.min()) < 0 or int(codec_ids.max()) >= self.codebook_size
            ):
                raise ValueError(
                    "Content token outside tokenizer codebook after removing offset: "
                    f"range=[{int(codec_ids.min())}, {int(codec_ids.max())}], "
                    f"codebook_size={self.codebook_size}."
                )
            dlm_sequences.append(
                torch.cat(
                    [
                        raw.new_tensor([self.bos_token_id]),
                        content,
                        raw.new_tensor([self.eos_token_id]),
                    ]
                )
            )
            codec_sequences.append(codec_ids)
            ids.append(str(sample.get("id", "")))

        dlm_length = max(item.numel() for item in dlm_sequences)
        content_length = max(item.numel() for item in codec_sequences)
        input_ids = torch.full((len(samples), dlm_length), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        codec_token_ids = torch.zeros((len(samples), content_length), dtype=torch.long)
        content_mask = torch.zeros((len(samples), content_length), dtype=torch.bool)
        for index, (dlm_ids, codec_ids) in enumerate(zip(dlm_sequences, codec_sequences)):
            input_ids[index, : dlm_ids.numel()] = dlm_ids
            attention_mask[index, : dlm_ids.numel()] = 1
            codec_token_ids[index, : codec_ids.numel()] = codec_ids
            content_mask[index, : codec_ids.numel()] = True
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "codec_token_ids": codec_token_ids,
            "content_mask": content_mask,
            "ids": ids,
        }
