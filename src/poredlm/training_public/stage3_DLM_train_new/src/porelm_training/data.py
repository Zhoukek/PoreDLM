from __future__ import annotations

import csv
import gzip
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .config import ConditioningConfig, DataConfig


def expand_paths(paths: Sequence[str]) -> list[str]:
    result: list[str] = []
    for value in paths:
        path = Path(value).expanduser()
        if path.is_dir():
            matches = sorted(
                item
                for pattern in ("*.npy", "*.bin", "*.dat", "*.memmap")
                for item in path.glob(pattern)
                if item.is_file()
            )
            if not matches:
                raise ValueError(f"No token files found in {path}")
            result.extend(str(item) for item in matches)
        else:
            result.append(str(path))
    if not result:
        raise ValueError("At least one token data path is required")
    return result


class TokenDataset(Dataset[dict[str, Any]]):
    """Reads row-major arrays or indexed one-dimensional token streams."""

    def __init__(
        self,
        paths: Sequence[str],
        *,
        max_length: int,
        dtype: str,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
    ):
        self.paths = expand_paths(paths)
        self.max_length = max_length
        self.dtype = np.dtype(dtype)
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.arrays: list[np.ndarray] = []
        self.indices: list[np.ndarray | None] = []
        self.offsets: list[tuple[int, int]] = []
        offset = 0
        for path in self.paths:
            array = self._load_array(path)
            if array.ndim not in {1, 2}:
                raise ValueError(f"Expected a one- or two-dimensional token array, got {array.shape} in {path}")
            index = self._load_index(path, array) if array.ndim == 1 else None
            count = len(index) if index is not None else len(array)
            self.arrays.append(array)
            self.indices.append(index)
            self.offsets.append((offset, offset + count))
            offset += count
        self.length = offset

    def _load_array(self, value: str) -> np.ndarray:
        try:
            return np.load(value, mmap_mode="r")
        except Exception:
            path = Path(value)
            if path.stat().st_size % self.dtype.itemsize:
                raise ValueError(f"Raw token file size is incompatible with {self.dtype}: {path}")
            return np.memmap(path, dtype=self.dtype, mode="r")

    @staticmethod
    def _load_index(value: str, array: np.ndarray) -> np.ndarray:
        index_path = Path(value).with_suffix(".csv.gz")
        if not index_path.exists():
            raise ValueError(f"One-dimensional token stream requires an index file: {index_path}")
        spans: list[tuple[int, int]] = []
        with gzip.open(index_path, "rt", encoding="utf-8", newline="") as handle:
            for row_number, row in enumerate(csv.reader(handle), start=1):
                if len(row) < 2:
                    raise ValueError(f"Malformed index row {row_number} in {index_path}")
                start, end = int(row[0]), int(row[1])
                if start < 0 or end <= start or end > len(array):
                    raise ValueError(f"Invalid span [{start}, {end}) in {index_path}")
                spans.append((start, end))
        return np.asarray(spans, dtype=np.int64).reshape(-1, 2)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        for array_number, (start, end) in enumerate(self.offsets):
            if start <= index < end:
                local_index = index - start
                break
        else:
            raise IndexError(index)
        array = self.arrays[array_number]
        spans = self.indices[array_number]
        if spans is None:
            tokens = np.asarray(array[local_index], dtype=np.int64)
        else:
            item_start, item_end = spans[local_index]
            tokens = np.asarray(array[item_start:item_end], dtype=np.int64)
        tokens = tokens.reshape(-1)
        while len(tokens) and int(tokens[-1]) == self.pad_token_id:
            tokens = tokens[:-1]
        if len(tokens) < 2 or int(tokens[0]) != self.bos_token_id or int(tokens[-1]) != self.eos_token_id:
            raise ValueError(f"Sample {index} must begin with BOS and end with EOS")
        content = tokens[1:-1][: self.max_length - 2]
        normalized = np.concatenate(
            [np.asarray([self.bos_token_id]), content, np.asarray([self.eos_token_id])]
        ).astype(np.int64)
        return {"input_ids": normalized, "index": index}


class ConditionMaskSampler:
    TASK_IDS = {"unconditional": 0, "prefix_suffix": 1, "single_span": 2, "multi_span": 3}

    def __init__(self, config: ConditioningConfig):
        self.config = config
        weights = torch.tensor(
            [config.prefix_suffix_weight, config.single_span_weight, config.multi_span_weight],
            dtype=torch.float,
        )
        if (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("Conditional task weights must be non-negative and have a positive sum")
        self.weights = weights

    @staticmethod
    def _place_span(mask: torch.Tensor, content_length: int, span_length: int) -> bool:
        span_length = min(span_length, content_length)
        candidates = [
            start
            for start in range(content_length - span_length + 1)
            if bool(mask[1 + start : 1 + start + span_length].all())
        ]
        if not candidates:
            return False
        start = candidates[int(torch.randint(len(candidates), ()).item())]
        mask[1 + start : 1 + start + span_length] = False
        return True

    def __call__(self, valid_length: int) -> tuple[torch.Tensor, int]:
        mask = torch.zeros(valid_length, dtype=torch.bool)
        if valid_length:
            mask[0] = True
        if valid_length > 1:
            mask[-1] = True
        content_length = max(valid_length - 2, 0)
        if content_length == 0 or self.config.mode == "unconditional":
            return mask, self.TASK_IDS["unconditional"]
        if self.config.mode == "mixed" and torch.rand(()) < self.config.unconditional_probability:
            return mask, self.TASK_IDS["unconditional"]

        pattern = self.config.pattern
        if pattern == "mixed":
            pattern = ("prefix_suffix", "single_span", "multi_span")[
                int(torch.multinomial(self.weights, 1).item())
            ]
        mask[1:-1] = True
        min_span = min(self.config.min_span_length, content_length)
        max_span = min(self.config.max_span_length, content_length)
        if pattern == "single_span":
            length = int(torch.randint(min_span, max_span + 1, ()).item())
            self._place_span(mask, content_length, length)
        elif pattern == "prefix_suffix":
            length = int(torch.randint(min_span, max_span + 1, ()).item())
            left = min(length, content_length // 2)
            right = min(length, content_length - left)
            mask[1 : 1 + left] = False
            mask[1 + content_length - right : 1 + content_length] = False
        elif pattern == "multi_span":
            count = int(torch.randint(self.config.multi_min_spans, self.config.multi_max_spans + 1, ()).item())
            for _ in range(count):
                length = int(torch.randint(min_span, max_span + 1, ()).item())
                if not self._place_span(mask, content_length, length):
                    break
        else:
            raise ValueError(f"Unknown condition pattern: {pattern}")
        return mask, self.TASK_IDS[pattern]


class BatchCollator:
    def __init__(self, max_length: int, pad_token_id: int, conditioning: ConditioningConfig):
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.condition_sampler = ConditionMaskSampler(conditioning)

    def __call__(self, samples: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = torch.full((len(samples), self.max_length), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        condition_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        task_ids = torch.zeros(len(samples), dtype=torch.long)
        for row, sample in enumerate(samples):
            tokens = torch.as_tensor(sample["input_ids"][: self.max_length], dtype=torch.long)
            length = len(tokens)
            input_ids[row, :length] = tokens
            attention_mask[row, :length] = True
            condition, task_id = self.condition_sampler(length)
            condition_mask[row, :length] = condition
            task_ids[row] = task_id

        is_condition = condition_mask
        is_valid = attention_mask
        encoder_attention_mask = (
            (is_condition[:, :, None] & is_condition[:, None, :])
            | (~is_condition[:, :, None] & is_valid[:, None, :])
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "encoder_attention_mask": encoder_attention_mask,
            "condition_mask": condition_mask,
            "condition_task_ids": task_ids,
            "index": torch.tensor([sample["index"] for sample in samples]),
        }


def build_dataloader(
    paths: Sequence[str],
    data_config: DataConfig,
    conditioning_config: ConditioningConfig,
    *,
    max_length: int,
    batch_size: int,
    distributed: bool,
    shuffle: bool,
    drop_last: bool,
) -> tuple[DataLoader, DistributedSampler | None]:
    dataset = TokenDataset(
        paths,
        max_length=max_length,
        dtype=data_config.dtype,
        pad_token_id=data_config.pad_token_id,
        bos_token_id=data_config.bos_token_id,
        eos_token_id=data_config.eos_token_id,
    )
    sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        drop_last=drop_last,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        collate_fn=BatchCollator(max_length, data_config.pad_token_id, conditioning_config),
    )
    return loader, sampler
