from __future__ import annotations

import csv
import gzip
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import DataConfig, TrainConfig
from .data.iterable_dataset import IterableDataset
from .exceptions import OLMoConfigurationError
from .torch_util import barrier, get_global_rank, get_world_size

log = logging.getLogger(__name__)


def _expand_token_paths(paths: Sequence[str]) -> List[str]:
    expanded: List[str] = []
    for path in paths:
        path_obj = Path(path).expanduser()
        if path_obj.is_dir():
            matched = sorted(
                candidate
                for pattern in ("*.npy", "*.bin", "*.dat", "*.memmap")
                for candidate in path_obj.glob(pattern)
                if candidate.is_file()
            )
            if not matched:
                raise OLMoConfigurationError(
                    f"DLM data directory {str(path_obj)!r} does not contain any .npy/.bin/.dat/.memmap token files"
                )
            expanded.extend(str(candidate) for candidate in matched)
        else:
            expanded.append(str(path_obj))

    if not expanded:
        raise OLMoConfigurationError("DLM data requires at least one file in cfg.data.paths")
    return expanded


def _build_self_attn_cond_masks(is_cond: np.ndarray, is_valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    encoder_attention_mask = (
        (is_cond[:, :, None] & is_cond[:, None, :]) | (~is_cond[:, :, None] & is_valid[:, None, :])
    ).astype(np.float32)
    attention_mask = is_valid.astype(np.float32)
    cond_seq_mask = is_cond.astype(np.float32)
    return encoder_attention_mask, attention_mask, cond_seq_mask


def _pad_and_truncate(ids_list: Sequence[np.ndarray], target_len: int, pad_token_id: int) -> Tuple[np.ndarray, np.ndarray]:
    padded: List[np.ndarray] = []
    lengths: List[int] = []
    for ids in ids_list:
        ids = np.asarray(ids)
        orig_len = min(len(ids), target_len)
        ids = ids[:target_len]
        if orig_len < target_len:
            pad = np.full(target_len - orig_len, pad_token_id, dtype=ids.dtype)
            ids = np.concatenate([ids, pad])
        padded.append(ids)
        lengths.append(orig_len)
    return np.stack(padded), np.asarray(lengths, dtype=np.int64)


class DLMTokensDataset(Dataset[Dict[str, Any]]):
    """
    Map-style token dataset for stage-3 DLM training.

    This intentionally lives outside the original ELF port. It reads OLMo's
    ``cfg.data.paths`` and returns item dictionaries compatible with the ELF
    collate logic: ``input_ids`` plus an optional ``condition_input_ids``.
    """

    def __init__(
        self,
        paths: Sequence[str],
        *,
        chunk_size: int,
        memmap_dtype: type[np.generic],
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        include_instance_metadata: bool = False,
        metadata: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        if not paths:
            raise OLMoConfigurationError("DLM data requires at least one path in cfg.data.paths")
        if metadata is not None and len(metadata) != len(paths):
            raise OLMoConfigurationError(
                f"DLM metadata has {len(metadata)} entries for {len(paths)} configured paths"
            )
        self.paths = []
        self.metadata: List[Dict[str, Any]] = []
        for path_index, path in enumerate(paths):
            expanded = _expand_token_paths([path])
            self.paths.extend(expanded)
            item_metadata = metadata[path_index] if metadata is not None else {}
            self.metadata.extend(dict(item_metadata) for _ in expanded)
        self.chunk_size = chunk_size
        self.memmap_dtype = memmap_dtype
        self.pad_token_id = pad_token_id
        self.bos_token_id = int(bos_token_id)
        self.eos_token_id = int(eos_token_id)
        self.include_instance_metadata = include_instance_metadata
        self._arrays: List[np.ndarray] = []
        self._indices: List[Optional[np.ndarray]] = []
        self._offsets: List[Tuple[int, int]] = []

        start = 0
        for path in self.paths:
            array = self._load_array(path)
            if array.ndim == 0:
                raise OLMoConfigurationError(f"DLM data file {path!r} is scalar; expected 1D or 2D token IDs")
            if array.ndim > 2:
                raise OLMoConfigurationError(f"DLM data file {path!r} has shape {array.shape}; expected 1D or 2D")
            index = self._load_index(path, array)
            num_instances = int(index.shape[0]) if index is not None else int(array.shape[0])
            if num_instances == 0:
                raise OLMoConfigurationError(f"DLM data file {path!r} has no indexed training instances")
            self._arrays.append(array)
            self._indices.append(index)
            self._offsets.append((start, start + num_instances))
            start += num_instances

        self._length = start
        log.info("Built DLM token dataset with %d instances from %d file(s)", self._length, len(self.paths))

    def _load_array(self, path: str) -> np.ndarray:
        try:
            return np.load(path, mmap_mode="r")
        except Exception as exc:
            file_size = Path(path).stat().st_size
            item_size = np.dtype(self.memmap_dtype).itemsize
            token_count = file_size // item_size
            if file_size % item_size != 0:
                raise OLMoConfigurationError(
                    f"Raw token file {path!r} has {file_size} bytes, which is not divisible "
                    f"by dtype item size {item_size}"
                ) from exc
            if token_count == 0:
                raise OLMoConfigurationError(
                    f"Could not load {path!r} with np.load ({exc!r}), and the raw file is too small"
                ) from exc
            log.warning(
                "Could not load %s with np.load; treating it as a raw %s token memmap",
                path,
                np.dtype(self.memmap_dtype).name,
            )
            return np.memmap(path, dtype=self.memmap_dtype, mode="r", shape=(token_count,))

    def _load_index(self, path: str, array: np.ndarray) -> Optional[np.ndarray]:
        if array.ndim == 2:
            return None
        index_path = Path(path).with_suffix(".csv.gz")
        if not index_path.is_file():
            raise OLMoConfigurationError(
                f"One-dimensional token stream {path!r} requires sample-boundary index "
                f"{str(index_path)!r}; refusing to split it into arbitrary fixed chunks."
            )
        spans: List[Tuple[int, int]] = []
        with gzip.open(index_path, "rt", encoding="utf-8", newline="") as handle:
            for row_number, row in enumerate(csv.reader(handle), start=1):
                if len(row) < 2:
                    raise OLMoConfigurationError(
                        f"Malformed index row {row_number} in {str(index_path)!r}: {row!r}"
                    )
                try:
                    item_start, item_end = int(row[0]), int(row[1])
                except ValueError as exc:
                    raise OLMoConfigurationError(
                        f"Non-integer span at row {row_number} in {str(index_path)!r}: {row[:2]!r}"
                    ) from exc
                if item_start < 0 or item_end <= item_start or item_end > array.shape[0]:
                    raise OLMoConfigurationError(
                        f"Invalid span [{item_start}, {item_end}) at row {row_number} in "
                        f"{str(index_path)!r}; token stream length is {array.shape[0]}"
                    )
                spans.append((item_start, item_end))
        return np.asarray(spans, dtype=np.int64).reshape(-1, 2)

    def _normalize_instance(self, ids: np.ndarray, path: str, index: int) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        # A genuine 2D dataset may already be right-padded row by row.
        valid_end = ids.size
        while valid_end > 0 and int(ids[valid_end - 1]) == self.pad_token_id:
            valid_end -= 1
        ids = ids[:valid_end]
        if ids.size < 2:
            raise OLMoConfigurationError(
                f"DLM instance {index} in {path!r} is too short for BOS/EOS: length={ids.size}"
            )
        if int(ids[0]) != self.bos_token_id or int(ids[-1]) != self.eos_token_id:
            raise OLMoConfigurationError(
                f"DLM instance {index} in {path!r} must have BOS/EOS="
                f"{self.bos_token_id}/{self.eos_token_id}, got {int(ids[0])}/{int(ids[-1])}"
            )
        content = ids[1:-1][: max(0, self.chunk_size - 2)]
        return np.concatenate(
            [
                np.asarray([self.bos_token_id], dtype=np.int64),
                content,
                np.asarray([self.eos_token_id], dtype=np.int64),
            ]
        )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> Dict[str, Any]:
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        array_index = 0
        local_index = index
        for i, (start, end) in enumerate(self._offsets):
            if start <= index < end:
                array_index = i
                local_index = index - start
                break

        array = self._arrays[array_index]
        item_index = self._indices[array_index]
        if item_index is not None:
            item_start, item_end = item_index[local_index]
            input_ids = np.asarray(array[int(item_start) : int(item_end)], dtype=np.int64)
        else:
            input_ids = np.asarray(array[local_index], dtype=np.int64)
        input_ids = self._normalize_instance(input_ids, self.paths[array_index], local_index)

        out: Dict[str, Any] = {"input_ids": input_ids}
        if self.include_instance_metadata:
            out["metadata"] = {
                **self.metadata[array_index],
                "path": self.paths[array_index],
                "local_index": local_index,
            }
        return out


class DLMDataCollator:
    def __init__(
        self,
        *,
        max_seq_length: int,
        pad_token_id: int,
        max_input_seq_length: Optional[int] = None,
        conditioning_mode: str = "unconditional",
        unconditional_prob: float = 0.1,
        condition_pattern: str = "mixed",
        condition_min_mask_ratio: float = 0.1,
        condition_max_mask_ratio: float = 0.5,
        condition_min_span_length: int = 30,
        condition_max_span_length: int = 50,
        condition_multi_min_spans: int = 4,
        condition_multi_max_spans: int = 8,
        condition_prefix_suffix_weight: float = 1.0,
        condition_single_span_weight: float = 1.0,
        condition_multi_span_weight: float = 1.0,
    ) -> None:
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id
        self.max_input_seq_length = max_input_seq_length
        self.conditioning_mode = str(conditioning_mode).lower()
        self.unconditional_prob = float(unconditional_prob)
        self.condition_pattern = str(condition_pattern).lower()
        self.condition_min_mask_ratio = float(condition_min_mask_ratio)
        self.condition_max_mask_ratio = float(condition_max_mask_ratio)
        self.condition_min_span_length = int(condition_min_span_length)
        self.condition_max_span_length = int(condition_max_span_length)
        self.condition_multi_min_spans = int(condition_multi_min_spans)
        self.condition_multi_max_spans = int(condition_multi_max_spans)
        self.condition_pattern_weights = torch.tensor(
            [
                float(condition_prefix_suffix_weight),
                float(condition_single_span_weight),
                float(condition_multi_span_weight),
            ],
            dtype=torch.float32,
        )
        if self.conditioning_mode not in {"unconditional", "conditional", "mixed"}:
            raise ValueError("conditioning_mode must be unconditional, conditional, or mixed")
        if self.condition_pattern not in {
            "prefix", "infill", "prefix_suffix", "single_span", "multi_span", "mixed"
        }:
            raise ValueError(
                "condition_pattern must be prefix, infill, prefix_suffix, single_span, multi_span, or mixed"
            )
        if not 0.0 <= self.unconditional_prob <= 1.0:
            raise ValueError("unconditional_prob must be in [0, 1]")
        if not 0.0 < self.condition_min_mask_ratio <= self.condition_max_mask_ratio <= 1.0:
            raise ValueError("Require 0 < condition_min_mask_ratio <= condition_max_mask_ratio <= 1")
        if not 1 <= self.condition_min_span_length <= self.condition_max_span_length:
            raise ValueError("Require 1 <= condition_min_span_length <= condition_max_span_length")
        if not 1 <= self.condition_multi_min_spans <= self.condition_multi_max_spans:
            raise ValueError("Require 1 <= condition_multi_min_spans <= condition_multi_max_spans")
        if bool((self.condition_pattern_weights < 0).any()) or self.condition_pattern_weights.sum() <= 0:
            raise ValueError("Conditional pattern weights must be non-negative and sum to > 0")

    def _randint(self, low: int, high: int) -> int:
        """Sample from the inclusive integer interval [low, high]."""
        return low if high <= low else int(torch.randint(low, high + 1, ()).item())

    def _span_length(self, available: int) -> int:
        return min(available, self._randint(self.condition_min_span_length, self.condition_max_span_length))

    def _mask_single_span(self, mask: np.ndarray, content_length: int) -> None:
        length = self._span_length(content_length)
        start = self._randint(0, content_length - length)
        mask[1 + start : 1 + start + length] = False

    def _mask_prefix_suffix(self, mask: np.ndarray, content_length: int) -> None:
        # Predict one span at each content boundary. For very short sequences,
        # fall back to one span rather than allowing the two targets to overlap.
        left = self._span_length(content_length)
        remaining = content_length - left
        if remaining < self.condition_min_span_length:
            self._mask_single_span(mask, content_length)
            return
        right = self._span_length(remaining)
        mask[1 : 1 + left] = False
        mask[1 + content_length - right : 1 + content_length] = False

    def _mask_multiple_spans(self, mask: np.ndarray, content_length: int) -> None:
        requested = self._randint(self.condition_multi_min_spans, self.condition_multi_max_spans)
        max_possible = max(1, content_length // self.condition_min_span_length)
        num_spans = min(requested, max_possible)
        lengths = [self._span_length(content_length) for _ in range(num_spans)]
        while sum(lengths) > content_length:
            largest = max(range(num_spans), key=lengths.__getitem__)
            if lengths[largest] > self.condition_min_span_length:
                lengths[largest] -= 1
            else:
                lengths.pop()
                num_spans -= 1
        # Randomly distribute all unused tokens over the n+1 gaps. Interior
        # gaps receive at least one token whenever space permits, preventing
        # adjacent spans from silently merging into a single large target.
        free = content_length - sum(lengths)
        gaps = [0] * (num_spans + 1)
        if free >= num_spans - 1:
            for gap_index in range(1, num_spans):
                gaps[gap_index] = 1
            free -= num_spans - 1
        for _ in range(free):
            gaps[self._randint(0, num_spans)] += 1
        cursor = gaps[0]
        for span_index, length in enumerate(lengths):
            mask[1 + cursor : 1 + cursor + length] = False
            cursor += length + gaps[span_index + 1]

    def _sample_condition_mask(self, valid_length: int) -> Tuple[np.ndarray, int]:
        mask = np.zeros(self.max_seq_length, dtype=np.bool_)
        if valid_length >= 2:
            # Sequence boundaries are always known. "Unconditional" below
            # means unconditional generation of all codec content tokens.
            mask[0] = True
            mask[valid_length - 1] = True
        if self.conditioning_mode == "unconditional" or valid_length < 3:
            return mask, 0
        if self.conditioning_mode == "mixed" and torch.rand(()).item() < self.unconditional_prob:
            return mask, 0

        # The sequence layout is [BOS, content..., EOS]. Select a target span in
        # content-token coordinates; known positions become the fixed condition.
        content_length = valid_length - 2
        ratio = self.condition_min_mask_ratio + torch.rand(()).item() * (
            self.condition_max_mask_ratio - self.condition_min_mask_ratio
        )
        target_length = min(content_length, max(1, int(round(content_length * ratio))))
        pattern = self.condition_pattern
        if pattern == "mixed":
            pattern_index = int(torch.multinomial(self.condition_pattern_weights, 1).item())
            pattern = ("prefix_suffix", "single_span", "multi_span")[pattern_index]

        mask[:valid_length] = True
        if pattern == "prefix":
            # Generate only a content suffix. BOS and EOS stay fixed because
            # waveform reconstruction uses a known token length and only codec
            # tokens are valid generation targets.
            target_start = 1 + content_length - target_length
            mask[target_start : valid_length - 1] = False
        elif pattern == "infill":
            max_start = content_length - target_length
            content_start = int(torch.randint(max_start + 1, ()).item()) if max_start else 0
            target_start = 1 + content_start
            mask[target_start : target_start + target_length] = False
        elif pattern == "prefix_suffix":
            self._mask_prefix_suffix(mask, content_length)
        elif pattern == "single_span":
            self._mask_single_span(mask, content_length)
        else:
            self._mask_multiple_spans(mask, content_length)
        task_id = {
            "prefix": 1,
            "infill": 2,
            "prefix_suffix": 1,
            "single_span": 2,
            "multi_span": 3,
        }[pattern]
        return mask, task_id

    def __call__(self, batch_list: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        seq_list: List[np.ndarray] = []
        cond_lens: List[int] = []
        for item in batch_list:
            if "condition_input_ids" in item:
                max_input_len = self.max_input_seq_length
                cond = np.asarray(item["condition_input_ids"], dtype=np.int64)
                if max_input_len is not None:
                    cond = cond[:max_input_len]
                inp = np.asarray(item["input_ids"], dtype=np.int64)
                seq_list.append(np.concatenate([cond, inp]))
                cond_lens.append(len(cond))
            else:
                seq_list.append(np.asarray(item["input_ids"], dtype=np.int64))
                cond_lens.append(0)

        ids, total_lens = _pad_and_truncate(seq_list, self.max_seq_length, self.pad_token_id)
        pos = np.arange(self.max_seq_length)[None, :]
        is_valid = (pos < total_lens[:, None]) & (ids != self.pad_token_id)
        condition_task_ids: List[int]
        if any(cond_lens):
            # Backward-compatible path for explicitly concatenated conditions.
            is_cond = pos < np.asarray(cond_lens, dtype=np.int64)[:, None]
            condition_task_ids = [-1] * len(batch_list)
        else:
            sampled = [self._sample_condition_mask(int(length)) for length in total_lens]
            is_cond = np.stack([item[0] for item in sampled])
            condition_task_ids = [item[1] for item in sampled]
        is_cond &= is_valid
        encoder_attn, attn, cond_seq_mask = _build_self_attn_cond_masks(is_cond, is_valid)

        result: Dict[str, Any] = {
            "input_ids": torch.from_numpy(ids).long(),
            "encoder_attention_mask": torch.from_numpy(encoder_attn),
            "attention_mask": torch.from_numpy(attn),
            "cond_seq_mask": torch.from_numpy(cond_seq_mask),
            # 0=content-unconditional, 1=prefix+suffix, 2=single span,
            # 3=multi span, -1=legacy explicit condition_input_ids.
            "condition_task_ids": torch.tensor(condition_task_ids, dtype=torch.long),
        }
        if "index" in batch_list[0]:
            result["index"] = torch.tensor([int(item["index"]) for item in batch_list], dtype=torch.long)
        if "metadata" in batch_list[0]:
            result["metadata"] = [item["metadata"] for item in batch_list]
        return result


def _resolve_dlm_max_length(train_config: TrainConfig) -> int:
    dlm_max_length = getattr(train_config.dlm, "max_length", None)
    return int(dlm_max_length or train_config.model.max_sequence_length)


def build_train_dlm_dataloader(
    train_config: TrainConfig,
    *,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    fs_local_rank: Optional[int] = None,
    include_instance_metadata: bool = False,
) -> DataLoader:
    assert train_config.device_train_batch_size is not None
    if train_config.data.paths is None:
        raise OLMoConfigurationError("DLM train dataloader currently reads cfg.data.paths")
    if train_config.data.datasets is not None:
        raise OLMoConfigurationError("DLM train dataloader currently expects cfg.data.paths, not cfg.data.datasets")

    max_length = _resolve_dlm_max_length(train_config)
    dataset = DLMTokensDataset(
        train_config.data.paths,
        chunk_size=max_length,
        memmap_dtype=train_config.data.effective_memmap_dtype,
        pad_token_id=train_config.model.pad_token_id,
        bos_token_id=getattr(train_config.model, "bos_token_id", 2),
        eos_token_id=train_config.model.eos_token_id,
        include_instance_metadata=include_instance_metadata,
    )
    work_dir = Path(train_config.save_folder) / "train_data_dlm"
    if get_global_rank() == 0:
        if work_dir.is_dir() and not train_config.save_overwrite:
            raise OLMoConfigurationError(
                "DLM train data working directory already exists, use --save_overwrite to overwrite"
            )
        work_dir.mkdir(exist_ok=True, parents=True)

    seed = train_config.data.seed if train_config.data.seed is not None else train_config.seed
    iterable_dataset = IterableDataset(
        dataset,
        train_config.global_train_batch_size,
        seed=seed,
        epoch=train_config.epoch or 0,
        shuffle=True,
        drop_last=train_config.data.drop_last,
        world_size=world_size,
        rank=rank,
        fs_local_rank=fs_local_rank,
        work_dir=work_dir,
    )
    barrier()

    collator = DLMDataCollator(
        max_seq_length=train_config.model.max_sequence_length,
        pad_token_id=train_config.model.pad_token_id,
        max_input_seq_length=getattr(train_config.dlm, "max_input_length", None),
        conditioning_mode=getattr(train_config.dlm, "conditioning_mode", "unconditional"),
        unconditional_prob=getattr(train_config.dlm, "unconditional_prob", 0.1),
        condition_pattern=getattr(train_config.dlm, "condition_pattern", "mixed"),
        condition_min_mask_ratio=getattr(train_config.dlm, "condition_min_mask_ratio", 0.1),
        condition_max_mask_ratio=getattr(train_config.dlm, "condition_max_mask_ratio", 0.5),
        condition_min_span_length=getattr(train_config.dlm, "condition_min_span_length", 30),
        condition_max_span_length=getattr(train_config.dlm, "condition_max_span_length", 50),
        condition_multi_min_spans=getattr(train_config.dlm, "condition_multi_min_spans", 4),
        condition_multi_max_spans=getattr(train_config.dlm, "condition_multi_max_spans", 8),
        condition_prefix_suffix_weight=getattr(train_config.dlm, "condition_prefix_suffix_weight", 1.0),
        condition_single_span_weight=getattr(train_config.dlm, "condition_single_span_weight", 1.0),
        condition_multi_span_weight=getattr(train_config.dlm, "condition_multi_span_weight", 1.0),
    )
    return DataLoader(
        iterable_dataset,
        batch_size=train_config.device_train_batch_size,
        drop_last=train_config.data.drop_last,
        collate_fn=collator,
        num_workers=train_config.data.num_workers,
        pin_memory=train_config.data.pin_memory,
        prefetch_factor=None if train_config.data.num_workers == 0 else train_config.data.prefetch_factor,
        persistent_workers=False if train_config.data.num_workers == 0 else train_config.data.persistent_workers,
        timeout=train_config.data.timeout,
    )


def build_eval_dlm_dataloader(
    train_config: TrainConfig,
    data_config: DataConfig,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> DataLoader:
    """Build an eval loader with the same sample-boundary semantics as DLM training."""
    paths: List[str] = []
    metadata: List[Dict[str, Any]] = []
    if data_config.paths:
        if data_config.datasets:
            raise OLMoConfigurationError("DataConfig.paths is mutually exclusive with DataConfig.datasets")
        paths = list(data_config.paths)
        metadata = [{} for _ in paths]
    elif data_config.datasets:
        for label in sorted(data_config.datasets):
            label_paths = data_config.datasets[label]
            paths.extend(label_paths)
            metadata.extend({"label": label} for _ in label_paths)
    else:
        raise OLMoConfigurationError("One of DataConfig.paths or DataConfig.datasets is required")

    max_length = _resolve_dlm_max_length(train_config)
    dataset = DLMTokensDataset(
        paths,
        chunk_size=max_length,
        memmap_dtype=data_config.effective_memmap_dtype,
        pad_token_id=train_config.model.pad_token_id,
        bos_token_id=getattr(train_config.model, "bos_token_id", 2),
        eos_token_id=train_config.model.eos_token_id,
        include_instance_metadata=True,
        metadata=metadata,
    )
    samples_per_device = len(dataset) // get_world_size()
    if data_config.drop_last:
        batch_size = min(batch_size, samples_per_device)
        if batch_size < 1:
            raise OLMoConfigurationError("DLM eval dataset is too small for the distributed world size")
    seed = data_config.seed if data_config.seed is not None else train_config.seed
    sampler = torch.utils.data.DistributedSampler(
        dataset,
        drop_last=data_config.drop_last,
        shuffle=shuffle,
        num_replicas=get_world_size(),
        rank=get_global_rank(),
        seed=seed,
    )
    collator = DLMDataCollator(
        max_seq_length=train_config.model.max_sequence_length,
        pad_token_id=train_config.model.pad_token_id,
        max_input_seq_length=getattr(train_config.dlm, "max_input_length", None),
        # Keep eval deterministic and comparable: evaluate full unconditional
        # denoising here. Conditional generation is measured separately.
        conditioning_mode="unconditional",
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        sampler=sampler,
        drop_last=data_config.drop_last,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        prefetch_factor=None if data_config.num_workers == 0 else data_config.prefetch_factor,
        persistent_workers=False if data_config.num_workers == 0 else data_config.persistent_workers,
        timeout=data_config.timeout,
    )
