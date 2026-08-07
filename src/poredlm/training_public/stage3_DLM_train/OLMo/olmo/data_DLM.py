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
    def __init__(self, *, max_seq_length: int, pad_token_id: int, max_input_seq_length: Optional[int] = None) -> None:
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id
        self.max_input_seq_length = max_input_seq_length

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
        is_cond = pos < np.asarray(cond_lens, dtype=np.int64)[:, None]
        is_valid = (pos < total_lens[:, None]) & (ids != self.pad_token_id)
        encoder_attn, attn, cond_seq_mask = _build_self_attn_cond_masks(is_cond, is_valid)

        result: Dict[str, Any] = {
            "input_ids": torch.from_numpy(ids).long(),
            "encoder_attention_mask": torch.from_numpy(encoder_attn),
            "attention_mask": torch.from_numpy(attn),
            "cond_seq_mask": torch.from_numpy(cond_seq_mask),
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
