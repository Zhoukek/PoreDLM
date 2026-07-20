from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import TrainConfig
from .data.iterable_dataset import IterableDataset
from .exceptions import OLMoConfigurationError
from .torch_util import barrier, get_global_rank

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
        include_instance_metadata: bool = False,
    ) -> None:
        if not paths:
            raise OLMoConfigurationError("DLM data requires at least one path in cfg.data.paths")
        self.paths = _expand_token_paths(paths)
        self.chunk_size = chunk_size
        self.memmap_dtype = memmap_dtype
        self.pad_token_id = pad_token_id
        self.include_instance_metadata = include_instance_metadata
        self._arrays: List[np.ndarray] = []
        self._offsets: List[Tuple[int, int]] = []

        start = 0
        for path in self.paths:
            array = self._load_array(path)
            if array.ndim == 0:
                raise OLMoConfigurationError(f"DLM data file {path!r} is scalar; expected 1D or 2D token IDs")
            if array.ndim > 2:
                raise OLMoConfigurationError(f"DLM data file {path!r} has shape {array.shape}; expected 1D or 2D")
            num_instances = self._num_instances(array)
            if num_instances == 0:
                raise OLMoConfigurationError(f"DLM data file {path!r} has no complete training instances")
            self._arrays.append(array)
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
            usable_tokens = (token_count // self.chunk_size) * self.chunk_size
            if usable_tokens == 0:
                raise OLMoConfigurationError(
                    f"Could not load {path!r} with np.load ({exc!r}), and the raw file is too small"
                ) from exc
            log.warning(
                "Could not load %s with np.load; treating it as a raw %s token memmap",
                path,
                np.dtype(self.memmap_dtype).name,
            )
            return np.memmap(path, dtype=self.memmap_dtype, mode="r", shape=(usable_tokens,))

    def _num_instances(self, array: np.ndarray) -> int:
        if array.ndim == 1:
            return int(array.shape[0] // self.chunk_size)
        return int(array.shape[0])

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
        if array.ndim == 1:
            start = local_index * self.chunk_size
            input_ids = np.asarray(array[start : start + self.chunk_size], dtype=np.int64)
        else:
            input_ids = np.asarray(array[local_index], dtype=np.int64)

        out: Dict[str, Any] = {"input_ids": input_ids}
        if self.include_instance_metadata:
            out["metadata"] = {"path": self.paths[array_index]}
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
