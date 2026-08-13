# -*- coding: utf-8 -*-
from __future__ import annotations

import gzip
import math
from dataclasses import dataclass
from typing import List, Optional

from torch.utils.data import DataLoader, IterableDataset, Subset

from basecall.data_multifolder import (
    JsonlFile,
    MultiJsonlSignalRefDataset,
    StreamingJsonlSignalRefDataset,
    create_collate_fn,
    create_vq_collate_fn,
    scan_jsonl_files,
    split_indices,
    split_jsonl_files_by_group,
    split_jsonl_records_per_file,
)

from .config import DataConfig, ModelConfig, TrainConfig


@dataclass
class DatasetBundle:
    train_dataset: object
    val_dataset: Optional[object]
    test_dataset: Optional[object]
    summary: str


@dataclass
class LoaderBundle:
    train_loader: DataLoader
    val_loader: Optional[DataLoader]
    test_loader: Optional[DataLoader]
    steps_per_epoch: int


def parse_path_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _scan(paths: List[str], cfg: DataConfig, *, group_by: str) -> List[JsonlFile]:
    return scan_jsonl_files(paths, group_by=group_by, recursive=cfg.recursive)


def _file_stream(files: List[JsonlFile], cfg: DataConfig, *, split_name: str = "train") -> Optional[StreamingJsonlSignalRefDataset]:
    if not files:
        return None
    return StreamingJsonlSignalRefDataset(
        jsonl_files=files,
        split_name=split_name,
        split_mode="file",
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
        seed=cfg.split_seed,
        token_offset=cfg.token_offset,
        shuffle_buffer_size=cfg.shuffle_buffer_size if split_name == "train" else 0,
    )


def build_datasets(cfg: DataConfig) -> DatasetBundle:
    explicit_train = parse_path_list(cfg.train_jsonl_paths)
    explicit_val = parse_path_list(cfg.val_jsonl_paths)
    explicit_test = parse_path_list(cfg.test_jsonl_paths)

    if explicit_train or explicit_val or explicit_test:
        if not explicit_train:
            raise ValueError("--train_jsonl_paths is required when explicit split paths are used.")
        scan_group = cfg.group_by if cfg.group_by in {"folder", "file"} else "file"
        train_files = _scan(explicit_train, cfg, group_by=scan_group)
        val_files = _scan(explicit_val, cfg, group_by=scan_group) if explicit_val else []
        test_files = _scan(explicit_test, cfg, group_by=scan_group) if explicit_test else []
        if cfg.streaming:
            return DatasetBundle(
                train_dataset=_file_stream(train_files, cfg) or [],
                val_dataset=_file_stream(val_files, cfg, split_name="val"),
                test_dataset=_file_stream(test_files, cfg, split_name="test"),
                summary=f"explicit splits streaming train_files={len(train_files)} val_files={len(val_files)} test_files={len(test_files)}",
            )
        return DatasetBundle(
            train_dataset=MultiJsonlSignalRefDataset(train_files, token_offset=cfg.token_offset),
            val_dataset=MultiJsonlSignalRefDataset(val_files, token_offset=cfg.token_offset) if val_files else None,
            test_dataset=MultiJsonlSignalRefDataset(test_files, token_offset=cfg.token_offset) if test_files else None,
            summary=f"explicit splits eager train_files={len(train_files)} val_files={len(val_files)} test_files={len(test_files)}",
        )

    jsonl_paths = parse_path_list(cfg.jsonl_paths)
    if not jsonl_paths:
        raise ValueError("Provide --jsonl_paths or explicit split paths.")

    scan_group = cfg.group_by if cfg.group_by in {"folder", "file"} else "file"
    files = _scan(jsonl_paths, cfg, group_by=scan_group)

    if cfg.group_by in {"record", "record_per_file"} and not cfg.streaming and not cfg.allow_eager_record_split:
        raise ValueError(
            f"--group_by {cfg.group_by} without --streaming eagerly loads all records. "
            "Use --streaming or pass --allow_eager_record_split for small datasets."
        )

    if cfg.group_by == "record":
        if cfg.streaming:
            return DatasetBundle(
                train_dataset=StreamingJsonlSignalRefDataset(files, "train", "record", cfg.train_ratio, cfg.val_ratio, cfg.test_ratio, cfg.split_seed, cfg.token_offset, cfg.shuffle_buffer_size),
                val_dataset=StreamingJsonlSignalRefDataset(files, "val", "record", cfg.train_ratio, cfg.val_ratio, cfg.test_ratio, cfg.split_seed, cfg.token_offset) if cfg.val_ratio > 0 else None,
                test_dataset=StreamingJsonlSignalRefDataset(files, "test", "record", cfg.train_ratio, cfg.val_ratio, cfg.test_ratio, cfg.split_seed, cfg.token_offset) if cfg.test_ratio > 0 else None,
                summary=f"record streaming files={len(files)}",
            )
        all_dataset = MultiJsonlSignalRefDataset(files, token_offset=cfg.token_offset)
        train_idx, val_idx, test_idx = split_indices(len(all_dataset), cfg.train_ratio, cfg.val_ratio, cfg.test_ratio, cfg.split_seed)
        return DatasetBundle(
            train_dataset=Subset(all_dataset, train_idx),
            val_dataset=Subset(all_dataset, val_idx) if val_idx else None,
            test_dataset=Subset(all_dataset, test_idx) if test_idx else None,
            summary=f"record eager records={len(all_dataset)}",
        )

    if cfg.group_by == "record_per_file":
        if cfg.streaming:
            return DatasetBundle(
                train_dataset=StreamingJsonlSignalRefDataset(files, "train", "record_per_file", cfg.train_ratio, cfg.val_ratio, cfg.test_ratio, cfg.split_seed, cfg.token_offset, cfg.shuffle_buffer_size),
                val_dataset=StreamingJsonlSignalRefDataset(files, "val", "record_per_file", cfg.train_ratio, cfg.val_ratio, cfg.test_ratio, cfg.split_seed, cfg.token_offset) if cfg.val_ratio > 0 else None,
                test_dataset=StreamingJsonlSignalRefDataset(files, "test", "record_per_file", cfg.train_ratio, cfg.val_ratio, cfg.test_ratio, cfg.split_seed, cfg.token_offset) if cfg.test_ratio > 0 else None,
                summary=f"record_per_file streaming files={len(files)}",
            )
        train_ds, val_ds, test_ds = split_jsonl_records_per_file(files, cfg.train_ratio, cfg.val_ratio, cfg.test_ratio, cfg.split_seed, cfg.token_offset)
        return DatasetBundle(train_ds, val_ds, test_ds, summary=f"record_per_file eager files={len(files)}")

    train_files, val_files, test_files = split_jsonl_files_by_group(files, cfg.train_ratio, cfg.val_ratio, cfg.test_ratio, cfg.split_seed)
    if cfg.streaming:
        return DatasetBundle(
            train_dataset=_file_stream(train_files, cfg) or [],
            val_dataset=_file_stream(val_files, cfg, split_name="val"),
            test_dataset=_file_stream(test_files, cfg, split_name="test"),
            summary=f"{cfg.group_by} streaming train_files={len(train_files)} val_files={len(val_files)} test_files={len(test_files)}",
        )
    return DatasetBundle(
        train_dataset=MultiJsonlSignalRefDataset(train_files, token_offset=cfg.token_offset),
        val_dataset=MultiJsonlSignalRefDataset(val_files, token_offset=cfg.token_offset) if val_files else None,
        test_dataset=MultiJsonlSignalRefDataset(test_files, token_offset=cfg.token_offset) if test_files else None,
        summary=f"{cfg.group_by} eager train_files={len(train_files)} val_files={len(val_files)} test_files={len(test_files)}",
    )


def make_collate(model_cfg: ModelConfig, tokenizer):
    return create_vq_collate_fn() if model_cfg.feature_source == "vq_embedding" else create_collate_fn(tokenizer)


def _safe_len(value) -> Optional[int]:
    try:
        return len(value)
    except TypeError:
        return None


def _count_jsonl_records(path: str) -> int:
    count = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def estimate_streaming_steps(dataset, batch_size: int) -> Optional[int]:
    if not isinstance(dataset, StreamingJsonlSignalRefDataset):
        return None
    total = sum(_count_jsonl_records(item.path) for item in dataset.jsonl_files)
    if dataset.split_mode not in {"file", "folder"}:
        total = int(round(total * float(dataset.train_ratio)))
    return max(int(math.ceil(total / max(batch_size, 1))), 1)


def build_loaders(bundle: DatasetBundle, model_cfg: ModelConfig, train_cfg: TrainConfig, tokenizer, *, pin_memory: bool) -> LoaderBundle:
    collate_fn = make_collate(model_cfg, tokenizer)
    train_loader = DataLoader(
        bundle.train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=not isinstance(bundle.train_dataset, IterableDataset),
        num_workers=train_cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
    )
    val_loader = None
    if bundle.val_dataset is not None:
        val_loader = DataLoader(
            bundle.val_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=train_cfg.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            collate_fn=collate_fn,
        )
    test_loader = None
    if bundle.test_dataset is not None:
        test_loader = DataLoader(
            bundle.test_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=train_cfg.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            collate_fn=collate_fn,
        )

    if train_cfg.max_steps_per_epoch > 0:
        steps = train_cfg.max_steps_per_epoch
    else:
        steps = _safe_len(train_loader) or train_cfg.steps_per_epoch or estimate_streaming_steps(bundle.train_dataset, train_cfg.batch_size)
    if steps is None or steps <= 0:
        raise ValueError("Could not infer steps_per_epoch; set --steps_per_epoch or --max_steps_per_epoch.")

    return LoaderBundle(train_loader, val_loader, test_loader, int(steps))
