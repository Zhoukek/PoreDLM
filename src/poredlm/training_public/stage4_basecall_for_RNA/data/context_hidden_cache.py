"""Disk-backed cache for frozen PoreDLM context-encoder outputs."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import torch


def token_sequence_key(token_ids: np.ndarray | list[int] | torch.Tensor) -> str:
    """Stable key for a non-padded tokenizer sequence (including special tokens)."""
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().to(device="cpu", dtype=torch.int32).numpy()
    array = np.asarray(token_ids, dtype=np.int32).reshape(-1)
    return hashlib.sha256(array.tobytes()).hexdigest()


def valid_sequence_lengths(input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> list[int]:
    if attention_mask is None:
        return [int(input_ids.shape[1])] * int(input_ids.shape[0])
    return [int(x) for x in attention_mask.to(dtype=torch.int64).sum(dim=1).tolist()]


def model_source_signature(model_name_or_path: str) -> str:
    """Fingerprint local config/weights/custom model code by path, size, and mtime."""
    root = Path(model_name_or_path).expanduser().resolve()
    if not root.exists():
        return hashlib.sha256(str(model_name_or_path).encode("utf-8")).hexdigest()
    if root.is_file():
        candidates = [root]
    else:
        candidates = [
            path for path in root.iterdir()
            if path.is_file()
            and (
                path.name == "config.json"
                or path.name.startswith("modeling_")
                or path.name.endswith(".safetensors")
                or path.name.endswith(".bin")
                or path.name.endswith(".index.json")
            )
        ]
    entries = []
    for path in sorted(candidates):
        stat = path.stat()
        entries.append((path.name, int(stat.st_size), int(stat.st_mtime_ns)))
    payload = json.dumps(entries, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ContextHiddenCache:
    """Read context features from a flat fp16 memmap plus a SQLite sequence index."""

    def __init__(
        self,
        cache_dir: str,
        *,
        model_name_or_path: str,
        tokenizer_type: str,
        tokenizer_token_offset: int,
        token_offset: int,
        add_cls_token: bool,
        cls_token_id: int,
        hidden_size: int,
        model_signature: str,
    ):
        self.cache_dir = Path(cache_dir)
        self.manifest_path = self.cache_dir / "manifest.json"
        self.index_path = self.cache_dir / "index.sqlite3"
        self.features_path = self.cache_dir / "context_hidden.fp16"
        if not self.manifest_path.is_file() or not self.index_path.is_file() or not self.features_path.is_file():
            raise FileNotFoundError(
                f"Incomplete context-hidden cache at {self.cache_dir}; expected manifest.json, "
                "index.sqlite3, and context_hidden.fp16."
            )
        with self.manifest_path.open("r", encoding="utf-8") as f:
            self.manifest: dict[str, Any] = json.load(f)

        expected = {
            "model_name_or_path": str(Path(model_name_or_path).expanduser().resolve()),
            "tokenizer_type": str(tokenizer_type),
            "tokenizer_token_offset": int(tokenizer_token_offset),
            "token_offset": int(token_offset),
            "add_cls_token": bool(add_cls_token),
            "cls_token_id": int(cls_token_id),
            "hidden_size": int(hidden_size),
            "model_signature": str(model_signature),
            "storage_dtype": "float16",
        }
        mismatches = {
            key: (self.manifest.get(key), value)
            for key, value in expected.items()
            if self.manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Context-hidden cache metadata mismatch (cached, requested): {mismatches}")

        self.hidden_size = int(hidden_size)
        self.total_tokens = int(self.manifest["total_tokens"])
        self._features = np.memmap(
            self.features_path,
            mode="r",
            dtype=np.float16,
            shape=(self.total_tokens, self.hidden_size),
        )
        self._connection = sqlite3.connect(f"file:{self.index_path}?mode=ro", uri=True)

    def get_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Fetch and pad a batch of cached context vectors to [B, T, H]."""
        ids_cpu = input_ids.detach().to(device="cpu", dtype=torch.int32)
        mask_cpu = attention_mask.detach().to(device="cpu") if attention_mask is not None else None
        lengths = valid_sequence_lengths(ids_cpu, mask_cpu)
        if not lengths:
            return torch.empty((0, 0, self.hidden_size), dtype=torch.float16)

        max_len = max(lengths)
        batch_features = torch.zeros((len(lengths), max_len, self.hidden_size), dtype=torch.float16)
        for row, length in enumerate(lengths):
            key = token_sequence_key(ids_cpu[row, :length])
            found = self._connection.execute(
                "SELECT token_offset, token_length FROM features WHERE sequence_key = ?", (key,)
            ).fetchone()
            if found is None:
                raise KeyError(
                    f"No cached context_hidden for token sequence {key[:16]}...; "
                    "regenerate the cache with the same data and tokenizer settings."
                )
            token_offset, token_length = int(found[0]), int(found[1])
            if token_length != length:
                raise ValueError(
                    f"Cached sequence length mismatch for {key[:16]}...: {token_length} vs {length}."
                )
            batch_features[row, :length] = torch.from_numpy(
                np.asarray(self._features[token_offset:token_offset + token_length]).copy()
            )
        return batch_features

    def close(self) -> None:
        if getattr(self, "_connection", None) is not None:
            self._connection.close()
            self._connection = None

