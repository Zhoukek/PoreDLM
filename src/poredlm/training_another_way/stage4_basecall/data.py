"""Raw signal and reference-pair dataset for continuous basecalling."""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def find_signal_reference_pairs(paths: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in paths:
        path = Path(item)
        candidates = []
        if path.is_dir():
            candidates = [Path(p) for p in glob.glob(str(path / "**" / "*_chunks.npy"), recursive=True)]
        elif path.is_file():
            candidates = [path]
        else:
            raise FileNotFoundError(f"Basecalling input path not found: {item}")
        for signal_path in sorted(candidates):
            if signal_path.name.endswith("_references.npy"):
                continue
            if signal_path.name.endswith("_chunks.npy"):
                reference_path = signal_path.with_name(
                    signal_path.name[: -len("_chunks.npy")] + "_references.npy"
                )
            else:
                reference_path = signal_path.with_name(signal_path.stem + "_references.npy")
            if not reference_path.exists():
                raise FileNotFoundError(f"Missing reference pair for {signal_path}: {reference_path}")
            pairs.append((str(signal_path), str(reference_path)))
    if not pairs:
        raise FileNotFoundError("No *_chunks.npy files with matching *_references.npy files found.")
    return sorted(set(pairs))


class SignalReferenceDataset(Dataset):
    def __init__(self, paths: list[str]):
        self.pairs = find_signal_reference_pairs(paths)
        self.entries: list[tuple[int, int]] = []
        self._signals: dict[int, np.ndarray] = {}
        self._references: dict[int, np.ndarray] = {}
        for file_index, (signal_path, reference_path) in enumerate(self.pairs):
            signal = np.load(signal_path, mmap_mode="r")
            reference = np.load(reference_path, mmap_mode="r")
            if signal.ndim != 2 or reference.ndim != 2 or signal.shape[0] != reference.shape[0]:
                raise ValueError(
                    f"Invalid signal/reference shapes: {signal_path}={signal.shape}, "
                    f"{reference_path}={reference.shape}"
                )
            self.entries.extend((file_index, row) for row in range(signal.shape[0]))
        print(f"[Dataset] files={len(self.pairs)} reads={len(self.entries)}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, object]:
        file_index, row = self.entries[index]
        if file_index not in self._signals:
            signal_path, reference_path = self.pairs[file_index]
            self._signals[file_index] = np.load(signal_path, mmap_mode="r")
            self._references[file_index] = np.load(reference_path, mmap_mode="r")
        signal = torch.from_numpy(np.asarray(self._signals[file_index][row], dtype=np.float32).copy())
        reference = np.asarray(self._references[file_index][row]).reshape(-1)
        labels = torch.from_numpy(reference[reference > 0].astype(np.int64, copy=False))
        return {"signal": signal, "labels": labels}


def collate_signal_reference(batch: list[dict[str, object]]) -> dict[str, torch.Tensor | list[list[int]]]:
    signals = [item["signal"] for item in batch]
    labels = [item["labels"] for item in batch]
    lengths = torch.tensor([int(signal.numel()) for signal in signals], dtype=torch.long)
    max_length = int(lengths.max().item()) if len(signals) else 0
    padded = torch.zeros((len(signals), max_length), dtype=torch.float32)
    for index, signal in enumerate(signals):
        padded[index, : signal.numel()] = signal
    target_lengths = torch.tensor([int(target.numel()) for target in labels], dtype=torch.long)
    target_labels = torch.cat(labels) if labels else torch.empty(0, dtype=torch.long)
    return {
        "signal": padded,
        "signal_lengths": lengths,
        "target_labels": target_labels,
        "target_lengths": target_lengths,
        "target_seqs": [target.tolist() for target in labels],
    }
