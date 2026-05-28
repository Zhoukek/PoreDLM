"""Dataset definitions for nanopore raw signal and derived embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SignalDatasetConfig:
    """Configuration shared by signal datasets."""

    data_dir: Path
    split: str = "train"
    sample_rate: int | None = None
    stride: int | None = None


class NanoporeSignalDataset:
    """Placeholder dataset for raw nanopore signal reads."""

    def __init__(self, config: SignalDatasetConfig) -> None:
        self.config = config

    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int) -> dict:
        raise IndexError("NanoporeSignalDataset is a scaffold; implement data indexing first.")
