"""Datasets and collators for BERT encoder pretraining."""

from __future__ import annotations

import gzip
import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


BWAV_TOKEN_PATTERN = re.compile(r"<\|bwav:(\d+)\|>")


@dataclass(frozen=True)
class Stage2Batch:
    """Token-id batch consumed by the Stage 2 BERT model."""

    input_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None


def parse_bwav_token_text(text: str) -> torch.Tensor:
    """Parse ``<|bwav:123|>`` token text into integer token ids."""

    ids = BWAV_TOKEN_PATTERN.findall(text)
    if not ids:
        raise ValueError("No <|bwav:id|> tokens found in jsonl text field.")
    return torch.tensor([int(token_id) for token_id in ids], dtype=torch.long)


class Stage2TokenJsonlDataset(Dataset):
    """JSONL/JSONL.GZ dataset for VQ token ids.

    Each line should be a JSON object with a `text` field containing tokens like
    `<|bwav:3073|><|bwav:32601|>...`.
    """

    def __init__(
        self,
        data_dir: str,
        pattern: str = "*.jsonl.gz",
        max_cache_files: int = 2,
    ) -> None:
        self.data_dir = data_dir
        self.pattern = pattern
        self.max_cache_files = max_cache_files
        self.files = sorted(Path(data_dir).glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No files matching {pattern!r} under {data_dir!r}.")

        self.file_line_counts: list[int] = []
        self.offsets = [0]
        for path in self.files:
            line_count = self._count_lines(path)
            self.file_line_counts.append(line_count)
            self.offsets.append(self.offsets[-1] + line_count)

        self._cache: OrderedDict[Path, list[torch.Tensor]] = OrderedDict()

    def __len__(self) -> int:
        return self.offsets[-1]

    @staticmethod
    def _open_text(path: Path):
        if path.suffix == ".gz":
            return gzip.open(path, "rt", encoding="utf-8")
        return path.open("r", encoding="utf-8")

    def _count_lines(self, path: Path) -> int:
        with self._open_text(path) as handle:
            return sum(1 for line in handle if line.strip())

    def _load_file(self, path: Path) -> list[torch.Tensor]:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]

        samples: list[torch.Tensor] = []
        with self._open_text(path) as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if "text" not in item:
                    raise KeyError(f"Missing 'text' field in {path} line {line_number}.")
                samples.append(parse_bwav_token_text(item["text"]))

        if len(self._cache) >= self.max_cache_files:
            self._cache.popitem(last=False)
        self._cache[path] = samples
        return samples

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range [0, {len(self)})")

        for file_id in range(len(self.offsets) - 1):
            if self.offsets[file_id] <= index < self.offsets[file_id + 1]:
                return file_id, index - self.offsets[file_id]

        raise IndexError(f"Index {index} was not found in any jsonl file.")

    def __getitem__(self, index: int) -> torch.Tensor:
        file_id, local_index = self._locate(index)
        samples = self._load_file(self.files[file_id])
        return samples[local_index]


class Stage2Collator:
    """Pad token-id samples into one batch."""

    def __init__(self, pad_token_id: int = 0, max_length: int | None = None) -> None:
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, samples: list[torch.Tensor]) -> Stage2Batch:
        if self.max_length is not None:
            samples = [sample[: self.max_length] for sample in samples]

        max_len = max(sample.shape[0] for sample in samples)
        input_ids = torch.full(
            (len(samples), max_len),
            fill_value=self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((len(samples), max_len), dtype=torch.long)
        for i, sample in enumerate(samples):
            if sample.ndim != 1:
                raise ValueError(f"Token samples must have shape [seq_len]. Got {tuple(sample.shape)}.")
            length = sample.shape[0]
            input_ids[i, :length] = sample.long()
            attention_mask[i, :length] = 1
        return Stage2Batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )