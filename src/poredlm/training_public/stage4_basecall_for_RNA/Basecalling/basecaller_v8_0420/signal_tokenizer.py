from __future__ import annotations

import re
from typing import Sequence

import torch


class BwavTokenizer:
    """Minimal tokenizer for public PoreDLM ``<|bwav:N|>`` signal strings."""

    _pattern = re.compile(r"<\|bwav:(\d+)\|>", flags=re.IGNORECASE)

    def __init__(
        self,
        vocab_size: int = 65664,
        token_offset: int = 128,
        pad_token_id: int = 1,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        mask_token_id: int = 4,
        add_cls_token: bool = False,
        cls_token_id: int = 5,
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.token_offset = int(token_offset)
        self.pad_token_id = int(pad_token_id)
        self.bos_token_id = int(bos_token_id)
        self.eos_token_id = int(eos_token_id)
        self.mask_token_id = int(mask_token_id)
        self.add_cls_token = bool(add_cls_token)
        self.cls_token_id = int(cls_token_id)
        # CTC outputs should align to signal-code positions only. BOS/CLS/EOS
        # remain in the backbone input as context, but are not CTC time steps.
        self.ctc_prefix_tokens = 1 + int(self.add_cls_token)  # BOS (+ CLS)
        self.ctc_suffix_tokens = 1  # EOS
        if self.add_cls_token and self.cls_token_id in {
            self.pad_token_id,
            self.bos_token_id,
            self.eos_token_id,
            self.mask_token_id,
        }:
            raise ValueError("cls_token_id must differ from pad, bos, eos, and mask token IDs")

    def ctc_input_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Convert full tokenizer lengths to the number of signal-code steps."""
        special_count = self.ctc_prefix_tokens + self.ctc_suffix_tokens
        return (input_lengths.to(dtype=torch.long) - special_count).clamp_min(0)

    def __len__(self) -> int:
        return self.vocab_size

    def _encode(self, text: str) -> list[int]:
        code_ids = [int(value) for value in self._pattern.findall(str(text))]
        if not code_ids:
            raise ValueError("Signal contains no <|bwav:N|> tokens.")
        ids = [self.bos_token_id, *(value + self.token_offset for value in code_ids), self.eos_token_id]
        if self.add_cls_token:
            ids.insert(0, self.cls_token_id)
        bad = [value for value in ids if not 0 <= value < self.vocab_size]
        if bad:
            raise ValueError(
                f"Token id {bad[0]} is outside vocab_size={self.vocab_size}; "
                "check whether the input has already been offset."
            )
        return ids

    def __call__(
        self,
        texts: str | Sequence[str],
        *,
        return_tensors: str | None = None,
        padding: bool = False,
        truncation: bool = False,
        **_kwargs,
    ) -> dict[str, torch.Tensor | list[list[int]]]:
        del truncation
        if isinstance(texts, str):
            texts = [texts]
        sequences = [self._encode(text) for text in texts]
        width = max(len(ids) for ids in sequences) if padding else None
        if width is not None:
            masks = [[1] * len(ids) + [0] * (width - len(ids)) for ids in sequences]
            sequences = [ids + [self.pad_token_id] * (width - len(ids)) for ids in sequences]
        else:
            masks = [[1] * len(ids) for ids in sequences]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(sequences, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        return {"input_ids": sequences, "attention_mask": masks}
