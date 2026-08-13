# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List

import torch

from basecall.ctc_crf import decode as ctc_crf_decode
from basecall.metrics import ctc_viterbi_decode, koi_beam_search_decode
from basecall.utils import BLANK_IDX


def rebuild_target_seqs(target_labels: torch.Tensor, target_lengths: torch.Tensor) -> List[List[int]]:
    labels = target_labels.detach().cpu().tolist()
    lengths = target_lengths.detach().cpu().tolist()
    out: List[List[int]] = []
    offset = 0
    for length in lengths:
        n = int(length)
        out.append([int(x) for x in labels[offset : offset + n]])
        offset += n
    return out


def decode_batch(
    logits_tbc: torch.Tensor,
    input_lengths: torch.Tensor,
    *,
    decoder: str,
    koi_blank_score: float,
) -> List[List[int]]:
    if decoder == "koi":
        return koi_beam_search_decode(logits_tbc, blank_score=koi_blank_score, input_lengths=input_lengths)
    if decoder == "ctc_viterbi":
        return ctc_viterbi_decode(logits_tbc, input_lengths=input_lengths, blank_idx=BLANK_IDX)
    if decoder == "ctc_crf":
        logits_tbc = logits_tbc.float()
        pred: List[List[int]] = []
        for batch_idx, step_len in enumerate(input_lengths.detach().cpu().tolist()):
            n = int(step_len)
            if n <= 0:
                pred.append([])
                continue
            decoded = ctc_crf_decode(logits_tbc[:n, batch_idx : batch_idx + 1, :], blank_idx=BLANK_IDX)[0]
            pred.append(decoded[:n])
        return pred
    raise ValueError(f"Unsupported decoder: {decoder}")
