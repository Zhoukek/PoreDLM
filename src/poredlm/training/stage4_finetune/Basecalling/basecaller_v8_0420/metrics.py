# -*- coding: utf-8 -*-
from typing import List, Dict, Any, Optional, Tuple
import csv
import os
import re
import importlib.util

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import parasail
from collections import defaultdict, OrderedDict


from .utils import BLANK_IDX, ID2BASE, BASE2ID
from .ctc import decode as ctc_decode


_SPLIT_CIGAR = re.compile(r"(?P<len>\d+)(?P<op>\D+)")


def koi_beam_search_decode(
    logits_tbc: torch.Tensor,
    beam_width: int = 32,
    beam_cut: float = 100.0,
    scale: float = 1.0,
    offset: float = 0.0,
    blank_score: float = 2.0,
    reverse: bool = False,
    input_lengths: Optional[torch.Tensor] = None,
) -> List[List[int]]:
    try:
        from koi.decode import beam_search, to_str  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Koi beam_search decoder requested but ont-koi is not installed. "
            "Install ont-koi and ensure it matches your PyTorch/CUDA version."
        ) from exc

    if input_lengths is None:
        lengths = [logits_tbc.shape[0]] * logits_tbc.shape[1]
    else:
        lengths = [min(int(x), logits_tbc.shape[0]) for x in input_lengths.cpu().tolist()]
    decoded: List[List[int]] = []
    for b, length in enumerate(lengths):
        if length <= 0:
            decoded.append([])
            continue
        scores = logits_tbc[:length, b : b + 1, :]
        if scores.is_cuda:
            scores = scores.half()
            scores = scores.transpose(0, 1)
        sequence, _qstring, _moves = beam_search(
            scores,
            beam_width=beam_width,
            beam_cut=beam_cut,
            scale=scale,
            offset=offset,
            blank_score=blank_score,
        )
        if reverse:
            sequence = sequence[::-1]
        seq_str = sequence if isinstance(sequence[0], str) else to_str(sequence[0])
        decoded.append([BASE2ID.get(base, BLANK_IDX) for base in seq_str])
    return decoded



def ctc_viterbi_decode(
    logits_tbc: torch.Tensor,
    input_lengths: Optional[torch.Tensor] = None,
    blank_idx: int = BLANK_IDX,
) -> List[List[int]]:
    """
    Bonito-style CTC Viterbi path collapse (beamsize=1 equivalent):
    timestep argmax -> collapse repeats -> remove blank.
    """
    return ctc_decode(logits_tbc=logits_tbc, input_lengths=input_lengths, blank_idx=blank_idx, beamsize=1)



def _ids_to_bases(ids: List[int], drop_blank: bool = True) -> str:
    bases: List[str] = []
    for i in ids:
        if drop_blank and i == BLANK_IDX:
            continue
        base = ID2BASE.get(i, "")
        if base:
            bases.append(base)
    return "".join(bases)


def _normalize_base_seq(seq: Any) -> str:
    """Normalize sequence input to a DNA base string for alignment.
    Accepts str, list/tuple/np.ndarray/torch.Tensor of token ids.
    """
    if seq is None:
        return ""
    if isinstance(seq, str):
        return seq
    if isinstance(seq, torch.Tensor):
        seq = seq.detach().cpu().tolist()
    if isinstance(seq, np.ndarray):
        seq = seq.tolist()
    if isinstance(seq, (list, tuple)):
        ids: List[int] = []
        for x in seq:
            try:
                ids.append(int(x))
            except Exception:
                continue
        return _ids_to_bases(ids, drop_blank=True)
    return str(seq)



def _safe_parasail_alignment(pred_seq: str, ref_seq: str):
    """Run parasail trace alignment safely; return None when traceback/cigar is unavailable."""
    if len(ref_seq) == 0 or len(pred_seq) == 0:
        return None
    try:
        alignment = parasail.sw_trace_striped_32(pred_seq, ref_seq, 8, 4, parasail.dnafull)
    except Exception:
        return None
    try:
        _ = alignment.cigar  # ensure traceback/cigar is available in this parasail result
    except Exception:
        return None
    if not hasattr(alignment, "traceback") or alignment.traceback is None:
        return None
    return alignment


def parasail_to_sam(result, seq):
    """
    Extract reference start and sam compatible cigar string.

    :param result: parasail alignment result.
    :param seq: query sequence.

    :returns: reference start coordinate, cigar string.
    """
    cigstr = result.cigar.decode.decode()
    first = re.search(_SPLIT_CIGAR, cigstr)
    if first is None:
        raise ValueError(f"Invalid or empty CIGAR from parasail result: {cigstr!r}")

    first_count, first_op = first.groups()
    prefix = first.group()
    rstart = result.cigar.beg_ref
    cliplen = result.cigar.beg_query

    clip = '' if cliplen == 0 else '{}S'.format(cliplen)
    if first_op == 'I':
        pre = '{}S'.format(int(first_count) + cliplen)
    elif first_op == 'D':
        pre = clip
        rstart = int(first_count)
    else:
        pre = '{}{}'.format(clip, prefix)

    mid = cigstr[len(prefix):]
    end_clip = len(seq) - result.end_query - 1
    suf = '{}S'.format(end_clip) if end_clip > 0 else ''
    new_cigstr = ''.join((pre, mid, suf))
    return rstart, new_cigstr


def cal_bonito_accuracy(pred_seq, ref_seq, balanced=False, min_coverage=0.0):
    """
    Calculate the accuracy between `ref` and `seq`
    """
    pred_seq = _normalize_base_seq(pred_seq)
    ref_seq = _normalize_base_seq(ref_seq)
    if len(ref_seq) == 0:
        return 0.0
    if len(pred_seq) == 0:
        return 0.0

    alignment = _safe_parasail_alignment(pred_seq, ref_seq)
    if alignment is None:
        return 0.0

    counts = defaultdict(int)

    try:
        _, cigar = parasail_to_sam(alignment, pred_seq)
    except Exception:
        return 0.0

    for count, op in re.findall(_SPLIT_CIGAR, cigar):
        counts[op] += int(count)

    r_coverage = len(alignment.traceback.ref) / max(len(ref_seq), 1)
    if r_coverage < min_coverage:
        return 0.0

    if balanced:
        accuracy = (counts['='] - counts['I']) / (counts['='] + counts['X'] + counts['D'])
    else:
        accuracy = counts['='] / (counts['='] + counts['I'] + counts['X'] + counts['D'])
    return accuracy * 100


def parasail_error_counts(pred_seq: str, ref_seq: str) -> Dict[str, int]:
    """
    Return cigar-derived counts from parasail alignment using the same rules as cal_bonito_accuracy.
    """
    pred_seq = _normalize_base_seq(pred_seq)
    ref_seq = _normalize_base_seq(ref_seq)
    counts = defaultdict(int)
    if len(ref_seq) == 0:
        return {"match": 0, "mismatch": 0, "ins": 0, "del": 0}
    if len(pred_seq) == 0:
        return {"match": 0, "mismatch": 0, "ins": 0, "del": len(ref_seq)}

    alignment = _safe_parasail_alignment(pred_seq, ref_seq)
    if alignment is None:
        return {"match": 0, "mismatch": 0, "ins": 0, "del": len(ref_seq)}

    try:
        _, cigar = parasail_to_sam(alignment, pred_seq)
    except Exception:
        return {"match": 0, "mismatch": 0, "ins": 0, "del": len(ref_seq)}
    for count, op in re.findall(_SPLIT_CIGAR, cigar):
        counts[op] += int(count)
    return {
        "match": int(counts.get("=", 0)),
        "mismatch": int(counts.get("X", 0)),
        "ins": int(counts.get("I", 0)),
        "del": int(counts.get("D", 0)),
    }


def parasail_match_vector(pred_seq: str, ref_seq: str) -> List[int]:
    """
    Build a reference-coordinate match vector (1=match, 0=mismatch/delete) from parasail CIGAR.
    Insertions and soft clips do not consume reference positions and are ignored.
    """
    pred_seq = _normalize_base_seq(pred_seq)
    ref_seq = _normalize_base_seq(ref_seq)
    if len(ref_seq) == 0:
        return []
    if len(pred_seq) == 0:
        return [0] * len(ref_seq)

    alignment = _safe_parasail_alignment(pred_seq, ref_seq)
    if alignment is None:
        return [0] * len(ref_seq)

    try:
        _, cigar = parasail_to_sam(alignment, pred_seq)
    except Exception:
        return [0] * len(ref_seq)
    match: List[int] = []
    for count, op in re.findall(_SPLIT_CIGAR, cigar):
        span = int(count)
        if op == "=":
            match.extend([1] * span)
        elif op in {"X", "D"}:
            match.extend([0] * span)
        else:
            continue
    return match


def batch_bonito_accuracy(
    pred_seqs: List[List[int]],
    ref_seqs: List[List[int]],
    balanced: bool = False,
    min_coverage: float = 0.0,
) -> float:
    """
    Bonito-style accuracy across a batch, averaged by per-read accuracy.
    返回百分比（0-100）。
    """
    if not pred_seqs or not ref_seqs:
        return 0.0
    scores = []
    for p_ids, r_ids in zip(pred_seqs, ref_seqs):
        p_str = _ids_to_bases(p_ids, drop_blank=True)
        r_str = _ids_to_bases(r_ids, drop_blank=True)
        scores.append(cal_bonito_accuracy(p_str, r_str, balanced=balanced, min_coverage=min_coverage))
    return float(np.mean(scores)) if scores else 0.0
    


# ---------------- 曲线绘图 & metrics 保存 ----------------

def plot_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_accs: List[float],
    save_path: Optional[str] = None,
):
    """画 Loss + Accuracy 曲线"""
    max_len = min(len(train_losses), len(val_losses), len(val_accs))
    if max_len == 0:
        print("[plot_curves] Empty metrics.")
        return

    train_losses = train_losses[:max_len]
    val_losses = val_losses[:max_len]
    val_accs = val_accs[:max_len]

    epochs = range(1, max_len + 1)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        "Training Metrics: Loss & Validation Accuracy",
        fontsize=14,
        fontweight="bold",
    )

    ax1.grid(True, linestyle="--", alpha=0.4, axis="both")

    color_loss_train = "#1f77b4"
    color_loss_val = "#ff7f0e"

    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", color=color_loss_train, fontsize=12)

    line1 = ax1.plot(
        epochs,
        train_losses,
        label="Train Loss",
        color=color_loss_train,
        linewidth=2,
    )
    line2 = ax1.plot(
        epochs,
        val_losses,
        label="Val Loss",
        color=color_loss_val,
        linestyle="--",
        linewidth=2,
    )
    ax1.tick_params(axis="y", labelcolor=color_loss_train)

    # PBMA on twin axis
    ax2 = ax1.twinx()
    color_acc = "#2ca02c"
    ax2.set_ylabel("Validation Accuracy", color=color_acc, fontsize=12)
    line3 = ax2.plot(
        epochs,
        val_accs,
        label="Val Acc",
        color=color_acc,
        linestyle=":",
        linewidth=2,
    )
    ax2.tick_params(axis="y", labelcolor=color_acc)

    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xlim(left=1, right=max_len)

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(
        lines,
        labels,
        loc="upper right",
        frameon=True,
        fancybox=True,
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"[Plot] Saved training curves to: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def save_metrics_csv(
    train_losses: List[float],
    val_losses: List[float],
    val_accs: List[float],
    csv_path: str,
):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc"])
        for e, tr, vl, pa in zip(
            range(1, len(train_losses) + 1),
            train_losses,
            val_losses,
            val_accs,
        ):
            writer.writerow([e, tr, vl, pa])
    print(f"[Metrics] Saved to {csv_path}")
