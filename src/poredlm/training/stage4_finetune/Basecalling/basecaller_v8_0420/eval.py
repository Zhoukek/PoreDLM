# -*- coding: utf-8 -*-
"""
eval.py

Evaluate jsonl.gz reads and report accuracy + error type proportions.
Also export alignment heatmaps for a few reads.

Example:
  python eval.py \
    --jsonl_paths /path/to/reads.jsonl.gz \
    --model_name_or_path your_hf_model \
    --ckpt ckpt_best.pt \
    --beam_width 32 \
    --batch_size 8 \
    --out_dir eval_out \
    --num_visualize 8 \
    --max_len 200
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data_multifolder import (
    scan_jsonl_files,
    MultiJsonlSignalRefDataset,
    scan_npy_pairs,
    MultiNpySignalRefDataset,
    create_collate_fn,
)
from .ctc_crf import decode as ctc_crf_decode
from .metrics import ctc_viterbi_decode, koi_beam_search_decode, batch_bonito_accuracy, cal_bonito_accuracy, parasail_error_counts
from .model import BasecallModel
from .utils import (
    ID2BASE,
    BLANK_IDX,
    seed_everything,
    infer_head_config_from_state_dict,
    infer_pre_head_type_from_state_dict,
    resolve_input_lengths,
)
from .callback import plot_alignment_heatmap, plot_aligned_heatmap_png, align_sequences_indel_aware


def _print_model_structure(model: torch.nn.Module, *, prefix: str = "[Model]") -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{prefix} class={model.__class__.__name__}")
    print(f"{prefix} pre_head={model.pre_head.__class__.__name__} head={model.base_head.__class__.__name__}")
    print(f"{prefix} params total={total_params:,} trainable={trainable_params:,}")
    print(f"{prefix} structure:")
    print(model)


def _parse_cigar(cigar: str) -> List[str]:
    ops = []
    num = ""
    for ch in cigar:
        if ch.isdigit():
            num += ch
        else:
            n = int(num) if num else 1
            ops.extend([ch] * n)
            num = ""
    return ops


def error_counts(pred_seq: str, ref_seq: str) -> Dict[str, int]:
    return parasail_error_counts(pred_seq, ref_seq)


def merge_counts(total: Dict[str, int], item: Dict[str, int]) -> Dict[str, int]:
    for k in total:
        total[k] += item.get(k, 0)
    return total


def counts_to_ratio(counts: Dict[str, int]) -> Dict[str, float]:
    denom = sum(counts.values())
    if denom == 0:
        return {k: 0.0 for k in counts}
    return {k: float(v) / float(denom) for k, v in counts.items()}


def _ids_to_bases(ids: List[int]) -> str:
    bases: List[str] = []
    for i in ids:
        if i == 0:
            continue
        base = ID2BASE.get(i, "")
        if base:
            bases.append(base)
    return "".join(bases)


def _normalize_base(ch: str) -> str:
    ch = (ch or "N").upper()
    if ch in {"A", "T", "G", "C"}:
        return ch
    return "N"


def _init_base_counts() -> Dict[str, int]:
    return {"A": 0, "T": 0, "G": 0, "C": 0, "N": 0}


def _init_mismatch_matrix() -> Dict[str, Dict[str, int]]:
    return {b: _init_base_counts() for b in ("A", "T", "G", "C", "N")}


def update_error_patterns(
    true_seq: str,
    pred_seq: str,
    mismatch_matrix: Dict[str, Dict[str, int]],
    deletion_bases: Dict[str, int],
    insertion_bases: Dict[str, int],
) -> None:
    true_aln, pred_aln = align_sequences_indel_aware(true_seq, pred_seq)
    L = min(len(true_aln), len(pred_aln))
    for i in range(L):
        t = _normalize_base(true_aln[i])
        p = _normalize_base(pred_aln[i])
        if true_aln[i] == "-" and pred_aln[i] != "-":
            insertion_bases[p] += 1
        elif true_aln[i] != "-" and pred_aln[i] == "-":
            deletion_bases[t] += 1
        elif true_aln[i] != "-" and pred_aln[i] != "-" and t != p:
            mismatch_matrix[t][p] += 1


def init_length_bins() -> List[int]:
    return [0, 50, 100, 200, 400, 800, 1200, 2000, 3000, 5000, 10000]


def count_hist(values: List[int], bins: List[int]) -> Dict[str, int]:
    hist = {}
    for i in range(len(bins) - 1):
        start = bins[i]
        end = bins[i + 1]
        hist[f"{start}-{end}"] = 0
    hist[f">={bins[-1]}"] = 0
    for v in values:
        placed = False
        for i in range(len(bins) - 1):
            if bins[i] <= v < bins[i + 1]:
                hist[f"{bins[i]}-{bins[i + 1]}"] += 1
                placed = True
                break
        if not placed:
            hist[f">={bins[-1]}"] += 1
    return hist


def init_position_bins() -> List[float]:
    return [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def count_relative_positions(values: List[float], bins: List[float]) -> Dict[str, int]:
    hist = {}
    for i in range(len(bins) - 1):
        start = int(bins[i] * 100)
        end = int(bins[i + 1] * 100)
        hist[f"{start}-{end}%"] = 0
    for v in values:
        for i in range(len(bins) - 1):
            if bins[i] <= v < bins[i + 1]:
                key = f"{int(bins[i] * 100)}-{int(bins[i + 1] * 100)}%"
                hist[key] += 1
                break
    return hist


def collect_deletion_positions(true_seq: str, pred_seq: str, out: List[float]) -> None:
    true_aln, pred_aln = align_sequences_indel_aware(true_seq, pred_seq)
    ref_pos = 0
    ref_len = sum(1 for ch in true_aln if ch != "-")
    if ref_len <= 0:
        return
    for t_char, p_char in zip(true_aln, pred_aln):
        if t_char != "-":
            ref_pos += 1
        if t_char != "-" and p_char == "-":
            rel = min(max((ref_pos - 1) / ref_len, 0.0), 1.0)
            out.append(rel)


def load_checkpoint_state(path: str) -> Dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                return state[key]
    if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
        return state
    raise ValueError(f"Unsupported checkpoint format: {path}")


def _parse_path_list(value: str | None) -> List[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]

def _infer_crf_state_len(num_classes: int, n_base: int) -> int:
    if n_base <= 1:
        raise ValueError("Cannot infer CTC-CRF state_len with n_base <= 1.")
    candidates = []
    if num_classes % (n_base + 1) == 0:
        base = num_classes / (n_base + 1)
        state_len = np.log(base) / np.log(n_base)
        if np.isclose(state_len, round(state_len)):
            candidates.append(int(round(state_len)))
    base = np.log(num_classes) / np.log(n_base) - 1
    if np.isclose(base, round(base)):
        candidates.append(int(round(base)))
    if candidates:
        return candidates[0]
    raise ValueError(
        "Unable to infer CTC-CRF state_len from num_classes and n_base. "
        "Please pass --ctc_crf_state_len or set CTC_CRF_STATE_LEN."
    )


def _ctc_crf_decode_batch(
    logits_tbc: torch.Tensor,
    input_lengths: torch.Tensor,
) -> List[List[int]]:
    logits_tbc = logits_tbc.float()
    decoded: List[List[int]] = []
    for idx, step_len in enumerate(input_lengths.tolist()):
        if step_len <= 0:
            decoded.append([])
            continue
        sample_logits = logits_tbc[:step_len, idx : idx + 1, :]
        decoded_ids = ctc_crf_decode(sample_logits, blank_idx=BLANK_IDX)[0]
        decoded_len = min(len(decoded_ids), step_len)
        decoded.append(decoded_ids[:decoded_len])
    return decoded




@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl_paths", default=None,
                    help="Comma-separated .jsonl.gz files or folders (uses text/bases fields).")
    ap.add_argument("--npy_paths", default=None,
                    help="Comma-separated folders or tokens_*.npy/reference_*.npy files (uses token/reference pairs).")
    ap.add_argument("--recursive", action="store_true",
                    help="Scan subfolders for .jsonl.gz or tokens/reference .npy inputs.")
    ap.add_argument("--token_offset", type=int, default=0,
                    help="Add this offset to each <|bwav:ID|> token in input signal_str (e.g. 0->128).")
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--beam_width", type=int, default=32)
    ap.add_argument("--ctc_crf_blank_score", type=float, default=2.0,
                    help="Blank score used by CTC-CRF head logits (keep consistent with training).")
    ap.add_argument("--decoder", choices=["auto", "ctc_viterbi", "koi", "ctc_crf"], default="auto",
                    help="Decoder to use. auto picks ctc_viterbi for CTC head and ctc_crf for CTC-CRF head.")
    ap.add_argument("--head_type", choices=["ctc", "ctc_crf"], default=None,
                    help="Override head type (default: infer from checkpoint).")
    ap.add_argument("--ctc_crf_state_len", type=int, default=None,
                    help="Override CTC-CRF state_len (default: infer from head or CTC_CRF_STATE_LEN env).")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="eval_out")
    ap.add_argument("--num_visualize", type=int, default=100)
    ap.add_argument("--max_len", type=int, default=200)
    ap.add_argument("--fastq_out", type=str, default=None)
    ap.add_argument("--fastq_q", type=int, default=20)
    ap.add_argument("--hidden_layer", type=int, default=-1,
                    help="Which backbone hidden layer to use when --feature_source hidden.")
    ap.add_argument("--learnable_fuse_last_n_layers", type=int, default=0,
                    help="If >0, learn a softmax-weighted fusion over the last N hidden layers (overrides --hidden_layer).")
    ap.add_argument("--feature_source", "--feature-source", choices=["hidden", "embedding"], default="hidden",
                    help="Use transformer hidden states or input embeddings as head input features.")
    ap.add_argument("--pre_head_type", choices=["auto", "none", "bilstm", "transformer", "tcn"], default="auto",
                    help="Optional module before CTC-CRF head. Default auto-infers from checkpoint.")
    ap.add_argument("--pre_head_transformer_nhead", type=int, default=8,
                    help="Attention heads for --pre_head_type transformer.")
    ap.add_argument("--acc_balanced", action="store_true",
                    help="Use Bonito balanced accuracy: (match - ins) / (match + mismatch + del).")
    ap.add_argument("--acc_min_coverage", type=float, default=0.0,
                    help="Minimum reference coverage required to count a read for accuracy.")
    args = ap.parse_args()
    if args.token_offset < 0:
        raise ValueError("--token_offset must be >= 0")

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if args.jsonl_paths and args.npy_paths:
        raise ValueError("Do not mix jsonl inputs with tokens/reference npy inputs in the same run.")

    if args.npy_paths:
        npy_paths = _parse_path_list(args.npy_paths)
        npy_pairs = scan_npy_pairs(npy_paths, group_by="file", recursive=args.recursive)
        if not npy_pairs:
            raise ValueError(f"No tokens/reference npy files found under: {args.npy_paths}")
        dataset = MultiNpySignalRefDataset(npy_pairs, token_offset=args.token_offset)
    else:
        if not args.jsonl_paths:
            raise ValueError("Provide --jsonl_paths or --npy_paths.")
        jsonl_paths = _parse_path_list(args.jsonl_paths)
        jsonl_files = scan_jsonl_files(jsonl_paths, group_by="file", recursive=args.recursive)
        if not jsonl_files:
            raise ValueError(f"No jsonl files found under: {args.jsonl_paths}")
        dataset = MultiJsonlSignalRefDataset(jsonl_files, token_offset=args.token_offset)

    state_dict = load_checkpoint_state(args.ckpt)
    head_config = infer_head_config_from_state_dict(state_dict)
    inferred_pre_head_type = infer_pre_head_type_from_state_dict(state_dict)
    head_type = args.head_type or head_config.get("head_type", "ctc")
    pre_head_type = args.pre_head_type
    if pre_head_type == "auto":
        pre_head_type = inferred_pre_head_type
        print(f"[Model] pre_head_type auto -> {pre_head_type}")
    elif pre_head_type != inferred_pre_head_type:
        print(
            "[Model][Warning] --pre_head_type overrides checkpoint inference: "
            f"arg={pre_head_type}, inferred={inferred_pre_head_type}"
        )
    n_base = len(ID2BASE) - 1
    state_len = args.ctc_crf_state_len
    decoder_mode = args.decoder
    if decoder_mode == "auto":
        decoder_mode = "ctc_viterbi" if head_type == "ctc" else "ctc_crf"

    if head_type == "ctc" and decoder_mode not in {"ctc_viterbi", "koi"}:
        raise ValueError("CTC head supports --decoder ctc_viterbi, koi, or auto.")
    if head_type == "ctc_crf" and decoder_mode not in {"ctc_crf", "koi"}:
        raise ValueError("CTC-CRF head supports --decoder ctc_crf, koi, or auto.")
    if decoder_mode == "ctc_crf":
        if head_type != "ctc_crf":
            raise ValueError("--decoder ctc_crf requires checkpoint/model head_type=ctc_crf.")
        if state_len is None:
            env_state_len = os.environ.get("CTC_CRF_STATE_LEN")
            if env_state_len is not None:
                state_len = int(env_state_len)
        if state_len is None:
            state_len = _infer_crf_state_len(head_config["num_classes"], n_base)
        os.environ["CTC_CRF_STATE_LEN"] = str(state_len)
    model = BasecallModel(
        model_path=args.model_name_or_path,
        num_classes=head_config["num_classes"],
        hidden_layer=args.hidden_layer,
        learnable_fuse_last_n_layers=args.learnable_fuse_last_n_layers,
        feature_source=args.feature_source,
        pre_head_type=pre_head_type,
        pre_head_transformer_nhead=args.pre_head_transformer_nhead,
        head_type=head_type,
        head_crf_blank_score=float(args.ctc_crf_blank_score),
        head_crf_n_base=n_base,
        head_crf_state_len=state_len,
        head_crf_expand_blanks=True,
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    _print_model_structure(model)

    collate_fn = create_collate_fn(model.tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    total_counts = {"match": 0, "mismatch": 0, "ins": 0, "del": 0}
    acc_scores: List[float] = []
    per_read_acc: List[float] = []
    exact_match = 0
    pred_seq_samples: List[str] = []
    ref_seq_samples: List[str] = []
    mismatch_matrix = _init_mismatch_matrix()
    deletion_bases = _init_base_counts()
    insertion_bases = _init_base_counts()
    pred_lengths: List[int] = []
    ref_lengths: List[int] = []
    deletion_positions: List[float] = []
    fastq_handle = None
    if args.fastq_out:
        os.makedirs(os.path.dirname(args.fastq_out) or ".", exist_ok=True)
        fastq_handle = open(args.fastq_out, "w", encoding="utf-8")

    read_idx = 0
    for batch in tqdm(loader, desc="[eval]", unit="batch"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits_btc = model(input_ids, attention_mask=attention_mask)

        logits_tbc = logits_btc.transpose(0, 1)
        input_lengths = resolve_input_lengths(
            input_ids,
            attention_mask=attention_mask,
            input_lengths=batch.get("input_lengths"),
        )
        if decoder_mode == "ctc_crf":
            pred_ids = _ctc_crf_decode_batch(logits_tbc, input_lengths)
        elif decoder_mode == "ctc_viterbi":
            pred_ids = ctc_viterbi_decode(logits_tbc, input_lengths=input_lengths, blank_idx=BLANK_IDX)
        else:
            pred_ids = koi_beam_search_decode(
                logits_tbc,
                beam_width=args.beam_width,
                beam_cut=100.0,
                scale=1.0,
                offset=0.0,
                blank_score=float(args.ctc_crf_blank_score),
                reverse=False,
                input_lengths=input_lengths,
            )
        ref_ids = batch["target_seqs"]

        acc = batch_bonito_accuracy(
            pred_ids,
            ref_ids,
            balanced=args.acc_balanced,
            min_coverage=args.acc_min_coverage,
        )
        acc_scores.append(acc)

        for pred, ref in zip(pred_ids, ref_ids):
            read_idx += 1
            pred_seq = _ids_to_bases(pred)
            ref_seq = _ids_to_bases(ref)
            counts = error_counts(pred_seq, ref_seq)
            total_counts = merge_counts(total_counts, counts)
            acc_read = cal_bonito_accuracy(
                pred_seq,
                ref_seq,
                balanced=args.acc_balanced,
                min_coverage=args.acc_min_coverage,
            )
            per_read_acc.append(acc_read)
            pred_lengths.append(len(pred_seq))
            ref_lengths.append(len(ref_seq))
            update_error_patterns(
                true_seq=ref_seq,
                pred_seq=pred_seq,
                mismatch_matrix=mismatch_matrix,
                deletion_bases=deletion_bases,
                insertion_bases=insertion_bases,
            )
            collect_deletion_positions(
                true_seq=ref_seq,
                pred_seq=pred_seq,
                out=deletion_positions,
            )
            if pred_seq == ref_seq:
                exact_match += 1
            if len(pred_seq_samples) < args.num_visualize:
                pred_seq_samples.append(pred_seq)
                ref_seq_samples.append(ref_seq)
            if fastq_handle is not None:
                q_char = chr(max(0, min(args.fastq_q, 93)) + 33)
                qstr = q_char * len(pred_seq)
                fastq_handle.write(f"@read_{read_idx}\n{pred_seq}\n+\n{qstr}\n")

    acc_avg = float(np.mean(acc_scores)) if acc_scores else 0.0
    ratios = counts_to_ratio(total_counts)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    length_bins = init_length_bins()
    position_bins = init_position_bins()
    summary = {
        "accuracy": acc_avg,
        "read_level_accuracy": float(np.mean(per_read_acc)) if per_read_acc else 0.0,
        "read_exact_match_rate": exact_match / max(len(per_read_acc), 1),
        "counts": total_counts,
        "ratios": ratios,
        "mismatch_matrix": mismatch_matrix,
        "deletion_bases": deletion_bases,
        "insertion_bases": insertion_bases,
        "lengths": {
            "pred": count_hist(pred_lengths, length_bins),
            "ref": count_hist(ref_lengths, length_bins),
            "bins": length_bins,
        },
        "deletion_position": {
            "relative_bins": count_relative_positions(deletion_positions, position_bins),
            "bin_edges": position_bins,
        },
        "num_reads": len(dataset),
    }
    summary_path = os.path.join(out_dir, "metrics.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if fastq_handle is not None:
        fastq_handle.close()

    if pred_seq_samples and ref_seq_samples:
        fig = plot_alignment_heatmap(
            pred_seq_samples,
            ref_seq_samples,
            max_reads=len(pred_seq_samples),
            max_len=args.max_len,
        )
        heatmap_path = os.path.join(out_dir, "heatmap.png")
        fig.savefig(heatmap_path, dpi=200, bbox_inches="tight")

        per_read_dir = os.path.join(out_dir, "aligned_reads")
        for idx, (pred_seq, ref_seq) in enumerate(zip(pred_seq_samples, ref_seq_samples), start=1):
            out_png = os.path.join(per_read_dir, f"read_{idx:03d}.png")
            title = f"read {idx}"
            plot_aligned_heatmap_png(
                true_seq=ref_seq,
                pred_seq=pred_seq,
                out_png=out_png,
                title=title,
                max_len=args.max_len,
            )

        seqs_path = os.path.join(out_dir, "sequences.jsonl")
        with open(seqs_path, "w", encoding="utf-8") as f:
            for idx, (pred_seq, ref_seq) in enumerate(zip(pred_seq_samples, ref_seq_samples), start=1):
                record = {
                    "index": idx,
                    "pred_seq": pred_seq,
                    "ref_seq": ref_seq,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[OK] metrics saved: {summary_path}")


if __name__ == "__main__":
    main()
