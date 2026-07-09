#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm


THIS_FILE = Path(__file__).resolve()
SCRIPT_DIR = THIS_FILE.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_c_modification_embedding_distribution import (  # noqa: E402
    BasecallModel,
    ReadRecord,
    build_model,
    forward_sequence_hidden,
    iter_records,
    make_batch,
    normalize_embedding_source,
)
from plot_lb06_lb07_same_site_c_embedding_distribution import (  # noqa: E402
    DEFAULT_LB06_JSONL,
    DEFAULT_LB07_JSONL,
    parse_maybe_json,
    safe_name,
    sample_span_to_center_token,
    sample_span_to_token_span,
    sequence_key_for_record,
)


def parse_int_list(value: str) -> list[int]:
    items = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        number = int(part)
        if number <= 0:
            raise ValueError("--read-depths must contain positive integers.")
        items.append(number)
    if not items:
        raise ValueError("--read-depths must contain at least one positive integer.")
    return sorted(set(items))


def load_limited_records(path: str, *, limit_reads: int | None) -> list[ReadRecord]:
    records: list[ReadRecord] = []
    for record in iter_records(Path(path)):
        if limit_reads is not None and len(records) >= limit_reads:
            break
        records.append(record)
    return records


def group_records_by_sequence(records: list[ReadRecord], *, sequence_key_mode: str) -> dict[str, list[ReadRecord]]:
    grouped: dict[str, list[ReadRecord]] = defaultdict(list)
    for record in records:
        grouped[sequence_key_for_record(record, sequence_key_mode)].append(record)
    return dict(grouped)


def sample_records_by_sequence(
    records: list[ReadRecord],
    *,
    sequence_key_mode: str,
    depth: int,
    rng: random.Random,
) -> list[ReadRecord]:
    sampled: list[ReadRecord] = []
    for _sequence_key, seq_records in sorted(group_records_by_sequence(records, sequence_key_mode=sequence_key_mode).items()):
        if len(seq_records) <= depth:
            sampled.extend(seq_records)
        else:
            sampled.extend(seq_records[index] for index in sorted(rng.sample(range(len(seq_records)), depth)))
    return sampled


def token_indices_for_span(
    span: Any,
    *,
    token_pool: str,
    samples_per_token: int,
    token_count: int,
) -> list[int]:
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        return []
    start_sample, end_sample = span
    if start_sample is None or end_sample is None:
        return []
    try:
        start_sample = int(start_sample)
        end_sample = int(end_sample)
    except (TypeError, ValueError):
        return []
    if token_pool == "center":
        center_token = sample_span_to_center_token(
            start_sample,
            end_sample,
            samples_per_token=samples_per_token,
            token_count=token_count,
        )
        return [] if center_token is None else [int(center_token)]
    token_start, token_end = sample_span_to_token_span(
        start_sample,
        end_sample,
        samples_per_token=samples_per_token,
        token_count=token_count,
    )
    return list(range(token_start, token_end))


def pool_token_embeddings(hidden: np.ndarray, token_indices: list[int], *, token_pool: str) -> np.ndarray | None:
    if not token_indices:
        return None
    token_indices = sorted(set(int(index) for index in token_indices if 0 <= int(index) < hidden.shape[0]))
    if not token_indices:
        return None
    token_embeddings = hidden[token_indices]
    if token_pool in {"mean", "center"}:
        return token_embeddings.mean(axis=0).astype(np.float32)
    if token_pool == "max":
        return token_embeddings.max(axis=0).astype(np.float32)
    raise ValueError(f"Unsupported token_pool: {token_pool}")


def collect_site_embeddings(
    args: argparse.Namespace,
    *,
    model: BasecallModel,
    device: torch.device,
    records: list[ReadRecord],
    dataset_name: str,
    embedding_source: str,
) -> tuple[dict[str, dict[int, list[np.ndarray]]], dict[str, str], dict[str, int]]:
    site_embeddings: dict[str, dict[int, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    sequence_refs: dict[str, str] = {}
    valid_record_counts: dict[str, int] = defaultdict(int)
    batch: list[ReadRecord] = []
    pbar = tqdm(total=len(records), desc=f"extracting {dataset_name} site embeddings", unit="read")

    def flush_batch() -> None:
        nonlocal batch
        if not batch:
            return
        input_ids, attention_mask, effective_lengths = make_batch(
            batch,
            pad_token_id=args.pad_token_id,
            max_length=args.max_length,
            device=device,
        )
        with torch.inference_mode():
            hidden = forward_sequence_hidden(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                backbone_chunk_size=args.backbone_chunk_size,
                embedding_source=embedding_source,
            ).float().detach().cpu().numpy()

        for idx, record in enumerate(batch):
            sequence_key = sequence_key_for_record(record, args.sequence_key)
            meta = record.meta or {}
            ref = meta.get("ref")
            spans = parse_maybe_json(meta.get("base_sample_span_ref"))
            if not isinstance(ref, str) or not isinstance(spans, list):
                continue
            token_count = min(int(effective_lengths[idx]), hidden.shape[1])
            if token_count <= 0:
                continue
            sequence_refs.setdefault(sequence_key, ref)
            valid_record_counts[sequence_key] += 1
            max_base_count = min(len(ref), len(spans))
            record_hidden = hidden[idx, :token_count]
            for base_index in range(max_base_count):
                token_indices = token_indices_for_span(
                    spans[base_index],
                    token_pool=args.token_pool,
                    samples_per_token=args.samples_per_token,
                    token_count=token_count,
                )
                pooled = pool_token_embeddings(record_hidden, token_indices, token_pool=args.token_pool)
                if pooled is None:
                    continue
                site_embeddings[sequence_key][base_index + 1].append(pooled)
        pbar.update(len(batch))
        batch = []

    for record in records:
        batch.append(record)
        if len(batch) >= args.batch_size:
            flush_batch()
    flush_batch()
    pbar.close()
    return site_embeddings, sequence_refs, dict(valid_record_counts)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float(1.0 - float(np.dot(a, b) / denom))


def site_metric(
    lb06_values: list[np.ndarray],
    lb07_values: list[np.ndarray],
    *,
    metric: str,
) -> tuple[float, dict[str, Any]]:
    if not lb06_values or not lb07_values:
        return float("nan"), {}
    x06 = np.stack(lb06_values, axis=0).astype(np.float32)
    x07 = np.stack(lb07_values, axis=0).astype(np.float32)
    mean06 = x06.mean(axis=0)
    mean07 = x07.mean(axis=0)
    diff = mean06 - mean07
    l2 = float(np.linalg.norm(diff))
    cos = cosine_distance(mean06, mean07)
    norm_diff = float(np.linalg.norm(mean06) - np.linalg.norm(mean07))
    pooled_std = float(np.sqrt(0.5 * (float(x06.var(axis=0).mean()) + float(x07.var(axis=0).mean()))) + 1e-8)
    cohen_l2 = float(l2 / pooled_std)
    if metric == "l2":
        value = l2
    elif metric == "cosine":
        value = cos
    elif metric == "norm-diff":
        value = norm_diff
    elif metric == "abs-norm-diff":
        value = abs(norm_diff)
    elif metric == "cohen-l2":
        value = cohen_l2
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    details = {
        "lb06_count": int(x06.shape[0]),
        "lb07_count": int(x07.shape[0]),
        "l2": l2,
        "cosine": cos,
        "norm_diff": norm_diff,
        "abs_norm_diff": abs(norm_diff),
        "cohen_l2": cohen_l2,
    }
    return value, details


def build_heatmap_matrix(
    lb06_sites: dict[str, dict[int, list[np.ndarray]]],
    lb07_sites: dict[str, dict[int, list[np.ndarray]]],
    sequence_refs: dict[str, str],
    *,
    metric: str,
    min_reads_per_site: int,
) -> tuple[list[str], np.ndarray, dict[str, dict[int, dict[str, Any]]]]:
    sequence_keys = sorted(set(lb06_sites) & set(lb07_sites))
    max_len = max((len(sequence_refs.get(sequence_key, "")) for sequence_key in sequence_keys), default=0)
    matrix = np.full((len(sequence_keys), max_len), np.nan, dtype=np.float32)
    details_by_sequence: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row_index, sequence_key in enumerate(sequence_keys):
        seq_len = len(sequence_refs.get(sequence_key, ""))
        for base_position in range(1, seq_len + 1):
            lb06_values = lb06_sites.get(sequence_key, {}).get(base_position, [])
            lb07_values = lb07_sites.get(sequence_key, {}).get(base_position, [])
            if len(lb06_values) < min_reads_per_site or len(lb07_values) < min_reads_per_site:
                continue
            value, details = site_metric(lb06_values, lb07_values, metric=metric)
            matrix[row_index, base_position - 1] = value
            details_by_sequence[sequence_key][base_position] = details
    return sequence_keys, matrix, details_by_sequence


def plot_heatmaps(
    matrices: dict[int, tuple[list[str], np.ndarray]],
    output_png: Path,
    *,
    title: str,
    metric: str,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depths = sorted(matrices)
    if not depths:
        raise ValueError("No heatmap matrices to plot.")
    ncols = len(depths)
    fig_width = max(9.0, 4.8 * ncols)
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(fig_width, 5.5), squeeze=False)
    finite_values = np.concatenate(
        [matrix[np.isfinite(matrix)] for _keys, matrix in matrices.values() if np.isfinite(matrix).any()],
        axis=0,
    ) if any(np.isfinite(matrix).any() for _keys, matrix in matrices.values()) else np.asarray([], dtype=np.float32)
    auto_vmin = float(np.nanpercentile(finite_values, 2)) if finite_values.size else 0.0
    auto_vmax = float(np.nanpercentile(finite_values, 98)) if finite_values.size else 1.0
    plot_vmin = auto_vmin if vmin is None else vmin
    plot_vmax = auto_vmax if vmax is None else vmax
    if math.isclose(plot_vmin, plot_vmax):
        plot_vmax = plot_vmin + 1e-6

    for col_index, depth in enumerate(depths):
        sequence_keys, matrix = matrices[depth]
        ax = axes[0][col_index]
        masked = np.ma.masked_invalid(matrix)
        im = ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, vmin=plot_vmin, vmax=plot_vmax)
        ax.set_title(f"N={depth} reads/seq")
        ax.set_xlabel("base position")
        if col_index == 0:
            ax.set_ylabel("sequence")
            ax.set_yticks(np.arange(len(sequence_keys)))
            ax.set_yticklabels(sequence_keys, fontsize=7)
        else:
            ax.set_yticks([])
        if matrix.shape[1] > 0:
            tick_count = min(8, matrix.shape[1])
            ticks = np.linspace(0, matrix.shape[1] - 1, tick_count, dtype=int)
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(tick + 1) for tick in ticks], fontsize=7)
        cbar = fig.colorbar(im, ax=ax, shrink=0.78)
        cbar.set_label(metric)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def save_matrix_tsv(path: Path, sequence_keys: list[str], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        max_pos = matrix.shape[1]
        handle.write("sequence_key\t" + "\t".join(str(pos) for pos in range(1, max_pos + 1)) + "\n")
        for row_index, sequence_key in enumerate(sequence_keys):
            values = []
            for value in matrix[row_index]:
                values.append("" if not np.isfinite(value) else f"{float(value):.8g}")
            handle.write(sequence_key + "\t" + "\t".join(values) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot LB06-vs-LB07 per-base-site embedding difference heatmaps. Each base site can cover "
            "multiple signal tokens; token embeddings are pooled per read before comparing datasets."
        )
    )
    parser.add_argument("--model-name-or-path", required=True, help="Stage3 HF DLM model directory.")
    parser.add_argument("--lb07-jsonl", default=DEFAULT_LB07_JSONL)
    parser.add_argument("--lb06-jsonl", default=DEFAULT_LB06_JSONL)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embedding-source", choices=("bert", "dlm", "context_hidden", "ode_hidden"), default="bert")
    parser.add_argument("--sequence-key", choices=("auto", "label", "ref", "seq"), default="label")
    parser.add_argument("--read-depths", default="5,10,20,50", help="Comma-separated number of reads per sequence per dataset.")
    parser.add_argument("--limit-lb07-reads", type=int, default=None)
    parser.add_argument("--limit-lb06-reads", type=int, default=None)
    parser.add_argument("--samples-per-token", type=int, default=5)
    parser.add_argument("--token-pool", choices=("mean", "center", "max"), default="mean")
    parser.add_argument("--metric", choices=("l2", "cosine", "norm-diff", "abs-norm-diff", "cohen-l2"), default="l2")
    parser.add_argument("--min-reads-per-site", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2000)
    parser.add_argument("--pad-token-id", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--backbone-chunk-size", type=int, default=2000)
    parser.add_argument("--elf-ode-steps", type=int, default=4)
    parser.add_argument("--elf-ode-start-t", type=float, default=0.95)
    parser.add_argument("--elf-self-cond-cfg-scale", type=float, default=1.0)
    parser.add_argument("--cmap", default="inferno")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    read_depths = parse_int_list(args.read_depths)
    rng = random.Random(args.seed)
    embedding_source = normalize_embedding_source(args.embedding_source)
    embedding_label = "bert" if embedding_source == "context_hidden" else "dlm"
    device = torch.device(args.device)

    lb07_records_all = load_limited_records(args.lb07_jsonl, limit_reads=args.limit_lb07_reads)
    lb06_records_all = load_limited_records(args.lb06_jsonl, limit_reads=args.limit_lb06_reads)
    if not lb07_records_all:
        raise ValueError(f"No LB07 records loaded from {args.lb07_jsonl}")
    if not lb06_records_all:
        raise ValueError(f"No LB06 records loaded from {args.lb06_jsonl}")

    model = build_model(args, device, embedding_source)
    matrices: dict[int, tuple[list[str], np.ndarray]] = {}
    depth_summaries: dict[str, Any] = {}

    for depth in read_depths:
        depth_rng = random.Random(rng.randint(0, 2**31 - 1))
        lb07_records = sample_records_by_sequence(
            lb07_records_all,
            sequence_key_mode=args.sequence_key,
            depth=depth,
            rng=depth_rng,
        )
        lb06_records = sample_records_by_sequence(
            lb06_records_all,
            sequence_key_mode=args.sequence_key,
            depth=depth,
            rng=depth_rng,
        )
        lb07_sites, lb07_refs, lb07_counts = collect_site_embeddings(
            args,
            model=model,
            device=device,
            records=lb07_records,
            dataset_name=f"LB07_N{depth}",
            embedding_source=embedding_source,
        )
        lb06_sites, lb06_refs, lb06_counts = collect_site_embeddings(
            args,
            model=model,
            device=device,
            records=lb06_records,
            dataset_name=f"LB06_N{depth}",
            embedding_source=embedding_source,
        )
        sequence_refs = dict(lb07_refs)
        sequence_refs.update(lb06_refs)
        sequence_keys, matrix, details = build_heatmap_matrix(
            lb06_sites,
            lb07_sites,
            sequence_refs,
            metric=args.metric,
            min_reads_per_site=args.min_reads_per_site,
        )
        matrices[depth] = (sequence_keys, matrix)
        matrix_tsv = output_dir / f"LB06_vs_LB07_site_embedding_{embedding_label}_{args.metric}_N{depth}.tsv"
        save_matrix_tsv(matrix_tsv, sequence_keys, matrix)
        finite = matrix[np.isfinite(matrix)]
        depth_summaries[str(depth)] = {
            "lb07_records": len(lb07_records),
            "lb06_records": len(lb06_records),
            "lb07_valid_records_by_sequence": lb07_counts,
            "lb06_valid_records_by_sequence": lb06_counts,
            "sequence_count": len(sequence_keys),
            "site_count": int(matrix.size),
            "valid_site_count": int(finite.size),
            "metric_mean": float(finite.mean()) if finite.size else None,
            "metric_p95": float(np.quantile(finite, 0.95)) if finite.size else None,
            "matrix_tsv": str(matrix_tsv),
            "site_details": {
                sequence_key: {
                    str(position): detail
                    for position, detail in sorted(position_map.items())
                }
                for sequence_key, position_map in sorted(details.items())
            },
        }

    output_png = output_dir / f"LB06_vs_LB07_site_embedding_{embedding_label}_{args.metric}_heatmap.png"
    plot_heatmaps(
        matrices,
        output_png,
        title=(
            f"LB06 vs LB07 per-site embedding difference | {embedding_label} | "
            f"metric={args.metric}, token_pool={args.token_pool}"
        ),
        metric=args.metric,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
    )

    summary = {
        "lb07_jsonl": args.lb07_jsonl,
        "lb06_jsonl": args.lb06_jsonl,
        "model_name_or_path": args.model_name_or_path,
        "embedding_source_requested": args.embedding_source,
        "embedding_source_normalized": embedding_source,
        "embedding_label": embedding_label,
        "sequence_key": args.sequence_key,
        "read_depths": read_depths,
        "samples_per_token": args.samples_per_token,
        "token_pool": args.token_pool,
        "metric": args.metric,
        "min_reads_per_site": args.min_reads_per_site,
        "seed": args.seed,
        "heatmap_png": str(output_png),
        "depths": depth_summaries,
    }
    summary_path = output_dir / f"LB06_vs_LB07_site_embedding_{embedding_label}_{args.metric}_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Embedding source: {args.embedding_source} -> {embedding_source}")
    print(f"Read depths: {','.join(str(depth) for depth in read_depths)}")
    print(f"Heatmap: {output_png}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
