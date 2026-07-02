#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from tqdm.auto import tqdm


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[5]
TRAINING_DIR = REPO_ROOT / "src" / "poredlm" / "training"
STAGE4_DIR = TRAINING_DIR / "stage4_finetune"
for import_root in (STAGE4_DIR, TRAINING_DIR, REPO_ROOT / "src", REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from Basecalling.basecaller_v8_0420.model_dlm import BasecallModel  # noqa: E402


@dataclass
class ReadRecord:
    record_id: str
    input_ids: list[int]
    valid_len: int
    pattern: str
    base_sample_spans_rel: list
    modification_label: list[int]
    row_index: int
    meta: dict


def parse_positions(value: str | None) -> list[int] | None:
    if value is None or str(value).strip() == "":
        return None
    positions = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if any(pos <= 0 for pos in positions):
        raise ValueError(f"Modified base positions must be 1-based positive integers, got {positions}")
    return positions


def open_text(path: Path, mode: str = "rt"):
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))[:180]


def iter_records(jsonl_path: Path) -> Iterator[ReadRecord]:
    with open_text(jsonl_path, "rt") as handle:
        for row_index, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            meta = item.get("meta") or {}
            pattern = item.get("pattern") or meta.get("pattern")
            spans = meta.get("base_sample_spans_rel")
            labels = meta.get("modification_label")
            input_ids = item.get("input_ids")

            if not isinstance(pattern, str):
                raise ValueError(f"line {row_index + 1}: missing pattern string.")
            if not isinstance(spans, list):
                raise ValueError(f"line {row_index + 1}: missing meta.base_sample_spans_rel.")
            if not isinstance(labels, list):
                raise ValueError(f"line {row_index + 1}: missing meta.modification_label.")
            if not isinstance(input_ids, list):
                raise ValueError(f"line {row_index + 1}: missing input_ids list.")

            valid_len = int(meta.get("original_token_len", len(labels)))
            valid_len = max(0, min(valid_len, len(input_ids), len(labels)))
            record_id = str(item.get("id") or item.get("read_id") or meta.get("read_id") or f"row_{row_index}")
            yield ReadRecord(
                record_id=record_id,
                input_ids=[int(token_id) for token_id in input_ids],
                valid_len=valid_len,
                pattern=pattern,
                base_sample_spans_rel=spans,
                modification_label=[int(x) for x in labels],
                row_index=row_index,
                meta=meta,
            )


def make_batch(
    records: list[ReadRecord],
    *,
    pad_token_id: int,
    max_length: int | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    lengths = [min(len(record.input_ids), record.valid_len) for record in records]
    target_len = max(lengths) if max_length is None else min(max(lengths), int(max_length))
    input_ids = torch.full((len(records), target_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(records), target_len), dtype=torch.long)
    effective_lengths: list[int] = []

    for row, record in enumerate(records):
        valid_len = min(record.valid_len, target_len)
        ids = record.input_ids[:target_len]
        if ids:
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        if valid_len > 0:
            attention_mask[row, :valid_len] = 1
        effective_lengths.append(valid_len)
    return input_ids.to(device), attention_mask.to(device), effective_lengths


def forward_sequence_hidden(
    model: BasecallModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    backbone_chunk_size: int,
    embedding_source: str,
) -> torch.Tensor:
    if not hasattr(model.backbone, "context_encoder"):
        raise ValueError("The loaded Stage3 model does not expose backbone.context_encoder.")
    if embedding_source == "ode_hidden" and not hasattr(model.backbone, "elf_denoiser"):
        raise ValueError("The loaded Stage3 model does not expose backbone.elf_denoiser.")

    hidden_parts = []
    chunk_size = max(0, int(backbone_chunk_size))
    if chunk_size <= 0 or input_ids.shape[1] <= chunk_size:
        ranges = [(0, input_ids.shape[1])]
    else:
        ranges = [(start, min(start + chunk_size, input_ids.shape[1])) for start in range(0, input_ids.shape[1], chunk_size)]

    old_feature_source = model.feature_source
    try:
        model.feature_source = "context_hidden"
        for start, end in ranges:
            chunk_ids = input_ids[:, start:end]
            chunk_mask = attention_mask[:, start:end]
            context = model._forward_backbone_hidden(chunk_ids, attention_mask=chunk_mask)
            if embedding_source == "context_hidden":
                hidden_parts.append(context)
            elif embedding_source == "ode_hidden":
                hidden_parts.append(model._ode_from_context_hidden(context, attention_mask=chunk_mask))
            else:
                raise ValueError(f"Unsupported embedding_source={embedding_source!r}")
    finally:
        model.feature_source = old_feature_source
    return torch.cat(hidden_parts, dim=1)


def base_span_to_token_span(
    start_sample: int,
    end_sample: int,
    *,
    samples_per_token: int,
    token_count: int,
) -> tuple[int, int]:
    token_start = max(0, int(start_sample) // samples_per_token)
    token_end = min(token_count, int(math.ceil(float(end_sample) / samples_per_token)))
    return token_start, max(token_start, token_end)


def build_offset_modification_label(
    record: ReadRecord,
    *,
    token_count: int,
    samples_per_token: int,
    base_span_offset: int,
    modified_base_positions: list[int] | None,
) -> list[int]:
    if modified_base_positions is None:
        modified_base_positions = record.meta.get("modification_base_positions_1based")
    if not isinstance(modified_base_positions, list):
        return record.modification_label[:token_count]

    labels = [0] * token_count
    span_shift = int(base_span_offset)
    for base_position in modified_base_positions:
        base_index = int(base_position) - 1
        span_index = base_index + span_shift
        if span_index < 0 or span_index >= len(record.base_sample_spans_rel):
            continue
        span = record.base_sample_spans_rel[span_index]
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            continue
        token_start, token_end = base_span_to_token_span(
            int(span[0]),
            int(span[1]),
            samples_per_token=samples_per_token,
            token_count=token_count,
        )
        for token_index in range(token_start, token_end):
            labels[token_index] = 1
    return labels


def collect_c_token_embeddings(
    record: ReadRecord,
    sequence_hidden: np.ndarray,
    *,
    samples_per_token: int,
    base_span_offset: int,
    modified_base_positions: list[int] | None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rows = []
    labels = []
    points: list[dict] = []
    token_count = min(record.valid_len, sequence_hidden.shape[0], len(record.modification_label))
    modified_base_set = set(int(pos) for pos in modified_base_positions) if modified_base_positions is not None else None
    fallback_aligned_label = None
    if modified_base_set is None:
        fallback_aligned_label = build_offset_modification_label(
            record,
            token_count=token_count,
            samples_per_token=samples_per_token,
            base_span_offset=base_span_offset,
            modified_base_positions=None,
        )

    span_shift = int(base_span_offset)
    for base_index in range(len(record.pattern)):
        base = record.pattern[base_index].upper()
        if base != "C":
            continue
        span_index = base_index + span_shift
        if span_index < 0 or span_index >= len(record.base_sample_spans_rel):
            continue
        span = record.base_sample_spans_rel[span_index]
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            continue
        token_start, token_end = base_span_to_token_span(
            int(span[0]),
            int(span[1]),
            samples_per_token=samples_per_token,
            token_count=token_count,
        )
        for token_index in range(token_start, token_end):
            if modified_base_set is not None:
                label = int((base_index + 1) in modified_base_set)
            else:
                label = int(fallback_aligned_label[token_index] == 1)
            rows.append(sequence_hidden[token_index])
            labels.append(label)
            points.append(
                {
                    "base_position_1based": base_index + 1,
                    "base_span_position_1based": span_index + 1,
                    "base_span_offset": base_span_offset,
                    "token_position": token_index,
                    "modified": label,
                    "sample_span": [int(span[0]), int(span[1])],
                }
            )

    if not rows:
        return np.empty((0, sequence_hidden.shape[-1]), dtype=np.float32), np.empty((0,), dtype=np.int64), points
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int64), points


def summarize_modified_base_mapping(
    record: ReadRecord,
    *,
    token_count: int,
    samples_per_token: int,
    base_span_offset: int,
    modified_base_positions: list[int] | None,
) -> list[dict]:
    if modified_base_positions is None:
        modified_base_positions = record.meta.get("modification_base_positions_1based")
    if not isinstance(modified_base_positions, list):
        return []

    rows = []
    span_shift = int(base_span_offset)
    for base_position in modified_base_positions:
        base_index = int(base_position) - 1
        span_index = base_index + span_shift
        base = record.pattern[base_index].upper() if 0 <= base_index < len(record.pattern) else None
        plotted_token_count = sum(
            1
            for point in collect_points_for_base(
                record,
                base_index=base_index,
                span_index=span_index,
                token_count=token_count,
                samples_per_token=samples_per_token,
            )
        )
        row = {
            "base_position_1based": int(base_position),
            "base": base,
            "base_span_position_1based": span_index + 1,
            "base_span_offset": int(base_span_offset),
            "in_pattern": 0 <= base_index < len(record.pattern),
            "in_base_sample_spans_rel": 0 <= span_index < len(record.base_sample_spans_rel),
            "plotted_token_count": int(plotted_token_count),
        }
        if row["in_base_sample_spans_rel"]:
            span = record.base_sample_spans_rel[span_index]
            if isinstance(span, (list, tuple)) and len(span) == 2:
                token_start, token_end = base_span_to_token_span(
                    int(span[0]),
                    int(span[1]),
                    samples_per_token=samples_per_token,
                    token_count=token_count,
                )
                row["sample_span"] = [int(span[0]), int(span[1])]
                row["token_span"] = [int(token_start), int(token_end)]
                row["token_count"] = int(max(0, token_end - token_start))
        rows.append(row)
    return rows


def collect_points_for_base(
    record: ReadRecord,
    *,
    base_index: int,
    span_index: int,
    token_count: int,
    samples_per_token: int,
) -> list[int]:
    if base_index < 0 or base_index >= len(record.pattern):
        return []
    if record.pattern[base_index].upper() != "C":
        return []
    if span_index < 0 or span_index >= len(record.base_sample_spans_rel):
        return []
    span = record.base_sample_spans_rel[span_index]
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        return []
    token_start, token_end = base_span_to_token_span(
        int(span[0]),
        int(span[1]),
        samples_per_token=samples_per_token,
        token_count=token_count,
    )
    return list(range(token_start, token_end))


def pca_2d(x: np.ndarray) -> np.ndarray:
    if x.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)
    if x.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float32)
    centered = x - x.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:2].T
        coords = centered @ components
        if coords.shape[1] == 1:
            coords = np.concatenate([coords, np.zeros_like(coords)], axis=1)
        return coords[:, :2].astype(np.float32)
    except np.linalg.LinAlgError:
        return np.zeros((x.shape[0], 2), dtype=np.float32)


def plot_read_distribution(
    record: ReadRecord,
    embeddings: np.ndarray,
    labels: np.ndarray,
    points: list[dict],
    output_path: Path,
    *,
    annotate_modified: bool,
    base_span_offset: int,
    modified_base_mapping: list[dict],
    embedding_source: str,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    coords = pca_2d(embeddings)
    norms = np.linalg.norm(embeddings, axis=1) if embeddings.size else np.empty((0,), dtype=np.float32)
    modified_mask = labels == 1
    unmodified_mask = labels == 0

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5.6))
    ax = axes[0]
    if unmodified_mask.any():
        ax.scatter(coords[unmodified_mask, 0], coords[unmodified_mask, 1], s=18, alpha=0.65, color="#1f77b4", label=f"C unmodified tokens ({int(unmodified_mask.sum())})")
    if modified_mask.any():
        ax.scatter(coords[modified_mask, 0], coords[modified_mask, 1], s=48, alpha=0.95, color="#d62728", edgecolors="black", linewidths=0.4, marker="*", label=f"C modified tokens ({int(modified_mask.sum())})")
    if annotate_modified and modified_mask.any():
        for idx in np.flatnonzero(modified_mask):
            point = points[int(idx)]
            ax.annotate(
                f"B{point['base_position_1based']}:T{point['token_position']}",
                xy=(coords[idx, 0], coords[idx, 1]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color="#7f1d1d",
            )
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title(f"{embedding_source} PCA for C-base tokens")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax2 = axes[1]
    if unmodified_mask.any():
        ax2.hist(norms[unmodified_mask], bins=30, alpha=0.65, color="#1f77b4", label="C unmodified")
    if modified_mask.any():
        ax2.hist(norms[modified_mask], bins=15, alpha=0.8, color="#d62728", label="C modified")
        ax2.scatter(norms[modified_mask], np.zeros(int(modified_mask.sum())), s=35, color="#d62728", marker="|")
    ax2.set_xlabel(f"{embedding_source} L2 norm")
    ax2.set_ylabel("count")
    ax2.set_title("Embedding norm distribution")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best")

    raw_signal_len = record.meta.get("raw_signal_len")
    original_token_len = record.meta.get("original_token_len")
    fig.suptitle(
        f"{record.record_id} | C tokens={len(labels)} | modified C tokens={int(modified_mask.sum())} "
        f"| raw_signal_len={raw_signal_len} | original_token_len={original_token_len} "
        f"| base_span_offset={base_span_offset}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    modified_points = [points[int(idx)] for idx in np.flatnonzero(modified_mask)]
    return {
        "id": record.record_id,
        "row_index": record.row_index,
        "output_png": str(output_path),
        "num_c_tokens": int(len(labels)),
        "num_modified_c_tokens": int(modified_mask.sum()),
        "num_unmodified_c_tokens": int(unmodified_mask.sum()),
        "base_span_offset": int(base_span_offset),
        "embedding_source": embedding_source,
        "modified_base_mapping": modified_base_mapping,
        "modified_points": modified_points,
    }


def build_model(args: argparse.Namespace, device: torch.device) -> BasecallModel:
    model = BasecallModel(
        model_path=args.model_name_or_path,
        feature_source="context_hidden",
        freeze_backbone=True,
        pre_head_type="none",
        head_type="ctc",
        backbone_chunk_size=args.backbone_chunk_size,
        elf_ode_steps=args.elf_ode_steps,
        elf_ode_start_t=args.elf_ode_start_t,
        elf_self_cond_cfg_scale=args.elf_self_cond_cfg_scale,
    )
    model.eval().to(device)
    if args.dtype == "float16":
        model = model.half()
    elif args.dtype == "bfloat16":
        model = model.to(dtype=torch.bfloat16)
    elif args.dtype == "float32":
        model = model.float()
    return model


def process_batch(
    args: argparse.Namespace,
    model: BasecallModel,
    device: torch.device,
    records: list[ReadRecord],
    summary_handle,
) -> int:
    input_ids, attention_mask, effective_lengths = make_batch(
        records,
        pad_token_id=args.pad_token_id,
        max_length=args.max_length,
        device=device,
    )
    with torch.inference_mode():
        if args.debug_compare_context_ode:
            context_hidden_t = forward_sequence_hidden(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                backbone_chunk_size=args.backbone_chunk_size,
                embedding_source="context_hidden",
            ).float()
            ode_hidden_t = forward_sequence_hidden(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                backbone_chunk_size=args.backbone_chunk_size,
                embedding_source="ode_hidden",
            ).float()
            sequence_hidden_t = context_hidden_t if args.embedding_source == "context_hidden" else ode_hidden_t
            delta_l2_t = torch.linalg.vector_norm(ode_hidden_t - context_hidden_t, dim=-1)
            delta_cos_t = 1.0 - torch.nn.functional.cosine_similarity(context_hidden_t, ode_hidden_t, dim=-1, eps=1e-8)
            sequence_hidden = sequence_hidden_t.detach().cpu().numpy()
            delta_l2 = delta_l2_t.detach().cpu().numpy()
            delta_cos = delta_cos_t.detach().cpu().numpy()
        else:
            sequence_hidden = forward_sequence_hidden(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                backbone_chunk_size=args.backbone_chunk_size,
                embedding_source=args.embedding_source,
            ).float().detach().cpu().numpy()
            delta_l2 = None
            delta_cos = None

    written = 0
    for idx, record in enumerate(records):
        valid_len = min(effective_lengths[idx], sequence_hidden.shape[1])
        embeddings, labels, points = collect_c_token_embeddings(
            record,
            sequence_hidden[idx, :valid_len],
            samples_per_token=args.samples_per_token,
            base_span_offset=args.base_span_offset,
            modified_base_positions=args.modified_base_positions,
        )
        modified_base_mapping = summarize_modified_base_mapping(
            record,
            token_count=min(valid_len, len(record.modification_label)),
            samples_per_token=args.samples_per_token,
            base_span_offset=args.base_span_offset,
            modified_base_positions=args.modified_base_positions,
        )
        if embeddings.shape[0] == 0:
            print(f"Skip {record.record_id}: no C-base token embeddings found.")
            continue
        output_png = Path(args.output_dir) / f"{safe_name(record.record_id)}.png"
        summary = plot_read_distribution(
            record,
            embeddings,
            labels,
            points,
            output_png,
            annotate_modified=not args.no_annotate_modified,
            base_span_offset=args.base_span_offset,
            modified_base_mapping=modified_base_mapping,
            embedding_source=args.embedding_source,
        )
        if delta_l2 is not None and delta_cos is not None:
            valid_delta_l2 = delta_l2[idx, :valid_len]
            valid_delta_cos = delta_cos[idx, :valid_len]
            summary["context_ode_delta_mean_l2"] = float(valid_delta_l2.mean()) if valid_delta_l2.size else None
            summary["context_ode_delta_max_l2"] = float(valid_delta_l2.max()) if valid_delta_l2.size else None
            summary["context_ode_delta_mean_cosine_distance"] = float(valid_delta_cos.mean()) if valid_delta_cos.size else None
            summary["context_ode_delta_max_cosine_distance"] = float(valid_delta_cos.max()) if valid_delta_cos.size else None
        summary_handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
        written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-read context_hidden or ode_hidden PCA distributions for tokens covered by C bases."
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--jsonl", required=True, help="Input jsonl/jsonl.gz containing input_ids, pattern, base_sample_spans_rel, and modification_label.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1600)
    parser.add_argument("--pad-token-id", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--backbone-chunk-size", type=int, default=1600)
    parser.add_argument("--elf-ode-steps", type=int, default=4)
    parser.add_argument("--elf-ode-start-t", type=float, default=0.85)
    parser.add_argument("--elf-self-cond-cfg-scale", type=float, default=1.0)
    parser.add_argument(
        "--embedding-source",
        choices=("context_hidden", "ode_hidden"),
        default="ode_hidden",
        help="Which Stage3 representation to plot for C-base tokens.",
    )
    parser.add_argument(
        "--debug-compare-context-ode",
        action="store_true",
        help="Also compute per-read context_hidden vs ode_hidden delta stats in the summary JSONL.",
    )
    parser.add_argument("--samples-per-token", type=int, default=5)
    parser.add_argument(
        "--base-span-offset",
        type=int,
        default=1,
        help=(
            "Offset from pattern base index to base_sample_spans_rel index. "
            "offset=0 means pattern[0] uses spans[0]; offset=3 means pattern[0] uses spans[3]. "
            "Negative values move toward earlier spans."
        ),
    )
    parser.add_argument(
        "--modified-base-positions",
        type=parse_positions,
        default=parse_positions("14,33,52,71,90,109,128"),
        help=(
            "Comma-separated 1-based modified base positions. These are used to rebuild "
            "the modified token labels with --base-span-offset. Use an empty string to "
            "fall back to meta.modification_base_positions_1based or meta.modification_label."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-annotate-modified", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = build_model(args, device)

    total = 0
    written = 0
    summary_path = output_dir / "c_base_ode_embedding_distribution_summary.jsonl"
    with summary_path.open("w", encoding="utf-8") as summary_handle:
        batch: list[ReadRecord] = []
        iterator = iter_records(Path(args.jsonl))
        pbar = tqdm(desc=f"plotting C-base {args.embedding_source} embeddings", unit="read")
        for record in iterator:
            if args.limit is not None and total >= args.limit:
                break
            batch.append(record)
            total += 1
            if len(batch) < args.batch_size:
                continue
            written += process_batch(args, model, device, batch, summary_handle)
            pbar.update(len(batch))
            batch = []
        if batch:
            written += process_batch(args, model, device, batch, summary_handle)
            pbar.update(len(batch))
        pbar.close()

    print(f"Processed reads: {total}")
    print(f"Written plots: {written}")
    print(f"Output dir: {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
