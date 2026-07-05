#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import gzip
import json
import random
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
    c_modification_label: list[int]
    valid_len: int
    row_index: int
    meta: dict


def open_text(path: Path, mode: str = "rt"):
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))[:180]


def normalize_embedding_source(value: str) -> str:
    value = str(value).strip()
    aliases = {
        "bert": "context_hidden",
        "bert_context": "context_hidden",
        "context": "context_hidden",
        "context_hidden": "context_hidden",
        "dlm": "ode_hidden",
        "ode": "ode_hidden",
        "ode_hidden": "ode_hidden",
    }
    if value not in aliases:
        raise ValueError(f"Unsupported embedding source {value!r}; use bert, dlm, context_hidden, or ode_hidden.")
    return aliases[value]


def iter_records(jsonl_path: Path) -> Iterator[ReadRecord]:
    with open_text(jsonl_path, "rt") as handle:
        for row_index, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            meta = item.get("meta") or {}
            input_ids = item.get("input_ids")
            labels = item.get("c_modification_label")
            if labels is None:
                labels = meta.get("c_modification_label")

            if not isinstance(input_ids, list):
                raise ValueError(f"line {row_index + 1}: missing input_ids list.")
            if not isinstance(labels, list):
                raise ValueError(f"line {row_index + 1}: missing c_modification_label list.")

            valid_len = int(meta.get("original_token_len", len(labels)))
            valid_len = max(0, min(valid_len, len(input_ids), len(labels)))
            record_id = str(item.get("id") or item.get("read_id") or meta.get("read_id") or f"row_{row_index}")
            yield ReadRecord(
                record_id=record_id,
                input_ids=[int(token_id) for token_id in input_ids],
                c_modification_label=[int(label) for label in labels],
                valid_len=valid_len,
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
        raise ValueError("embedding_source=dlm/ode_hidden requires backbone.elf_denoiser.")

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
                raise ValueError(f"Unsupported normalized embedding_source={embedding_source!r}")
    finally:
        model.feature_source = old_feature_source
    return torch.cat(hidden_parts, dim=1)


def build_model(args: argparse.Namespace, device: torch.device, embedding_source: str) -> BasecallModel:
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
    if embedding_source == "ode_hidden" and not hasattr(model.backbone, "elf_denoiser"):
        raise ValueError("Requested DLM/ode_hidden embeddings, but the model has no elf_denoiser.")
    return model


def collect_labeled_c_embeddings(
    record: ReadRecord,
    sequence_hidden: np.ndarray,
    *,
    valid_len: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    token_count = min(valid_len, sequence_hidden.shape[0], len(record.c_modification_label))
    rows = []
    labels = []
    points: list[dict] = []

    for token_index, label in enumerate(record.c_modification_label[:token_count]):
        if label not in (1, 2):
            continue
        rows.append(sequence_hidden[token_index])
        labels.append(1 if label == 2 else 0)
        points.append(
            {
                "read_id": record.record_id,
                "row_index": record.row_index,
                "token_position": token_index,
                "c_modification_label": int(label),
                "modified": int(label == 2),
            }
        )

    if not rows:
        return np.empty((0, sequence_hidden.shape[-1]), dtype=np.float32), np.empty((0,), dtype=np.int64), points
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int64), points


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


def sample_indices(indices: np.ndarray, max_n: int, rng: random.Random) -> np.ndarray:
    if max_n <= 0 or len(indices) <= max_n:
        return indices
    keep = sorted(rng.sample(indices.tolist(), max_n))
    return np.asarray(keep, dtype=np.int64)


def downsample_points(
    embeddings: np.ndarray,
    labels: np.ndarray,
    points: list[dict],
    *,
    max_unmodified_points: int,
    max_modified_points: int,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    unmodified_idx = np.flatnonzero(labels == 0)
    modified_idx = np.flatnonzero(labels == 1)
    keep_idx = np.concatenate(
        [
            sample_indices(unmodified_idx, max_unmodified_points, rng),
            sample_indices(modified_idx, max_modified_points, rng),
        ]
    )
    keep_idx.sort()
    return embeddings[keep_idx], labels[keep_idx], [points[int(idx)] for idx in keep_idx.tolist()]


def plot_distribution(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_png: Path,
    *,
    title: str,
    embedding_source_label: str,
    num_reads: int,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_png.parent.mkdir(parents=True, exist_ok=True)
    coords = pca_2d(embeddings)
    norms = np.linalg.norm(embeddings, axis=1) if embeddings.size else np.empty((0,), dtype=np.float32)
    modified_mask = labels == 1
    unmodified_mask = labels == 0

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    ax = axes[0]
    if unmodified_mask.any():
        ax.scatter(
            coords[unmodified_mask, 0],
            coords[unmodified_mask, 1],
            s=10 if num_reads > 1 else 20,
            alpha=0.32 if num_reads > 1 else 0.68,
            color="#2563eb",
            label=f"unmodified C tokens ({int(unmodified_mask.sum())})",
        )
    if modified_mask.any():
        ax.scatter(
            coords[modified_mask, 0],
            coords[modified_mask, 1],
            s=34 if num_reads > 1 else 52,
            alpha=0.9,
            color="#dc2626",
            edgecolors="black",
            linewidths=0.25,
            marker="*",
            label=f"modified C tokens ({int(modified_mask.sum())})",
        )
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title(f"{embedding_source_label} C-token PCA")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax2 = axes[1]
    if unmodified_mask.any():
        ax2.hist(norms[unmodified_mask], bins=60, alpha=0.62, color="#2563eb", density=num_reads > 1, label="unmodified C")
    if modified_mask.any():
        ax2.hist(norms[modified_mask], bins=30, alpha=0.74, color="#dc2626", density=num_reads > 1, label="modified C")
        ax2.scatter(norms[modified_mask], np.zeros(int(modified_mask.sum())), s=26, color="#dc2626", marker="|")
    ax2.set_xlabel(f"{embedding_source_label} L2 norm")
    ax2.set_ylabel("density" if num_reads > 1 else "count")
    ax2.set_title("Embedding norm distribution")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_png, dpi=220)
    plt.close(fig)

    return {
        "output_png": str(output_png),
        "num_reads": int(num_reads),
        "num_c_tokens": int(len(labels)),
        "num_modified_c_tokens": int(modified_mask.sum()),
        "num_unmodified_c_tokens": int(unmodified_mask.sum()),
        "unmodified_norm_mean": float(norms[unmodified_mask].mean()) if unmodified_mask.any() else None,
        "modified_norm_mean": float(norms[modified_mask].mean()) if modified_mask.any() else None,
        "unmodified_norm_p95": float(np.quantile(norms[unmodified_mask], 0.95)) if unmodified_mask.any() else None,
        "modified_norm_p95": float(np.quantile(norms[modified_mask], 0.95)) if modified_mask.any() else None,
    }


def process_model_batch(
    args: argparse.Namespace,
    model: BasecallModel,
    device: torch.device,
    records: list[ReadRecord],
    *,
    embedding_source: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[dict]], list[ReadRecord]]:
    input_ids, attention_mask, effective_lengths = make_batch(
        records,
        pad_token_id=args.pad_token_id,
        max_length=args.max_length,
        device=device,
    )
    with torch.inference_mode():
        sequence_hidden = forward_sequence_hidden(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            backbone_chunk_size=args.backbone_chunk_size,
            embedding_source=embedding_source,
        ).float().detach().cpu().numpy()

    embeddings_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    points_list: list[list[dict]] = []
    kept_records: list[ReadRecord] = []
    for idx, record in enumerate(records):
        valid_len = min(effective_lengths[idx], sequence_hidden.shape[1])
        embeddings, labels, points = collect_labeled_c_embeddings(
            record,
            sequence_hidden[idx, :valid_len],
            valid_len=valid_len,
        )
        if embeddings.shape[0] == 0:
            print(f"Skip {record.record_id}: no C-token labels with value 1/2.")
            continue
        embeddings_list.append(embeddings)
        labels_list.append(labels)
        points_list.append(points)
        kept_records.append(record)
    return embeddings_list, labels_list, points_list, kept_records


def flush_aggregate_group(
    args: argparse.Namespace,
    group_index: int,
    group_records: list[ReadRecord],
    group_embeddings: list[np.ndarray],
    group_labels: list[np.ndarray],
    group_points: list[list[dict]],
    *,
    output_dir: Path,
    embedding_source_label: str,
    rng: random.Random,
) -> dict | None:
    if not group_embeddings:
        return None

    embeddings = np.concatenate(group_embeddings, axis=0)
    labels = np.concatenate(group_labels, axis=0)
    points = [point for per_read in group_points for point in per_read]
    total_before_sampling = int(len(labels))
    modified_before_sampling = int((labels == 1).sum())
    embeddings, labels, points = downsample_points(
        embeddings,
        labels,
        points,
        max_unmodified_points=args.max_unmodified_points,
        max_modified_points=args.max_modified_points,
        rng=rng,
    )

    output_png = output_dir / "aggregate" / f"group_{group_index:05d}_{embedding_source_label}_reads{len(group_records)}.png"
    read_start = group_records[0].record_id
    read_end = group_records[-1].record_id
    summary = plot_distribution(
        embeddings,
        labels,
        output_png,
        title=(
            f"C-token embedding distribution | {embedding_source_label} | reads={len(group_records)} "
            f"| tokens={len(labels)} | modified={int((labels == 1).sum())}"
        ),
        embedding_source_label=embedding_source_label,
        num_reads=len(group_records),
    )
    summary.update(
        {
            "plot_mode": "aggregate",
            "group_index": int(group_index),
            "read_id_start": read_start,
            "read_id_end": read_end,
            "total_c_tokens_before_sampling": total_before_sampling,
            "total_modified_c_tokens_before_sampling": modified_before_sampling,
            "read_ids": [record.record_id for record in group_records],
        }
    )

    points_path = output_png.with_suffix(".points.jsonl")
    with points_path.open("w", encoding="utf-8") as handle:
        for point in points:
            handle.write(json.dumps(point, ensure_ascii=False) + "\n")
    summary["points_jsonl"] = str(points_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot modified vs unmodified C-token embedding distributions from tokenized "
            "DNA modification jsonl/jsonl.gz files with c_modification_label."
        )
    )
    parser.add_argument("--model-name-or-path", required=True, help="Stage3 HF DLM model directory.")
    parser.add_argument("--jsonl", required=True, help="Input jsonl/jsonl.gz from tokenize_signal_with_c_mod_labels.py.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--plot-mode", choices=("per-read", "aggregate", "both"), default="aggregate")
    parser.add_argument("--reads-per-plot", type=int, default=0, help="Aggregate this many reads per figure; 0 means all selected reads.")
    parser.add_argument("--limit-reads", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=None, help="Optional max token length passed into the model.")
    parser.add_argument("--pad-token-id", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--backbone-chunk-size", type=int, default=2000)
    parser.add_argument("--elf-ode-steps", type=int, default=4)
    parser.add_argument("--elf-ode-start-t", type=float, default=0.85)
    parser.add_argument("--elf-self-cond-cfg-scale", type=float, default=1.0)
    parser.add_argument(
        "--embedding-source",
        choices=("bert", "dlm", "context_hidden", "ode_hidden"),
        default="dlm",
        help="bert/context_hidden uses the context encoder; dlm/ode_hidden uses deterministic DLM ODE refinement.",
    )
    parser.add_argument("--max-unmodified-points", type=int, default=0, help="0 means keep all unmodified C tokens.")
    parser.add_argument("--max-modified-points", type=int, default=0, help="0 means keep all modified C tokens.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_source = normalize_embedding_source(args.embedding_source)
    embedding_source_label = "bert" if embedding_source == "context_hidden" else "dlm"
    device = torch.device(args.device)
    rng = random.Random(args.seed)
    model = build_model(args, device, embedding_source)

    summary_path = output_dir / f"c_modification_{embedding_source_label}_embedding_summary.jsonl"
    total_reads_seen = 0
    reads_with_c_tokens = 0
    per_read_written = 0
    aggregate_group_index = 0
    aggregate_group_records: list[ReadRecord] = []
    aggregate_group_embeddings: list[np.ndarray] = []
    aggregate_group_labels: list[np.ndarray] = []
    aggregate_group_points: list[list[dict]] = []

    def write_aggregate_if_ready(summary_handle, *, force: bool = False) -> None:
        nonlocal aggregate_group_index
        if not aggregate_group_records:
            return
        reads_per_plot = int(args.reads_per_plot)
        should_flush = force or (reads_per_plot > 0 and len(aggregate_group_records) >= reads_per_plot)
        if not should_flush:
            return
        aggregate_group_index += 1
        summary = flush_aggregate_group(
            args,
            aggregate_group_index,
            aggregate_group_records,
            aggregate_group_embeddings,
            aggregate_group_labels,
            aggregate_group_points,
            output_dir=output_dir,
            embedding_source_label=embedding_source_label,
            rng=rng,
        )
        if summary is not None:
            summary_handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
        aggregate_group_records.clear()
        aggregate_group_embeddings.clear()
        aggregate_group_labels.clear()
        aggregate_group_points.clear()

    with summary_path.open("w", encoding="utf-8") as summary_handle:
        batch: list[ReadRecord] = []
        iterator = iter_records(Path(args.jsonl))
        pbar = tqdm(desc=f"collecting C-token {embedding_source_label} embeddings", unit="read")
        for record in iterator:
            if args.limit_reads is not None and total_reads_seen >= args.limit_reads:
                break
            total_reads_seen += 1
            batch.append(record)
            if len(batch) < args.batch_size:
                continue

            embeddings_list, labels_list, points_list, kept_records = process_model_batch(
                args,
                model,
                device,
                batch,
                embedding_source=embedding_source,
            )
            for kept_record, embeddings, labels, points in zip(kept_records, embeddings_list, labels_list, points_list):
                reads_with_c_tokens += 1
                if args.plot_mode in {"per-read", "both"}:
                    output_png = output_dir / "per_read" / f"{safe_name(kept_record.record_id)}_{embedding_source_label}.png"
                    summary = plot_distribution(
                        embeddings,
                        labels,
                        output_png,
                        title=(
                            f"{kept_record.record_id} | {embedding_source_label} | "
                            f"C tokens={len(labels)} | modified={int((labels == 1).sum())}"
                        ),
                        embedding_source_label=embedding_source_label,
                        num_reads=1,
                    )
                    summary.update({"plot_mode": "per-read", "read_id": kept_record.record_id, "row_index": kept_record.row_index})
                    summary_handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
                    per_read_written += 1
                if args.plot_mode in {"aggregate", "both"}:
                    aggregate_group_records.append(kept_record)
                    aggregate_group_embeddings.append(embeddings)
                    aggregate_group_labels.append(labels)
                    aggregate_group_points.append(points)
                    write_aggregate_if_ready(summary_handle)
            pbar.update(len(batch))
            batch = []

        if batch:
            embeddings_list, labels_list, points_list, kept_records = process_model_batch(
                args,
                model,
                device,
                batch,
                embedding_source=embedding_source,
            )
            for kept_record, embeddings, labels, points in zip(kept_records, embeddings_list, labels_list, points_list):
                reads_with_c_tokens += 1
                if args.plot_mode in {"per-read", "both"}:
                    output_png = output_dir / "per_read" / f"{safe_name(kept_record.record_id)}_{embedding_source_label}.png"
                    summary = plot_distribution(
                        embeddings,
                        labels,
                        output_png,
                        title=(
                            f"{kept_record.record_id} | {embedding_source_label} | "
                            f"C tokens={len(labels)} | modified={int((labels == 1).sum())}"
                        ),
                        embedding_source_label=embedding_source_label,
                        num_reads=1,
                    )
                    summary.update({"plot_mode": "per-read", "read_id": kept_record.record_id, "row_index": kept_record.row_index})
                    summary_handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
                    per_read_written += 1
                if args.plot_mode in {"aggregate", "both"}:
                    aggregate_group_records.append(kept_record)
                    aggregate_group_embeddings.append(embeddings)
                    aggregate_group_labels.append(labels)
                    aggregate_group_points.append(points)
                    write_aggregate_if_ready(summary_handle)
            pbar.update(len(batch))
        write_aggregate_if_ready(summary_handle, force=True)
        pbar.close()

    print(f"Input jsonl: {args.jsonl}")
    print(f"Embedding source: {args.embedding_source} -> {embedding_source}")
    print(f"Reads seen: {total_reads_seen}")
    print(f"Reads with C-token labels: {reads_with_c_tokens}")
    print(f"Per-read plots: {per_read_written}")
    print(f"Aggregate plots: {aggregate_group_index}")
    print(f"Output dir: {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
