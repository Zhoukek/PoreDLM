#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
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
    pca_2d,
)


DEFAULT_LB07_JSONL = (
    "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/LB07_AND_LB06/"
    "all_data/split_LB07_only/validation_seq1_to_seq17_ref_target_cropped_token_c_modlabel.jsonl.gz"
)
DEFAULT_LB06_JSONL = (
    "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/LB07_AND_LB06/"
    "stage4_modification/validation_seq1_to_seq17_ref_target_cropped_token_c_modlabel.jsonl.gz"
)


@dataclass
class EmbeddingRow:
    dataset: str
    sequence_key: str
    read_id: str
    row_index: int
    token_position: int
    label: int
    embedding: np.ndarray


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))[:180]


def sequence_key_for_record(record: ReadRecord, mode: str) -> str:
    meta = record.meta or {}
    if mode == "label":
        value = meta.get("label")
    elif mode == "ref":
        value = meta.get("ref")
    elif mode == "seq":
        value = meta.get("seq")
    elif mode == "auto":
        value = meta.get("label") or meta.get("ref") or meta.get("seq")
    else:
        raise ValueError(f"Unsupported sequence key mode: {mode}")
    if value is None or str(value) == "":
        return "unknown_sequence"
    return str(value)


def collect_dataset_embeddings(
    args: argparse.Namespace,
    *,
    model: BasecallModel,
    device: torch.device,
    records: list[ReadRecord],
    dataset_name: str,
    embedding_source: str,
    wanted_positions_by_sequence: dict[str, set[int]] | None,
    wanted_label_values: set[int],
) -> list[EmbeddingRow]:
    rows: list[EmbeddingRow] = []
    batch: list[ReadRecord] = []
    pbar = tqdm(total=len(records), desc=f"extracting {dataset_name} {args.embedding_source}", unit="read")

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
            wanted_positions = None
            if wanted_positions_by_sequence is not None:
                wanted_positions = wanted_positions_by_sequence.get(sequence_key, set())
                if not wanted_positions:
                    continue
            token_count = min(effective_lengths[idx], hidden.shape[1], len(record.c_modification_label))
            for token_position, label in enumerate(record.c_modification_label[:token_count]):
                label = int(label)
                if label not in wanted_label_values:
                    continue
                if wanted_positions is not None and token_position not in wanted_positions:
                    continue
                rows.append(
                    EmbeddingRow(
                        dataset=dataset_name,
                        sequence_key=sequence_key,
                        read_id=record.record_id,
                        row_index=record.row_index,
                        token_position=int(token_position),
                        label=label,
                        embedding=hidden[idx, token_position].astype(np.float32),
                    )
                )
        pbar.update(len(batch))
        batch = []

    for record in records:
        batch.append(record)
        if len(batch) >= args.batch_size:
            flush_batch()
    flush_batch()
    pbar.close()
    return rows


def load_limited_records(path: str, *, limit_reads: int | None) -> list[ReadRecord]:
    records = []
    for record in iter_records(Path(path)):
        if limit_reads is not None and len(records) >= limit_reads:
            break
        records.append(record)
    return records


def modified_positions_from_rows(lb06_rows: list[EmbeddingRow]) -> dict[str, set[int]]:
    positions: dict[str, set[int]] = defaultdict(set)
    for row in lb06_rows:
        if row.label == 2:
            positions[row.sequence_key].add(row.token_position)
    return positions


def sample_rows(rows: list[EmbeddingRow], max_rows: int, rng: random.Random) -> list[EmbeddingRow]:
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows
    return [rows[index] for index in sorted(rng.sample(range(len(rows)), max_rows))]


def stack_embeddings(rows: list[EmbeddingRow]) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
    x = np.stack([row.embedding for row in rows], axis=0).astype(np.float32)
    y = np.asarray([1 if row.dataset == "LB06_modified" else 0 for row in rows], dtype=np.int64)
    return x, y


def plot_rows(
    rows: list[EmbeddingRow],
    output_png: Path,
    *,
    title: str,
    embedding_label: str,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_png.parent.mkdir(parents=True, exist_ok=True)
    x, y = stack_embeddings(rows)
    coords = pca_2d(x)
    norms = np.linalg.norm(x, axis=1) if x.size else np.empty((0,), dtype=np.float32)
    lb07_mask = y == 0
    lb06_mask = y == 1

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    ax = axes[0]
    if lb07_mask.any():
        ax.scatter(
            coords[lb07_mask, 0],
            coords[lb07_mask, 1],
            s=10,
            alpha=0.32,
            color="#2563eb",
            label=f"LB07 unmodified same-site C ({int(lb07_mask.sum())})",
        )
    if lb06_mask.any():
        ax.scatter(
            coords[lb06_mask, 0],
            coords[lb06_mask, 1],
            s=28,
            alpha=0.85,
            color="#dc2626",
            edgecolors="black",
            linewidths=0.2,
            marker="*",
            label=f"LB06 modified C ({int(lb06_mask.sum())})",
        )
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title(f"{embedding_label} same-site C PCA")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax2 = axes[1]
    if lb07_mask.any():
        ax2.hist(norms[lb07_mask], bins=70, alpha=0.62, color="#2563eb", density=True, label="LB07 same-site unmodified")
    if lb06_mask.any():
        ax2.hist(norms[lb06_mask], bins=45, alpha=0.72, color="#dc2626", density=True, label="LB06 modified")
        ax2.scatter(norms[lb06_mask], np.zeros(int(lb06_mask.sum())), s=22, color="#dc2626", marker="|")
    ax2.set_xlabel(f"{embedding_label} L2 norm")
    ax2.set_ylabel("density")
    ax2.set_title("Embedding norm distribution")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_png, dpi=220)
    plt.close(fig)

    return {
        "output_png": str(output_png),
        "num_points": int(len(rows)),
        "lb07_unmodified_same_site_points": int(lb07_mask.sum()),
        "lb06_modified_points": int(lb06_mask.sum()),
        "lb07_norm_mean": float(norms[lb07_mask].mean()) if lb07_mask.any() else None,
        "lb06_norm_mean": float(norms[lb06_mask].mean()) if lb06_mask.any() else None,
        "lb07_norm_p95": float(np.quantile(norms[lb07_mask], 0.95)) if lb07_mask.any() else None,
        "lb06_norm_p95": float(np.quantile(norms[lb06_mask], 0.95)) if lb06_mask.any() else None,
    }


def save_points(rows: list[EmbeddingRow], points_path: Path) -> None:
    points_path.parent.mkdir(parents=True, exist_ok=True)
    with points_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "dataset": row.dataset,
                        "sequence_key": row.sequence_key,
                        "read_id": row.read_id,
                        "row_index": row.row_index,
                        "token_position": row.token_position,
                        "c_modification_label": row.label,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def summarize_positions(positions_by_sequence: dict[str, set[int]]) -> dict[str, Any]:
    return {
        sequence_key: {
            "num_modified_token_positions": len(positions),
            "modified_token_positions": sorted(positions),
        }
        for sequence_key, positions in sorted(positions_by_sequence.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare LB06 modified C embeddings against LB07 unmodified C embeddings at the same "
            "sequence key and token positions. Uses c_modification_label: 1=unmodified C, 2=modified C."
        )
    )
    parser.add_argument("--model-name-or-path", required=True, help="Stage3 HF DLM model directory.")
    parser.add_argument("--lb07-jsonl", default=DEFAULT_LB07_JSONL)
    parser.add_argument("--lb06-jsonl", default=DEFAULT_LB06_JSONL)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embedding-source", choices=("bert", "dlm", "context_hidden", "ode_hidden"), default="bert")
    parser.add_argument("--sequence-key", choices=("auto", "label", "ref", "seq"), default="label")
    parser.add_argument("--plot-mode", choices=("all", "per-sequence", "both"), default="both")
    parser.add_argument("--limit-lb07-reads", type=int, default=None)
    parser.add_argument("--limit-lb06-reads", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2000)
    parser.add_argument("--pad-token-id", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--backbone-chunk-size", type=int, default=2000)
    parser.add_argument("--elf-ode-steps", type=int, default=4)
    parser.add_argument("--elf-ode-start-t", type=float, default=0.95)
    parser.add_argument("--elf-self-cond-cfg-scale", type=float, default=1.0)
    parser.add_argument("--max-lb07-points", type=int, default=100000, help="0 means keep all LB07 same-site points.")
    parser.add_argument("--max-lb06-points", type=int, default=0, help="0 means keep all LB06 modified points.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_source = normalize_embedding_source(args.embedding_source)
    embedding_label = "bert" if embedding_source == "context_hidden" else "dlm"
    device = torch.device(args.device)
    rng = random.Random(args.seed)

    lb06_records = load_limited_records(args.lb06_jsonl, limit_reads=args.limit_lb06_reads)
    lb07_records = load_limited_records(args.lb07_jsonl, limit_reads=args.limit_lb07_reads)
    if not lb06_records:
        raise ValueError(f"No LB06 records loaded from {args.lb06_jsonl}")
    if not lb07_records:
        raise ValueError(f"No LB07 records loaded from {args.lb07_jsonl}")

    model = build_model(args, device, embedding_source)

    lb06_rows = collect_dataset_embeddings(
        args,
        model=model,
        device=device,
        records=lb06_records,
        dataset_name="LB06_modified",
        embedding_source=embedding_source,
        wanted_positions_by_sequence=None,
        wanted_label_values={2},
    )
    positions_by_sequence = modified_positions_from_rows(lb06_rows)
    if not positions_by_sequence:
        raise ValueError("No LB06 label=2 modified C token positions were found.")

    lb07_rows = collect_dataset_embeddings(
        args,
        model=model,
        device=device,
        records=lb07_records,
        dataset_name="LB07_unmodified_same_site",
        embedding_source=embedding_source,
        wanted_positions_by_sequence=positions_by_sequence,
        wanted_label_values={1},
    )
    if not lb07_rows:
        raise ValueError("No LB07 label=1 C token embeddings were found at LB06 modified token positions.")

    summaries: list[dict[str, Any]] = []
    all_rows = sample_rows(lb07_rows, args.max_lb07_points, rng) + sample_rows(lb06_rows, args.max_lb06_points, rng)
    if args.plot_mode in {"all", "both"}:
        output_png = output_dir / f"LB06_vs_LB07_same_site_C_{embedding_label}_all.png"
        summary = plot_rows(
            all_rows,
            output_png,
            title=(
                f"LB06 modified C vs LB07 same-site unmodified C | {embedding_label} | "
                f"sequences={len(positions_by_sequence)}"
            ),
            embedding_label=embedding_label,
        )
        summary.update({"plot_mode": "all", "embedding_source": embedding_source})
        points_path = output_png.with_suffix(".points.jsonl")
        save_points(all_rows, points_path)
        summary["points_jsonl"] = str(points_path)
        summaries.append(summary)

    if args.plot_mode in {"per-sequence", "both"}:
        lb06_by_seq: dict[str, list[EmbeddingRow]] = defaultdict(list)
        lb07_by_seq: dict[str, list[EmbeddingRow]] = defaultdict(list)
        for row in lb06_rows:
            lb06_by_seq[row.sequence_key].append(row)
        for row in lb07_rows:
            lb07_by_seq[row.sequence_key].append(row)

        for sequence_key in sorted(positions_by_sequence):
            seq_rows = sample_rows(lb07_by_seq.get(sequence_key, []), args.max_lb07_points, rng)
            seq_rows += sample_rows(lb06_by_seq.get(sequence_key, []), args.max_lb06_points, rng)
            if not seq_rows:
                continue
            output_png = output_dir / "per_sequence" / f"{safe_name(sequence_key)}_{embedding_label}_same_site_C.png"
            summary = plot_rows(
                seq_rows,
                output_png,
                title=f"{sequence_key} | LB06 modified C vs LB07 same-site unmodified C | {embedding_label}",
                embedding_label=embedding_label,
            )
            summary.update(
                {
                    "plot_mode": "per-sequence",
                    "sequence_key": sequence_key,
                    "embedding_source": embedding_source,
                    "modified_token_positions": sorted(positions_by_sequence[sequence_key]),
                }
            )
            points_path = output_png.with_suffix(".points.jsonl")
            save_points(seq_rows, points_path)
            summary["points_jsonl"] = str(points_path)
            summaries.append(summary)

    summary = {
        "lb07_jsonl": args.lb07_jsonl,
        "lb06_jsonl": args.lb06_jsonl,
        "embedding_source_requested": args.embedding_source,
        "embedding_source_normalized": embedding_source,
        "embedding_label": embedding_label,
        "sequence_key": args.sequence_key,
        "lb06_records": len(lb06_records),
        "lb07_records": len(lb07_records),
        "lb06_modified_points_total": len(lb06_rows),
        "lb07_unmodified_same_site_points_total": len(lb07_rows),
        "lb06_modified_points_by_sequence": dict(Counter(row.sequence_key for row in lb06_rows)),
        "lb07_same_site_points_by_sequence": dict(Counter(row.sequence_key for row in lb07_rows)),
        "modified_positions_by_sequence": summarize_positions(positions_by_sequence),
        "plots": summaries,
    }
    summary_path = output_dir / f"LB06_vs_LB07_same_site_C_{embedding_label}_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"LB06 records: {len(lb06_records)}")
    print(f"LB07 records: {len(lb07_records)}")
    print(f"LB06 modified C points: {len(lb06_rows)}")
    print(f"LB07 same-site unmodified C points: {len(lb07_rows)}")
    print(f"Output dir: {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
