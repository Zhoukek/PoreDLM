#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm


THIS_FILE = Path(__file__).resolve()
SCRIPT_DIR = THIS_FILE.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_c_base_ode_embedding_distribution import (  # noqa: E402
    BasecallModel,
    ReadRecord,
    build_model,
    collect_c_token_embeddings,
    forward_sequence_hidden,
    iter_records,
    make_batch,
    parse_positions,
    pca_2d,
)


def safe_sample_indices(n: int, max_n: int, rng: random.Random) -> np.ndarray:
    if max_n <= 0 or n <= max_n:
        return np.arange(n, dtype=np.int64)
    return np.asarray(sorted(rng.sample(range(n), max_n)), dtype=np.int64)


def process_batch(
    args: argparse.Namespace,
    model: BasecallModel,
    device: torch.device,
    records: list[ReadRecord],
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict]]:
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
            embedding_source=args.embedding_source,
        ).float().detach().cpu().numpy()

    embeddings_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    point_rows: list[dict] = []
    for idx, record in enumerate(records):
        valid_len = min(effective_lengths[idx], sequence_hidden.shape[1])
        embeddings, labels, points = collect_c_token_embeddings(
            record,
            sequence_hidden[idx, :valid_len],
            samples_per_token=args.samples_per_token,
            base_span_offset=args.base_span_offset,
            modified_base_positions=args.modified_base_positions,
        )
        if embeddings.shape[0] == 0:
            continue
        embeddings_list.append(embeddings)
        labels_list.append(labels)
        for point in points:
            point_rows.append(
                {
                    "read_id": record.record_id,
                    "row_index": record.row_index,
                    **point,
                }
            )
    return embeddings_list, labels_list, point_rows


def plot_aggregate(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_png: Path,
    *,
    embedding_source: str,
    num_reads: int,
    base_span_offset: int,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_png.parent.mkdir(parents=True, exist_ok=True)
    coords = pca_2d(embeddings)
    norms = np.linalg.norm(embeddings, axis=1)
    modified_mask = labels == 1
    unmodified_mask = labels == 0

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    ax = axes[0]
    if unmodified_mask.any():
        ax.scatter(
            coords[unmodified_mask, 0],
            coords[unmodified_mask, 1],
            s=8,
            alpha=0.28,
            color="#1f77b4",
            label=f"C unmodified tokens ({int(unmodified_mask.sum())})",
        )
    if modified_mask.any():
        ax.scatter(
            coords[modified_mask, 0],
            coords[modified_mask, 1],
            s=30,
            alpha=0.9,
            color="#d62728",
            edgecolors="black",
            linewidths=0.25,
            marker="*",
            label=f"C modified tokens ({int(modified_mask.sum())})",
        )
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title(f"All reads C-token {embedding_source} PCA")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax2 = axes[1]
    if unmodified_mask.any():
        ax2.hist(norms[unmodified_mask], bins=80, alpha=0.62, color="#1f77b4", density=True, label="C unmodified")
    if modified_mask.any():
        ax2.hist(norms[modified_mask], bins=40, alpha=0.74, color="#d62728", density=True, label="C modified")
        ax2.scatter(norms[modified_mask], np.zeros(int(modified_mask.sum())), s=24, color="#d62728", marker="|")
    ax2.set_xlabel(f"{embedding_source} L2 norm")
    ax2.set_ylabel("density")
    ax2.set_title("Embedding norm distribution")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best")

    fig.suptitle(
        f"LB06 C-token embedding distribution | reads={num_reads} | "
        f"tokens={len(labels)} | modified={int(modified_mask.sum())} | base_span_offset={base_span_offset}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_png, dpi=220)
    plt.close(fig)

    return {
        "output_png": str(output_png),
        "embedding_source": embedding_source,
        "num_reads": int(num_reads),
        "num_c_tokens": int(len(labels)),
        "num_modified_c_tokens": int(modified_mask.sum()),
        "num_unmodified_c_tokens": int(unmodified_mask.sum()),
        "base_span_offset": int(base_span_offset),
        "unmodified_norm_mean": float(norms[unmodified_mask].mean()) if unmodified_mask.any() else None,
        "modified_norm_mean": float(norms[modified_mask].mean()) if modified_mask.any() else None,
        "unmodified_norm_p95": float(np.quantile(norms[unmodified_mask], 0.95)) if unmodified_mask.any() else None,
        "modified_norm_p95": float(np.quantile(norms[modified_mask], 0.95)) if modified_mask.any() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot one aggregate PCA distribution for C-base embeddings from many reads."
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--jsonl", required=True)
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
    parser.add_argument("--embedding-source", choices=("context_hidden", "ode_hidden"), default="ode_hidden")
    parser.add_argument("--samples-per-token", type=int, default=5)
    parser.add_argument("--base-span-offset", type=int, default=3)
    parser.add_argument("--modified-base-positions", type=parse_positions, default=parse_positions("14,33,52,71,90,109,128"))
    parser.add_argument("--limit-reads", type=int, default=None)
    parser.add_argument("--max-unmodified-points", type=int, default=0, help="0 means keep all unmodified C tokens.")
    parser.add_argument("--max-modified-points", type=int, default=0, help="0 means keep all modified C tokens.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = build_model(args, device)
    rng = random.Random(args.seed)

    all_embeddings: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_points: list[dict] = []
    total_reads = 0
    plotted_reads = 0

    batch: list[ReadRecord] = []
    pbar = tqdm(desc=f"collecting all-read C-token {args.embedding_source}", unit="read")
    for record in iter_records(Path(args.jsonl)):
        if args.limit_reads is not None and total_reads >= args.limit_reads:
            break
        total_reads += 1
        batch.append(record)
        if len(batch) < args.batch_size:
            continue
        embeddings_list, labels_list, point_rows = process_batch(args, model, device, batch)
        all_embeddings.extend(embeddings_list)
        all_labels.extend(labels_list)
        all_points.extend(point_rows)
        plotted_reads += len(embeddings_list)
        pbar.update(len(batch))
        batch = []
    if batch:
        embeddings_list, labels_list, point_rows = process_batch(args, model, device, batch)
        all_embeddings.extend(embeddings_list)
        all_labels.extend(labels_list)
        all_points.extend(point_rows)
        plotted_reads += len(embeddings_list)
        pbar.update(len(batch))
    pbar.close()

    if not all_embeddings:
        raise RuntimeError("No C-token embeddings were collected.")

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    points = np.asarray(all_points, dtype=object)

    unmodified_idx = np.flatnonzero(labels == 0)
    modified_idx = np.flatnonzero(labels == 1)
    keep_unmodified = unmodified_idx[safe_sample_indices(len(unmodified_idx), args.max_unmodified_points, rng)]
    keep_modified = modified_idx[safe_sample_indices(len(modified_idx), args.max_modified_points, rng)]
    keep_idx = np.concatenate([keep_unmodified, keep_modified])
    keep_idx.sort()

    sampled_embeddings = embeddings[keep_idx]
    sampled_labels = labels[keep_idx]
    sampled_points = [all_points[int(idx)] for idx in keep_idx.tolist()]

    output_png = output_dir / f"all_reads_C_{args.embedding_source}_pca.png"
    summary = plot_aggregate(
        sampled_embeddings,
        sampled_labels,
        output_png,
        embedding_source=args.embedding_source,
        num_reads=plotted_reads,
        base_span_offset=args.base_span_offset,
    )
    summary.update(
        {
            "input_jsonl": args.jsonl,
            "total_reads_seen": int(total_reads),
            "reads_with_c_tokens": int(plotted_reads),
            "total_c_tokens_before_sampling": int(len(labels)),
            "total_modified_c_tokens_before_sampling": int((labels == 1).sum()),
            "total_unmodified_c_tokens_before_sampling": int((labels == 0).sum()),
            "max_unmodified_points": int(args.max_unmodified_points),
            "max_modified_points": int(args.max_modified_points),
            "seed": int(args.seed),
        }
    )

    summary_path = output_dir / f"all_reads_C_{args.embedding_source}_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    points_path = output_dir / f"all_reads_C_{args.embedding_source}_points.jsonl"
    with points_path.open("w", encoding="utf-8") as handle:
        for point in sampled_points:
            handle.write(json.dumps(point, ensure_ascii=False) + "\n")

    print(f"Output plot: {output_png}")
    print(f"Summary: {summary_path}")
    print(f"Sampled points: {points_path}")


if __name__ == "__main__":
    main()
