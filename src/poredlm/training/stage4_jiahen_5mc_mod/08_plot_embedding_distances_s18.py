#!/usr/bin/env python3
"""Plot modified/unmodified embedding distances for S18 outputs.

This script reads the split parquet files and pooled embedding npy files
produced by the S18 pipeline. It computes:

1. Per-window distance to the unmodified center of the same 7-mer.
2. Per-7mer class-center distance between modified and unmodified windows.

The first table is useful for classifier-readout diagnostics. The second table
is closer to the geometry summaries used in representation reports.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


MODEL_KEYS = {
    "V600_Apple": "v600",
    "V610_Apple": "v610",
    "V003_Stone": "v003",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--chrom", choices=("chr16", "chr19"), default="chr16")
    parser.add_argument("--compare-role", default="test")
    parser.add_argument("--reference-role", default="baseline")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["V600_Apple", "V610_Apple", "V003_Stone"],
        choices=sorted(MODEL_KEYS),
    )
    parser.add_argument("--l2-normalize-embedding", action="store_true")
    parser.add_argument("--min-reference-per-7mer", type=int, default=3)
    parser.add_argument("--min-class-per-7mer", type=int, default=3)
    parser.add_argument("--output-prefix", default="embedding_distance")
    parser.add_argument("--max-strip-points", type=int, default=4000)
    return parser.parse_args()


def load_metadata(path: Path) -> dict[str, np.ndarray]:
    table = pq.read_table(path, columns=[
        "row_index",
        "window_id",
        "site_pos0",
        "label",
        "ref_7mer",
        "fast5_raw_read_id",
        "role",
    ])
    data = table.to_pydict()
    return {
        "row_index": np.asarray(data["row_index"], dtype=np.int64),
        "window_id": np.asarray(data["window_id"], dtype=object),
        "site_pos0": np.asarray(data["site_pos0"], dtype=np.int64),
        "label": np.asarray(data["label"], dtype=np.int8),
        "ref_7mer": np.asarray(data["ref_7mer"], dtype=object),
        "fast5_raw_read_id": np.asarray(data["fast5_raw_read_id"], dtype=object),
        "role": np.asarray(data["role"], dtype=object),
    }


def load_embedding(path: Path, rows: int, normalize: bool) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
    if values.shape != (rows, 768):
        raise RuntimeError(f"Unexpected embedding shape {values.shape} for {path}")
    if normalize:
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        values = values / np.maximum(norms, 1e-6)
    return values


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = np.sum(a * b, axis=1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cosine = numerator / np.maximum(denominator, 1e-6)
    return 1.0 - np.clip(cosine, -1.0, 1.0)


def per_7mer_centers(
    embedding: np.ndarray,
    indices: np.ndarray,
    kmers: np.ndarray,
    min_count: int,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    centers: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for kmer in sorted(set(kmers[indices].tolist())):
        selected = indices[kmers[indices] == kmer]
        if selected.size < min_count:
            continue
        centers[str(kmer)] = embedding[selected].mean(axis=0).astype(np.float32)
        counts[str(kmer)] = int(selected.size)
    return centers, counts


def calculate_distances_for_model(
    model_name: str,
    embedding: np.ndarray,
    meta: dict[str, np.ndarray],
    compare_indices: np.ndarray,
    reference_indices: np.ndarray,
    min_reference_per_7mer: int,
    min_class_per_7mer: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    labels = meta["label"]
    kmers = meta["ref_7mer"]
    reference_centers, reference_counts = per_7mer_centers(
        embedding,
        reference_indices,
        kmers,
        min_reference_per_7mer,
    )

    sample_rows: list[dict[str, object]] = []
    for index in compare_indices:
        kmer = str(kmers[index])
        center = reference_centers.get(kmer)
        if center is None:
            continue
        value = embedding[index:index + 1]
        center_2d = center[None, :]
        sample_rows.append({
            "model": model_name,
            "row_index": int(index),
            "window_id": str(meta["window_id"][index]),
            "site_pos0": int(meta["site_pos0"][index]),
            "fast5_raw_read_id": str(meta["fast5_raw_read_id"][index]),
            "ref_7mer": kmer,
            "label": int(labels[index]),
            "reference_n": int(reference_counts[kmer]),
            "l2_distance": float(np.linalg.norm(value - center_2d, axis=1)[0]),
            "cosine_distance": float(cosine_distance(value, center_2d)[0]),
        })

    center_rows: list[dict[str, object]] = []
    for kmer in sorted(reference_centers):
        pos = compare_indices[(kmers[compare_indices] == kmer) & (labels[compare_indices] == 1)]
        neg = compare_indices[(kmers[compare_indices] == kmer) & (labels[compare_indices] == 0)]
        if pos.size < min_class_per_7mer or neg.size < min_class_per_7mer:
            continue
        pos_center = embedding[pos].mean(axis=0, keepdims=True)
        neg_center = embedding[neg].mean(axis=0, keepdims=True)
        reference_center = reference_centers[kmer][None, :]
        center_rows.append({
            "model": model_name,
            "ref_7mer": kmer,
            "n_modified": int(pos.size),
            "n_unmodified_test": int(neg.size),
            "n_unmodified_reference": int(reference_counts[kmer]),
            "modified_vs_unmodified_test_l2": float(np.linalg.norm(pos_center - neg_center, axis=1)[0]),
            "modified_vs_unmodified_test_cosine": float(cosine_distance(pos_center, neg_center)[0]),
            "modified_vs_reference_l2": float(np.linalg.norm(pos_center - reference_center, axis=1)[0]),
            "modified_vs_reference_cosine": float(cosine_distance(pos_center, reference_center)[0]),
        })

    return sample_rows, center_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_site_distances(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["model"]), int(row["site_pos0"]))
        groups.setdefault(key, []).append(row)

    output: list[dict[str, object]] = []
    for (model, site_pos0), values in sorted(groups.items()):
        labels = {int(row["label"]) for row in values}
        kmers = {str(row["ref_7mer"]) for row in values}
        if len(labels) != 1 or len(kmers) != 1:
            raise RuntimeError(f"Inconsistent site group for {model} site {site_pos0}")
        read_ids = {str(row["fast5_raw_read_id"]) for row in values}
        output.append({
            "model": model,
            "site_pos0": int(site_pos0),
            "ref_7mer": next(iter(kmers)),
            "label": next(iter(labels)),
            "n_windows": len(values),
            "n_reads": len(read_ids),
            "mean_l2_distance": float(np.mean([float(row["l2_distance"]) for row in values])),
            "mean_cosine_distance": float(np.mean([float(row["cosine_distance"]) for row in values])),
            "median_l2_distance": float(np.median([float(row["l2_distance"]) for row in values])),
            "median_cosine_distance": float(np.median([float(row["cosine_distance"]) for row in values])),
        })
    return output


def plot_sample_distances(
    rows: list[dict[str, object]],
    output_base: Path,
    max_strip_points: int,
    l2_field: str = "l2_distance",
    cosine_field: str = "cosine_distance",
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
    })

    models = [model for model in MODEL_KEYS if any(row["model"] == model for row in rows)]
    metrics = [
        (l2_field, "L2 distance to unmodified center"),
        (cosine_field, "Cosine distance to unmodified center"),
    ]
    colors = {0: "#6c757d", 1: "#c43c39"}
    labels = {0: "Unmodified", 1: "Modified"}
    fig, axes = plt.subplots(len(models), len(metrics), figsize=(7.1, 2.0 * len(models)), squeeze=False)
    rng = np.random.default_rng(1729)

    for row_index, model in enumerate(models):
        model_rows = [row for row in rows if row["model"] == model]
        for col_index, (metric, title) in enumerate(metrics):
            ax = axes[row_index][col_index]
            data = []
            for label in (0, 1):
                values = np.asarray([
                    float(row[metric])
                    for row in model_rows
                    if int(row["label"]) == label
                ], dtype=np.float64)
                data.append(values)
            box = ax.boxplot(
                data,
                positions=[0, 1],
                widths=0.55,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.0},
                boxprops={"linewidth": 0.8},
                whiskerprops={"linewidth": 0.8},
                capprops={"linewidth": 0.8},
            )
            for patch, label in zip(box["boxes"], (0, 1)):
                patch.set_facecolor(colors[label])
                patch.set_alpha(0.28)
                patch.set_edgecolor(colors[label])
            for x_pos, label in enumerate((0, 1)):
                values = data[x_pos]
                if values.size == 0:
                    continue
                if values.size > max_strip_points:
                    values = values[rng.choice(values.size, size=max_strip_points, replace=False)]
                jitter = rng.normal(0.0, 0.055, size=values.size)
                ax.scatter(
                    np.full(values.size, x_pos) + jitter,
                    values,
                    s=3,
                    color=colors[label],
                    alpha=0.18,
                    linewidths=0,
                    rasterized=True,
                )
            ax.set_xticks([0, 1], [labels[0], labels[1]])
            ax.set_title(title if row_index == 0 else "")
            ax.set_ylabel(model if col_index == 0 else "")
            ax.grid(axis="y", color="#e5e5e5", linewidth=0.5)
            n0, n1 = len(data[0]), len(data[1])
            med0 = np.nanmedian(data[0]) if n0 else np.nan
            med1 = np.nanmedian(data[1]) if n1 else np.nan
            ax.text(
                0.02,
                0.96,
                f"n={n0}/{n1}\nmedian delta={med1 - med0:.3g}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=6.5,
            )

    fig.tight_layout()
    fig.savefig(f"{output_base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{output_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_base}.svg", bbox_inches="tight")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    meta_path = out_dir / f"{args.chrom}_split.parquet"
    embedding_dir = out_dir / "embeddings"
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    meta = load_metadata(meta_path)
    rows = int(meta["label"].size)
    compare_indices = np.flatnonzero(meta["role"] == args.compare_role)
    reference_indices = np.flatnonzero((meta["role"] == args.reference_role) & (meta["label"] == 0))
    if reference_indices.size == 0:
        reference_indices = np.flatnonzero((meta["role"] == args.compare_role) & (meta["label"] == 0))
    if compare_indices.size == 0 or reference_indices.size == 0:
        raise RuntimeError(
            f"Empty compare/reference split: compare={compare_indices.size}, reference={reference_indices.size}"
        )

    all_sample_rows: list[dict[str, object]] = []
    all_center_rows: list[dict[str, object]] = []
    for model_name in args.models:
        model_key = MODEL_KEYS[model_name]
        embedding = load_embedding(
            embedding_dir / f"{args.chrom}_{model_key}_l2_full.npy",
            rows,
            args.l2_normalize_embedding,
        )
        sample_rows, center_rows = calculate_distances_for_model(
            model_name,
            embedding,
            meta,
            compare_indices,
            reference_indices,
            args.min_reference_per_7mer,
            args.min_class_per_7mer,
        )
        all_sample_rows.extend(sample_rows)
        all_center_rows.extend(center_rows)

    suffix = f"{args.output_prefix}_{args.chrom}_{args.compare_role}"
    if args.l2_normalize_embedding:
        suffix += "_l2norm"
    sample_csv = figure_dir / f"{suffix}_sample_to_unmodified_center.csv"
    site_csv = figure_dir / f"{suffix}_site_to_unmodified_center.csv"
    center_csv = figure_dir / f"{suffix}_per_7mer_center_distance.csv"
    site_rows = aggregate_site_distances(all_sample_rows)
    write_csv(sample_csv, all_sample_rows)
    write_csv(site_csv, site_rows)
    write_csv(center_csv, all_center_rows)
    plot_sample_distances(
        all_sample_rows,
        figure_dir / f"{suffix}_sample_to_unmodified_center",
        args.max_strip_points,
    )
    plot_sample_distances(
        site_rows,
        figure_dir / f"{suffix}_site_to_unmodified_center",
        args.max_strip_points,
        l2_field="mean_l2_distance",
        cosine_field="mean_cosine_distance",
    )
    print({
        "sample_csv": str(sample_csv),
        "site_csv": str(site_csv),
        "center_csv": str(center_csv),
        "sample_rows": len(all_sample_rows),
        "site_rows": len(site_rows),
        "center_rows": len(all_center_rows),
        "sample_figure_prefix": str(figure_dir / f"{suffix}_sample_to_unmodified_center"),
        "site_figure_prefix": str(figure_dir / f"{suffix}_site_to_unmodified_center"),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
