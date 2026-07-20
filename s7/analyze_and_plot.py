#!/usr/bin/env python3
"""Evaluate MOD/UNMOD separation and create a publication-style diagnostic figure."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize


COLORS = {"MOD": "#D55E00", "UNMOD": "#0072B2"}
SITE_MARKERS = {"S1": "o", "S2": "s", "S3": "^", "S4": "D", "S5": "P"}
METHOD_COLORS = {"Signal stats": "#6B6B6B", "Token frequency": "#009E73", "OLMo embedding": "#CC79A7"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", default="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/s7/signal_windows.10nt.metadata.tsv")
    parser.add_argument("--tokens", default="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/s7/vqe_bert/signal_windows.vqe_tokens.jsonl")
    parser.add_argument("--embeddings", default="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/s7/vqe_dlm_zhou/signal_windows.vqe_dlm_ode_embeddings.npz")
    parser.add_argument("--sites", default="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/s7/selected_5_sites.tsv")
    parser.add_argument("--out-dir", default="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/s7/vqe_dlm_zhou/plot")
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def build_model(n_features: int, n_samples: int):
    n_components = min(20, n_features, n_samples - 2)
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components, random_state=0),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, solver="liblinear"),
    )


def leave_one_site_out(x: np.ndarray, y: np.ndarray, sites: np.ndarray) -> np.ndarray:
    predictions = np.full(y.shape, np.nan, dtype=float)
    for site in sorted(set(sites)):
        test = sites == site
        train = ~test
        model = build_model(x.shape[1], int(train.sum()))
        model.fit(x[train], y[train])
        predictions[test] = model.predict_proba(x[test])[:, 1]
    if np.isnan(predictions).any():
        raise RuntimeError("Cross-validation left missing predictions")
    return predictions


def metric_rows(method: str, y: np.ndarray, pred: np.ndarray, sites: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    groups = [("Pooled", np.ones(y.size, dtype=bool))]
    groups.extend((site, sites == site) for site in sorted(set(sites)))
    for test_site, keep in groups:
        labels = y[keep]
        scores = pred[keep]
        rows.append(
            {
                "method": method,
                "test_site": test_site,
                "n": int(keep.sum()),
                "n_mod": int(labels.sum()),
                "n_unmod": int((1 - labels).sum()),
                "auroc": float(roc_auc_score(labels, scores)),
                "balanced_accuracy": float(balanced_accuracy_score(labels, scores >= 0.5)),
                "accuracy": float(accuracy_score(labels, scores >= 0.5)),
            }
        )
    return rows


def permutation_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    sites: np.ndarray,
    observed: float,
    permutations: int,
    seed: int,
) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=float)
    for index in range(permutations):
        shuffled = y.copy()
        for site in sorted(set(sites)):
            mask = sites == site
            shuffled[mask] = rng.permutation(shuffled[mask])
        null[index] = roc_auc_score(shuffled, leave_one_site_out(x, shuffled, sites))
    pvalue = (1 + np.count_nonzero(null >= observed)) / (permutations + 1)
    return float(pvalue), null


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, width=0.8, color="#555555")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    plot_dir = out_dir / "plot"
    plot_dir.mkdir(parents=True, exist_ok=True)

    windows = read_jsonl(Path(args.windows))
    token_rows = read_jsonl(Path(args.tokens))
    archive = np.load(args.embeddings)
    embeddings = archive["embeddings"].astype(np.float64)
    y = archive["labels"].astype(int)
    sites = archive["site_ids"].astype(str)
    read_ids = archive["read_ids"].astype(str)
    datasets = archive["datasets"].astype(str)
    if len(set(read_ids)) != len(read_ids):
        raise RuntimeError("Read IDs are not unique; this would violate the evaluation unit")
    expected_keys = [(row["site_id"], row["dataset"], row["read_id"]) for row in windows]
    token_keys = [(row["site_id"], row["dataset"], row["read_id"]) for row in token_rows]
    archive_keys = list(zip(sites, datasets, read_ids))
    if expected_keys != token_keys or expected_keys != archive_keys:
        raise RuntimeError("Window, token, and embedding record orders differ")

    signal_stats = np.asarray(
        [
            [
                len(row["signal"]),
                np.mean(row["signal"]),
                np.std(row["signal"]),
                np.median(row["signal"]),
                np.mean(np.abs(np.diff(row["signal"]))) if len(row["signal"]) > 1 else 0.0,
            ]
            for row in windows
        ],
        dtype=float,
    )
    token_hist = np.zeros((len(token_rows), 2401), dtype=float)
    for index, row in enumerate(token_rows):
        values, counts = np.unique(np.asarray(row["raw_signal_tokens"], dtype=int), return_counts=True)
        token_hist[index, values] = counts
    token_hist = normalize(token_hist, norm="l1")

    features = {
        "Signal stats": signal_stats,
        "Token frequency": token_hist,
        "OLMo embedding": embeddings,
    }
    predictions: dict[str, np.ndarray] = {}
    metrics: list[dict] = []
    for method, x in features.items():
        predictions[method] = leave_one_site_out(x, y, sites)
        metrics.extend(metric_rows(method, y, predictions[method], sites))
    metrics_df = pd.DataFrame(metrics)

    observed_auc = float(
        metrics_df.query("method == 'OLMo embedding' and test_site == 'Pooled'")["auroc"].iloc[0]
    )
    permutation_features = PCA(n_components=20, random_state=args.seed).fit_transform(
        StandardScaler().fit_transform(embeddings)
    )
    permutation_observed_auc = roc_auc_score(
        y, leave_one_site_out(permutation_features, y, sites)
    )
    permutation_p, null_auc = permutation_pvalue(
        permutation_features, y, sites, permutation_observed_auc, args.permutations, args.seed
    )
    np.savez_compressed(out_dir / "embedding_permutation_null.npz", auroc=null_auc)

    embedding_scaled = StandardScaler().fit_transform(embeddings)
    pca = PCA(n_components=2, random_state=args.seed)
    pca_xy = pca.fit_transform(embedding_scaled)
    sample_df = pd.DataFrame(
        {
            "site_id": sites,
            "dataset": datasets,
            "label": y,
            "read_id": read_ids,
            "pc1": pca_xy[:, 0],
            "pc2": pca_xy[:, 1],
            "signal_len": signal_stats[:, 0].astype(int),
            "token_count": archive["token_counts"].astype(int),
            "oof_signal_stats_probability": predictions["Signal stats"],
            "oof_token_probability": predictions["Token frequency"],
            "oof_embedding_probability": predictions["OLMo embedding"],
        }
    )
    sample_df.to_csv(out_dir / "embedding_samples_with_oof_predictions.tsv", sep="\t", index=False)
    metrics_df.to_csv(out_dir / "separation_metrics.tsv", sep="\t", index=False, float_format="%.6f")

    with Path(args.sites).open(newline="") as handle:
        selected_sites = list(csv.DictReader(handle, delimiter="\t"))
    site_pos = {row["site_id"]: int(row["start0"]) for row in selected_sites}

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.5), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.flat

    for site in sorted(set(sites)):
        for dataset in ("UNMOD", "MOD"):
            mask = (sites == site) & (datasets == dataset)
            ax_a.scatter(
                pca_xy[mask, 0], pca_xy[mask, 1], s=24, marker=SITE_MARKERS[site],
                color=COLORS[dataset], alpha=0.75, linewidths=0.35, edgecolors="white",
            )
    ax_a.axhline(0, color="#D0D0D0", lw=0.6, zorder=0)
    ax_a.axvline(0, color="#D0D0D0", lw=0.6, zorder=0)
    ax_a.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax_a.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax_a.set_title("Unsupervised OLMo embedding")
    dataset_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=COLORS[name], label=name, markersize=5)
        for name in ("MOD", "UNMOD")
    ]
    site_handles = [
        Line2D([0], [0], marker=SITE_MARKERS[site], linestyle="", color="#555555", label=site, markersize=5)
        for site in sorted(set(sites))
    ]
    first_legend = ax_a.legend(handles=dataset_handles, frameon=False, loc="upper right", title="Dataset")
    ax_a.add_artist(first_legend)
    ax_a.legend(
        handles=site_handles, frameon=True, facecolor="white", edgecolor="none",
        framealpha=0.88, loc="lower right", ncol=1, title="Site",
    )
    style_axes(ax_a)

    rng = np.random.default_rng(args.seed)
    x_positions = {site: index for index, site in enumerate(sorted(set(sites)), start=1)}
    for site in sorted(set(sites)):
        center = x_positions[site]
        for dataset, offset in (("UNMOD", -0.15), ("MOD", 0.15)):
            mask = (sites == site) & (datasets == dataset)
            values = predictions["OLMo embedding"][mask]
            jitter = rng.uniform(-0.055, 0.055, size=values.size)
            ax_b.scatter(
                center + offset + jitter, values, s=14, color=COLORS[dataset], alpha=0.7,
                linewidths=0, zorder=2,
            )
            if values.size:
                q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
                ax_b.vlines(center + offset, q1, q3, color="#222222", lw=2.2, zorder=3)
                ax_b.hlines(median, center + offset - 0.08, center + offset + 0.08, color="#222222", lw=1, zorder=3)
    ax_b.axhline(0.5, color="#777777", lw=0.8, ls="--")
    ax_b.set_xticks(list(x_positions.values()), list(x_positions))
    ax_b.set_ylim(-0.03, 1.03)
    ax_b.set_ylabel("Out-of-fold MOD probability")
    ax_b.set_xlabel("Held-out genomic site")
    ax_b.set_title("Prediction on unseen sites")
    style_axes(ax_b)

    metric_names = [("auroc", "AUROC"), ("balanced_accuracy", "Balanced accuracy")]
    method_order = ["Signal stats", "Token frequency", "OLMo embedding"]
    for metric_index, (column, label) in enumerate(metric_names):
        base_x = np.arange(len(method_order)) + metric_index * 4.2
        for method_index, method in enumerate(method_order):
            site_values = metrics_df.query("method == @method and test_site != 'Pooled'")[column].to_numpy()
            pooled_value = float(
                metrics_df.query("method == @method and test_site == 'Pooled'")[column].iloc[0]
            )
            x = base_x[method_index]
            ax_c.scatter(
                np.full(site_values.size, x), site_values, s=18, facecolors="white",
                edgecolors=METHOD_COLORS[method], linewidths=0.8, zorder=2,
            )
            ax_c.scatter(x, pooled_value, s=48, color=METHOD_COLORS[method], edgecolors="white", linewidths=0.5, zorder=3)
        ax_c.text(base_x.mean(), 0.995, label, ha="center", va="top", fontsize=8, fontweight="bold")
    ax_c.axhline(0.5, color="#777777", lw=0.8, ls="--")
    ax_c.set_xlim(-0.7, 6.9)
    ax_c.set_ylim(0.25, 1.02)
    ax_c.set_xticks(list(np.arange(3)) + list(np.arange(3) + 4.2), ["Stats", "Tokens", "OLMo"] * 2, rotation=25)
    ax_c.set_ylabel("Score")
    ax_c.set_title("Cross-site discrimination")
    ax_c.text(
        0.98, 0.04, f"OLMo permutation P={permutation_p:.3g}",
        transform=ax_c.transAxes, ha="right", va="bottom", fontsize=7, color="#444444",
    )
    style_axes(ax_c)

    site_order = [row["site_id"] for row in selected_sites]
    y_pos = np.arange(len(site_order))
    mod_depth = np.asarray([int(row["mod_complete_window_depth"]) for row in selected_sites])
    unmod_depth = np.asarray([int(row["unmod_complete_window_depth"]) for row in selected_sites])
    height = 0.34
    ax_d.barh(y_pos + height / 2, mod_depth, height=height, color=COLORS["MOD"], label="MOD")
    ax_d.barh(y_pos - height / 2, unmod_depth, height=height, color=COLORS["UNMOD"], label="UNMOD")
    ax_d.axvline(5, color="#444444", lw=0.8, ls="--")
    ax_d.set_yticks(y_pos, [f"{site}  {site_pos[site] / 1e6:.1f} Mb" for site in site_order])
    ax_d.invert_yaxis()
    ax_d.set_xlabel("Reads with complete 10-nt window")
    ax_d.set_title("Selected site depth")
    ax_d.legend(frameon=False, loc="lower right")
    style_axes(ax_d)

    for label, ax in zip("abcd", axes.flat):
        ax.text(-0.13, 1.06, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")

    output_stem = plot_dir / "figure_mod_unmod_embedding_separation"
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(output_stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(output_stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(output_stem.with_suffix(".tiff"), dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    pooled = metrics_df.query("test_site == 'Pooled'").set_index("method")
    summary = {
        "records": len(y),
        "unique_reads": len(set(read_ids)),
        "mod_reads": int(y.sum()),
        "unmod_reads": int((1 - y).sum()),
        "sites": sorted(set(sites)),
        "pca_explained_variance_percent": (pca.explained_variance_ratio_ * 100).tolist(),
        "pooled_leave_one_site_out": {
            method: {
                "auroc": float(pooled.loc[method, "auroc"]),
                "balanced_accuracy": float(pooled.loc[method, "balanced_accuracy"]),
                "accuracy": float(pooled.loc[method, "accuracy"]),
            }
            for method in features
        },
        "olmo_embedding_permutation_test": {
            "permutations": args.permutations,
            "within_site_label_shuffling": True,
            "fixed_unsupervised_pca_components": 20,
            "observed_auroc_for_test": float(permutation_observed_auc),
            "pvalue": permutation_p,
            "null_auroc_mean": float(null_auc.mean()),
            "null_auroc_sd": float(null_auc.std(ddof=1)),
        },
        "interpretation_guardrail": (
            "Dataset labels are confounded with separate sequencing runs; observed separation is not specific proof of 5mC sensing."
        ),
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
