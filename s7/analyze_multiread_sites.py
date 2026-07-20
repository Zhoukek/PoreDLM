#!/usr/bin/env python3
"""Classify modification at the site level after aggregating multiple read embeddings."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, normalize


COLORS = {"MOD": "#D55E00", "UNMOD": "#0072B2"}
SITE_COLORS = {"S1": "#4E79A7", "S2": "#59A14F", "S3": "#F28E2B", "S4": "#B07AA1", "S5": "#9C755F"}
METHOD_COLORS = {"Signal stats": "#6B6B6B", "Token frequency": "#009E73", "OLMo embedding": "#CC79A7"}
GROUPS = [(site, dataset) for site in ("S1", "S2", "S3", "S4", "S5") for dataset in ("UNMOD", "MOD")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--sites", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--plot-dir", default=None)
    parser.add_argument("--figure-stem", default="figure_multiread_site_level_separation")
    parser.add_argument("--model-label", default="OLMo")
    parser.add_argument("--bootstraps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def aggregate_group(
    x: np.ndarray,
    sites: np.ndarray,
    datasets: np.ndarray,
    strands: np.ndarray,
    site: str,
    dataset: str,
) -> np.ndarray:
    strand_centroids = []
    for strand in ("+", "-"):
        keep = (sites == site) & (datasets == dataset) & (strands == strand)
        if not keep.any():
            raise RuntimeError(f"{site}/{dataset} has no {strand}-strand reads")
        strand_centroids.append(x[keep].mean(axis=0))
    return np.mean(strand_centroids, axis=0)


def bootstrap_group(
    x: np.ndarray,
    sites: np.ndarray,
    datasets: np.ndarray,
    strands: np.ndarray,
    site: str,
    dataset: str,
    rng: np.random.Generator,
) -> np.ndarray:
    strand_centroids = []
    for strand in ("+", "-"):
        values = x[(sites == site) & (datasets == dataset) & (strands == strand)]
        indices = rng.integers(0, len(values), size=len(values))
        strand_centroids.append(values[indices].mean(axis=0))
    return np.mean(strand_centroids, axis=0)


def reduce_fold(x: np.ndarray, train_reads: np.ndarray, max_components: int = 20) -> np.ndarray:
    scaler = StandardScaler().fit(x[train_reads])
    scaled_train = scaler.transform(x[train_reads])
    n_components = min(max_components, scaled_train.shape[0] - 1, scaled_train.shape[1])
    reducer = PCA(n_components=n_components, random_state=0).fit(scaled_train)
    return reducer.transform(scaler.transform(x))


def fit_group_classifier(x: np.ndarray, labels: np.ndarray) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=0.25, class_weight="balanced", solver="liblinear", max_iter=5000)
    model.fit(scaler.transform(x), labels)
    return scaler, model


def predict_group_classifier(
    scaler: StandardScaler, model: LogisticRegression, x: np.ndarray
) -> np.ndarray:
    return model.predict_proba(scaler.transform(np.atleast_2d(x)))[:, 1]


def evaluate_site_level(
    x: np.ndarray,
    labels_by_group: dict[tuple[str, str], int],
    sites: np.ndarray,
    datasets: np.ndarray,
    strands: np.ndarray,
) -> tuple[dict[tuple[str, str], float], dict[str, np.ndarray]]:
    predictions: dict[tuple[str, str], float] = {}
    reduced_by_fold: dict[str, np.ndarray] = {}
    for held_site in sorted(set(sites)):
        train_reads = sites != held_site
        reduced = reduce_fold(x, train_reads)
        reduced_by_fold[held_site] = reduced
        train_groups = [group for group in GROUPS if group[0] != held_site]
        train_x = np.vstack(
            [aggregate_group(reduced, sites, datasets, strands, *group) for group in train_groups]
        )
        train_y = np.asarray([labels_by_group[group] for group in train_groups])
        scaler, model = fit_group_classifier(train_x, train_y)
        for dataset in ("UNMOD", "MOD"):
            group = (held_site, dataset)
            test_x = aggregate_group(reduced, sites, datasets, strands, *group)
            predictions[group] = float(predict_group_classifier(scaler, model, test_x)[0])
    return predictions, reduced_by_fold


def ordered_scores(
    predictions: dict[tuple[str, str], float], labels_by_group: dict[tuple[str, str], int]
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray([labels_by_group[group] for group in GROUPS], dtype=int)
    score = np.asarray([predictions[group] for group in GROUPS], dtype=float)
    return y, score


def bootstrap_probabilities(
    reduced_by_fold: dict[str, np.ndarray],
    labels_by_group: dict[tuple[str, str], int],
    sites: np.ndarray,
    datasets: np.ndarray,
    strands: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[tuple[str, str], np.ndarray]:
    rng = np.random.default_rng(seed)
    output = {group: np.empty(iterations, dtype=float) for group in GROUPS}
    for held_site, reduced in reduced_by_fold.items():
        train_groups = [group for group in GROUPS if group[0] != held_site]
        for iteration in range(iterations):
            train_x = np.vstack(
                [
                    bootstrap_group(reduced, sites, datasets, strands, *group, rng)
                    for group in train_groups
                ]
            )
            train_y = np.asarray([labels_by_group[group] for group in train_groups])
            scaler, model = fit_group_classifier(train_x, train_y)
            for dataset in ("UNMOD", "MOD"):
                group = (held_site, dataset)
                test_x = bootstrap_group(reduced, sites, datasets, strands, *group, rng)
                output[group][iteration] = predict_group_classifier(scaler, model, test_x)[0]
    return output


def exact_site_swap_test(
    x: np.ndarray,
    sites: np.ndarray,
    datasets: np.ndarray,
    strands: np.ndarray,
    observed_auc: float,
) -> tuple[float, np.ndarray]:
    site_order = sorted(set(sites))
    null = []
    for swaps in itertools.product((False, True), repeat=len(site_order)):
        labels = {}
        for site, swap in zip(site_order, swaps):
            labels[(site, "UNMOD")] = int(swap)
            labels[(site, "MOD")] = int(not swap)
        predictions, _ = evaluate_site_level(x, labels, sites, datasets, strands)
        y, score = ordered_scores(predictions, labels)
        null.append(roc_auc_score(y, score))
    null_array = np.asarray(null)
    pvalue = np.count_nonzero(null_array >= observed_auc - 1e-12) / len(null_array)
    return float(pvalue), null_array


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, width=0.8, color="#555555")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    plot_dir = Path(args.plot_dir) if args.plot_dir else out_dir / "plot"
    plot_dir.mkdir(parents=True, exist_ok=True)

    windows = read_jsonl(Path(args.windows))
    token_rows = read_jsonl(Path(args.tokens))
    archive = np.load(args.embeddings)
    embeddings = archive["embeddings"].astype(np.float64)
    sites = archive["site_ids"].astype(str)
    datasets = archive["datasets"].astype(str)
    strands = archive["strands"].astype(str)
    read_ids = archive["read_ids"].astype(str)
    if len(set(read_ids)) != len(read_ids):
        raise RuntimeError("Read IDs must be unique before site aggregation")

    keys = [(row["site_id"], row["dataset"], row["read_id"]) for row in windows]
    token_keys = [(row["site_id"], row["dataset"], row["read_id"]) for row in token_rows]
    archive_keys = list(zip(sites, datasets, read_ids))
    if keys != token_keys or keys != archive_keys:
        raise RuntimeError("Input record orders do not match")

    signal_stats = np.asarray(
        [
            [
                len(row["signal"]), np.mean(row["signal"]), np.std(row["signal"]),
                np.median(row["signal"]),
                np.mean(np.abs(np.diff(row["signal"]))) if len(row["signal"]) > 1 else 0.0,
            ]
            for row in windows
        ],
        dtype=float,
    )
    observed_token_ids = sorted(
        {int(token) for row in token_rows for token in row["raw_signal_tokens"]}
    )
    token_column = {token: index for index, token in enumerate(observed_token_ids)}
    token_hist = np.zeros((len(token_rows), len(observed_token_ids)), dtype=float)
    for index, row in enumerate(token_rows):
        values, counts = np.unique(np.asarray(row["raw_signal_tokens"], dtype=int), return_counts=True)
        columns = [token_column[int(value)] for value in values]
        token_hist[index, columns] = counts
    token_hist = normalize(token_hist, norm="l1")

    labels_by_group = {(site, dataset): int(dataset == "MOD") for site, dataset in GROUPS}
    embedding_method = f"{args.model_label} embedding"
    feature_sets = {
        "Signal stats": signal_stats,
        "Token frequency": token_hist,
        embedding_method: embeddings,
    }
    predictions_by_method: dict[str, dict[tuple[str, str], float]] = {}
    reduced_embedding_folds: dict[str, np.ndarray] | None = None
    metrics = []
    for method, x in feature_sets.items():
        predictions, reduced = evaluate_site_level(x, labels_by_group, sites, datasets, strands)
        predictions_by_method[method] = predictions
        if method == embedding_method:
            reduced_embedding_folds = reduced
        y, score = ordered_scores(predictions, labels_by_group)
        metrics.append(
            {
                "method": method,
                "primary_units": 10,
                "auroc": float(roc_auc_score(y, score)),
                "balanced_accuracy": float(balanced_accuracy_score(y, score >= 0.5)),
                "accuracy": float(accuracy_score(y, score >= 0.5)),
            }
        )
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(out_dir / "multiread_site_level_metrics.tsv", sep="\t", index=False, float_format="%.6f")

    embedding_auc = float(metrics_df.loc[metrics_df["method"] == embedding_method, "auroc"].iloc[0])
    permutation_p, permutation_null = exact_site_swap_test(
        embeddings, sites, datasets, strands, embedding_auc
    )
    np.savez_compressed(out_dir / "multiread_site_swap_null.npz", auroc=permutation_null)
    assert reduced_embedding_folds is not None
    bootstrap_scores = bootstrap_probabilities(
        reduced_embedding_folds, labels_by_group, sites, datasets, strands,
        args.bootstraps, args.seed,
    )

    aggregate_embeddings = np.vstack(
        [aggregate_group(embeddings, sites, datasets, strands, *group) for group in GROUPS]
    )
    aggregate_scaled = StandardScaler().fit_transform(aggregate_embeddings)
    pca = PCA(n_components=2, random_state=args.seed)
    aggregate_pca = pca.fit_transform(aggregate_scaled)

    count_rows = []
    result_rows = []
    embedding_predictions = predictions_by_method[embedding_method]
    for index, group in enumerate(GROUPS):
        site, dataset = group
        plus_n = int(np.count_nonzero((sites == site) & (datasets == dataset) & (strands == "+")))
        minus_n = int(np.count_nonzero((sites == site) & (datasets == dataset) & (strands == "-")))
        low, median, high = np.quantile(bootstrap_scores[group], [0.025, 0.5, 0.975])
        count_rows.append(
            {"site_id": site, "dataset": dataset, "plus_reads": plus_n, "minus_reads": minus_n, "total_reads": plus_n + minus_n}
        )
        result_rows.append(
            {
                "site_id": site, "dataset": dataset, "label": labels_by_group[group],
                "total_reads": plus_n + minus_n, "plus_reads": plus_n, "minus_reads": minus_n,
                "pc1": aggregate_pca[index, 0], "pc2": aggregate_pca[index, 1],
                "oof_mod_probability": embedding_predictions[group],
                "bootstrap_probability_median": median,
                "bootstrap_probability_ci_low": low,
                "bootstrap_probability_ci_high": high,
            }
        )
    result_df = pd.DataFrame(result_rows)
    result_df.to_csv(out_dir / "multiread_site_aggregates.tsv", sep="\t", index=False, float_format="%.6f")
    count_df = pd.DataFrame(count_rows)

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
        indices = [GROUPS.index((site, dataset)) for dataset in ("UNMOD", "MOD")]
        ax_a.plot(
            aggregate_pca[indices, 0], aggregate_pca[indices, 1],
            color=SITE_COLORS[site], lw=1.2, alpha=0.75, zorder=1,
        )
        for dataset, index in zip(("UNMOD", "MOD"), indices):
            ax_a.scatter(
                aggregate_pca[index, 0], aggregate_pca[index, 1], s=52,
                color=COLORS[dataset], marker="o" if dataset == "UNMOD" else "s",
                edgecolors=SITE_COLORS[site], linewidths=1.3, zorder=2,
            )
        midpoint = aggregate_pca[indices].mean(axis=0)
        ax_a.text(midpoint[0], midpoint[1], site, color=SITE_COLORS[site], fontsize=7, ha="center", va="bottom")
    ax_a.axhline(0, color="#D0D0D0", lw=0.6, zorder=0)
    ax_a.axvline(0, color="#D0D0D0", lw=0.6, zorder=0)
    ax_a.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax_a.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax_a.set_title(f"{args.model_label} all-read site aggregates")
    handles = [
        Line2D([0], [0], marker="s", linestyle="", color=COLORS["MOD"], label="MOD", markersize=6),
        Line2D([0], [0], marker="o", linestyle="", color=COLORS["UNMOD"], label="UNMOD", markersize=6),
    ]
    ax_a.legend(handles=handles, frameon=False, loc="best")
    style_axes(ax_a)

    site_order = sorted(set(sites))
    for site_index, site in enumerate(site_order, start=1):
        for dataset, offset in (("UNMOD", -0.14), ("MOD", 0.14)):
            row = result_df[(result_df.site_id == site) & (result_df.dataset == dataset)].iloc[0]
            x_pos = site_index + offset
            ax_b.errorbar(
                x_pos, row.oof_mod_probability,
                yerr=np.asarray([[row.oof_mod_probability - row.bootstrap_probability_ci_low],
                                 [row.bootstrap_probability_ci_high - row.oof_mod_probability]]),
                fmt="o" if dataset == "UNMOD" else "s", markersize=5,
                color=COLORS[dataset], ecolor=COLORS[dataset], elinewidth=1.2,
                capsize=2.5, markeredgecolor="white", markeredgewidth=0.5,
            )
    ax_b.axhline(0.5, color="#777777", lw=0.8, ls="--")
    ax_b.set_xticks(range(1, 6), site_order)
    ax_b.set_ylim(-0.03, 1.03)
    ax_b.set_xlabel("Held-out genomic site")
    ax_b.set_ylabel("Aggregate MOD probability")
    ax_b.set_title("Prediction from multiple reads")
    style_axes(ax_b)

    method_order = ["Signal stats", "Token frequency", embedding_method]
    method_colors = {
        "Signal stats": METHOD_COLORS["Signal stats"],
        "Token frequency": METHOD_COLORS["Token frequency"],
        embedding_method: METHOD_COLORS["OLMo embedding"],
    }
    metric_columns = [("auroc", "AUROC"), ("balanced_accuracy", "Balanced accuracy")]
    for metric_index, (column, title) in enumerate(metric_columns):
        x_base = np.arange(3) + metric_index * 4.2
        for method_index, method in enumerate(method_order):
            value = float(metrics_df.loc[metrics_df.method == method, column].iloc[0])
            ax_c.scatter(
                x_base[method_index], value, s=65, color=method_colors[method],
                edgecolors="white", linewidths=0.6, zorder=3,
            )
        ax_c.text(x_base.mean(), 0.99, title, ha="center", va="top", fontsize=8, fontweight="bold")
    ax_c.axhline(0.5, color="#777777", lw=0.8, ls="--")
    ax_c.set_xlim(-0.7, 6.9)
    ax_c.set_ylim(0.25, 1.02)
    ax_c.set_xticks(
        list(np.arange(3)) + list(np.arange(3) + 4.2),
        ["Stats", "Tokens", args.model_label] * 2,
        rotation=25,
    )
    ax_c.set_ylabel("Site-level score")
    ax_c.set_title("Leave-one-site-out discrimination")
    ax_c.text(
        0.98, 0.80, f"{args.model_label} exact site-swap P={permutation_p:.3g}",
        transform=ax_c.transAxes, ha="right", va="bottom", fontsize=7, color="#444444",
    )
    style_axes(ax_c)

    bar_y = np.arange(len(site_order))
    bar_height = 0.34
    for dataset, offset in (("UNMOD", -bar_height / 2), ("MOD", bar_height / 2)):
        values = [int(count_df[(count_df.site_id == site) & (count_df.dataset == dataset)].total_reads.iloc[0]) for site in site_order]
        plus = [int(count_df[(count_df.site_id == site) & (count_df.dataset == dataset)].plus_reads.iloc[0]) for site in site_order]
        ax_d.barh(bar_y + offset, plus, height=bar_height, color=COLORS[dataset], alpha=0.95, label=dataset)
        ax_d.barh(
            bar_y + offset, np.asarray(values) - np.asarray(plus), left=plus, height=bar_height,
            color=COLORS[dataset], alpha=0.45, hatch="///", edgecolor="white", linewidth=0.3,
        )
    ax_d.set_yticks(bar_y, [f"{site}  {site_pos[site] / 1e6:.1f} Mb" for site in site_order])
    ax_d.invert_yaxis()
    ax_d.set_xlabel("Reads contributing to aggregate")
    ax_d.set_title("Aggregate read support")
    dataset_legend = ax_d.legend(frameon=False, loc="lower right", title="Dataset")
    ax_d.add_artist(dataset_legend)
    strand_handles = [
        Line2D([0], [0], color="#777777", lw=5, alpha=0.95, label="+ strand"),
        Line2D([0], [0], color="#777777", lw=5, alpha=0.45, label="− strand"),
    ]
    ax_d.legend(handles=strand_handles, frameon=False, loc="center right", title="Bar segment")
    style_axes(ax_d)

    for label, ax in zip("abcd", axes.flat):
        ax.text(-0.13, 1.06, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")

    stem = plot_dir / args.figure_stem
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    summary = {
        "primary_statistical_units": 10,
        "sites": 5,
        "conditions_per_site": 2,
        "underlying_unique_reads": len(set(read_ids)),
        "model_label": args.model_label,
        "observed_token_vocabulary": len(observed_token_ids),
        "aggregation": "mean within strand, then equal-weight mean of positive and negative strand centroids",
        "single_read_classification_used": False,
        "validation": "leave one genomic site out",
        "bootstrap_replicates": args.bootstraps,
        "bootstrap_role": "confidence intervals only; not independent primary observations",
        "metrics": metrics_df.set_index("method").to_dict(orient="index"),
        "embedding_exact_site_swap_test": {
            "assignments": 32,
            "observed_auroc": embedding_auc,
            "pvalue": permutation_p,
            "null_mean": float(permutation_null.mean()),
        },
        "interpretation_guardrail": "Only five sites and separate MOD/UNMOD sequencing runs; do not claim a 5mC-specific mechanism.",
    }
    (out_dir / "multiread_analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
