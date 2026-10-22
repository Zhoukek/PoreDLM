#!/usr/bin/env python3
"""Bootstrap CKA intervals and plot the extracted layerwise feature summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def linear_cka(x, y):
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    k = x @ x.T
    l = y @ y.T
    den = ((k * k).sum() * (l * l).sum()) ** 0.5
    return float((k * l).sum() / den) if den else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=Path("feature_analysis"))
    parser.add_argument("--paired-dir", type=Path, default=Path("feature_analysis/paired_features_n128"))
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261019)
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import numpy as np

    paired = np.load(args.paired_dir / "paired_pooled_features.npz")
    foundation_features_path = args.analysis_dir / "foundation_hidden_states" / "foundation_layerwise_pooled_features.npz"
    foundation_npz = np.load(foundation_features_path)
    with (args.analysis_dir / "foundation_hidden_states" / "foundation_layerwise_summary.json").open() as handle:
        foundation = json.load(handle)
    n = len(paired["row_indices"])
    rng = np.random.default_rng(args.seed)
    comparisons = {
        "Bonito mean vs BERT input mean": (paired["bonito_mean"], paired["foundation_input_embedding_mean"]),
        "Bonito mean vs BERT final mean": (paired["bonito_mean"], paired["foundation_last_hidden_mean"]),
        "Bonito mean vs DLM ODE mean": (paired["bonito_mean"], paired["dlm_ode_hidden_mean"]),
        "Bonito std vs BERT input std": (paired["bonito_std"], paired["foundation_input_embedding_std"]),
        "Bonito std vs BERT final std": (paired["bonito_std"], paired["foundation_last_hidden_std"]),
        "Bonito std vs DLM ODE std": (paired["bonito_std"], paired["dlm_ode_hidden_std"]),
        "BERT input mean vs final mean": (paired["foundation_input_embedding_mean"], paired["foundation_last_hidden_mean"]),
        "BERT final mean vs DLM ODE mean": (paired["foundation_last_hidden_mean"], paired["dlm_ode_hidden_mean"]),
        "BERT input mean vs DLM ODE mean": (paired["foundation_input_embedding_mean"], paired["dlm_ode_hidden_mean"]),
    }
    cka_results = {}
    for name, (x, y) in comparisons.items():
        estimate = linear_cka(x, y)
        draws = np.empty(args.bootstrap, dtype=np.float64)
        for i in range(args.bootstrap):
            indexes = rng.integers(0, n, size=n)
            draws[i] = linear_cka(x[indexes], y[indexes])
        cka_results[name] = {
            "estimate": estimate,
            "bootstrap_ci_95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            "bootstrap_replicates": args.bootstrap,
        }

    delta_draws = np.empty(args.bootstrap, dtype=np.float64)
    b_mean = paired["bonito_mean"]
    input_mean = paired["foundation_input_embedding_mean"]
    final_mean = paired["foundation_last_hidden_mean"]
    dlm_mean = paired["dlm_ode_hidden_mean"]
    for i in range(args.bootstrap):
        indexes = rng.integers(0, n, size=n)
        delta_draws[i] = linear_cka(b_mean[indexes], final_mean[indexes]) - linear_cka(b_mean[indexes], input_mean[indexes])
    dlm_delta_draws = np.empty(args.bootstrap, dtype=np.float64)
    for i in range(args.bootstrap):
        indexes = rng.integers(0, n, size=n)
        dlm_delta_draws[i] = linear_cka(b_mean[indexes], dlm_mean[indexes]) - linear_cka(b_mean[indexes], final_mean[indexes])

    summary = {
        "sample_count": n,
        "sample_ids_match": bool(len(paired["sample_ids"]) == n),
        "bootstrap_seed": args.seed,
        "cka": cka_results,
        "final_hidden_minus_input_cka": {
            "estimate": cka_results["Bonito mean vs BERT final mean"]["estimate"] - cka_results["Bonito mean vs BERT input mean"]["estimate"],
            "paired_bootstrap_ci_95": [float(np.quantile(delta_draws, 0.025)), float(np.quantile(delta_draws, 0.975))],
            "bootstrap_replicates": args.bootstrap,
        },
        "dlm_ode_minus_bert_final_cka": {
            "estimate": cka_results["Bonito mean vs DLM ODE mean"]["estimate"] - cka_results["Bonito mean vs BERT final mean"]["estimate"],
            "paired_bootstrap_ci_95": [float(np.quantile(dlm_delta_draws, 0.025)), float(np.quantile(dlm_delta_draws, 0.975))],
            "bootstrap_replicates": args.bootstrap,
        },
        "warning": "Chunk bootstrap intervals describe these 128 sampled chunks; CKA is temporal-pooled sample geometry, not tokenwise alignment or basecalling performance.",
    }
    (args.paired_dir / "paired_feature_bootstrap_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    paired_summary_path = args.paired_dir / "paired_feature_summary.json"
    paired_summary = json.loads(paired_summary_path.read_text(encoding="utf-8"))
    paired_summary["linear_cka_on_chunk_mean_features"] = {
        "bonito_vs_foundation_input_embedding": cka_results["Bonito mean vs BERT input mean"]["estimate"],
        "bonito_vs_foundation_last_hidden": cka_results["Bonito mean vs BERT final mean"]["estimate"],
        "bonito_vs_dlm_ode_hidden": cka_results["Bonito mean vs DLM ODE mean"]["estimate"],
        "foundation_input_embedding_vs_last_hidden": cka_results["BERT input mean vs final mean"]["estimate"],
        "foundation_last_hidden_vs_dlm_ode_hidden": cka_results["BERT final mean vs DLM ODE mean"]["estimate"],
    }
    paired_summary["cka_method"] = "linear CKA with feature columns centered before sample Gram matrices"
    paired_summary_path.write_text(json.dumps(paired_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    layer_means = foundation_npz["mean_features"].astype(np.float64)
    layer_names = foundation_npz["layer_names"].tolist()
    layerwise_cka = np.empty((len(layer_names), len(layer_names)), dtype=np.float32)
    for i in range(len(layer_names)):
        for j in range(len(layer_names)):
            layerwise_cka[i, j] = linear_cka(layer_means[:, i, :], layer_means[:, j, :])
    np.savez_compressed(
        foundation_features_path,
        row_indices=foundation_npz["row_indices"],
        sample_ids=foundation_npz["sample_ids"],
        layer_names=foundation_npz["layer_names"],
        mean_features=foundation_npz["mean_features"],
        std_features=foundation_npz["std_features"],
        layerwise_cka=layerwise_cka,
    )

    layers = foundation["layers"]
    labels = ["emb"] + [str(i) for i in range(1, 13)] + ["final"]
    adjacent = [layer["mean_adjacent_token_cosine"] for layer in layers]
    norms = [layer["mean_token_l2_norm"] for layer in layers]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    ax = axes[0]
    ax.plot(range(len(labels)), adjacent, "o-", color="#2474B5", linewidth=2, label="Adjacent token cosine")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlabel("BERT representation layer")
    ax.set_ylabel("Mean adjacent-token cosine", color="#2474B5")
    ax.tick_params(axis="y", labelcolor="#2474B5")
    ax.grid(alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(range(len(labels)), norms, "s--", color="#CC6A27", linewidth=1.8, label="Token L2 norm")
    ax2.set_ylabel("Mean token L2 norm", color="#CC6A27")
    ax2.tick_params(axis="y", labelcolor="#CC6A27")
    ax.set_title("Base-model hidden-state geometry (32 chunks)")

    ax = axes[1]
    names = list(cka_results)
    estimates = [cka_results[name]["estimate"] for name in names]
    y = np.arange(len(names))
    ax.scatter(estimates, y, color="#6B4C9A", s=42, zorder=3)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Linear CKA (chunk mean/std features)")
    ax.grid(axis="x", alpha=0.25)
    ax.set_title("Paired three-model geometry (128 chunks)")
    delta = summary["final_hidden_minus_input_cka"]
    delta_low, delta_high = delta["paired_bootstrap_ci_95"]
    ax.text(
        0.98,
        0.04,
        f"BERT final − input vs Bonito: {delta['estimate']:.3f}\nDLM ODE − BERT final vs Bonito: {summary['dlm_ode_minus_bert_final_cka']['estimate']:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#999999", "alpha": 0.95},
    )
    fig.suptitle("Feature-space comparison on evenly spaced chunks", fontsize=15)
    fig.savefig(args.analysis_dir / "feature_space_summary.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved {args.analysis_dir / 'feature_space_summary.png'}")


if __name__ == "__main__":
    main()
