#!/usr/bin/env python3
"""Create coordinate-invariant visual comparisons of Bonito, BERT, and DLM.

The pooled representation axes of independently trained models have no shared
meaning.  This script therefore shows both each model's own PCA geometry and
an orthogonal-Procrustes aligned overlay.  It also compares each model's
sample-to-sample distance matrix (RSA), which is invariant to axis rotation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODEL_NAMES = ("Bonito", "V543 BERT final", "DLM ODE")
MODEL_COLORS = {"Bonito": "#2878B5", "V543 BERT final": "#E07A32", "DLM ODE": "#2A9D62"}
MODEL_MARKERS = {"Bonito": "o", "V543 BERT final": "X", "DLM ODE": "^"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paired-npz",
        type=Path,
        default=Path("../feature_analysis/paired_features_n128/paired_pooled_features.npz"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    return parser.parse_args()


def centered_frobenius_normalized(x):
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(x)
    if norm == 0:
        raise ValueError("A representation has zero centered Frobenius norm.")
    return x / norm


def pca_2d(x):
    import numpy as np

    centered = x - x.mean(axis=0, keepdims=True)
    u, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    coordinates = u[:, :2] * singular_values[:2]
    total_variance = float((singular_values**2).sum())
    explained = (singular_values[:2] ** 2 / total_variance) if total_variance else np.zeros(2)
    return coordinates, explained


def procrustes_align(source, target):
    """Rotate source into target's axes without rescaling either geometry."""
    import numpy as np

    u, _, vt = np.linalg.svd(source.T @ target, full_matrices=False)
    rotation = u @ vt
    aligned = source @ rotation
    return aligned, float(np.linalg.norm(aligned - target) / np.linalg.norm(target))


def pairwise_distances(x):
    import numpy as np

    squared = np.sum(x * x, axis=1, keepdims=True)
    distances_sq = np.maximum(squared + squared.T - 2.0 * (x @ x.T), 0.0)
    return np.sqrt(distances_sq)


def linear_cka(x, y):
    import numpy as np

    k = x @ x.T
    l = y @ y.T
    denominator = float(((k * k).sum() * (l * l).sum()) ** 0.5)
    return float((k * l).sum() / denominator) if denominator else float("nan")


def upper_triangle(x):
    import numpy as np

    return x[np.triu_indices_from(x, k=1)]


def correlation(x, y):
    import numpy as np

    return float(np.corrcoef(x, y)[0, 1])


def representation_metrics(representations):
    import numpy as np

    normalized = {name: centered_frobenius_normalized(value) for name, value in representations.items()}
    distances = {name: pairwise_distances(value) for name, value in normalized.items()}
    cka = np.empty((len(MODEL_NAMES), len(MODEL_NAMES)), dtype=float)
    rsa = np.empty_like(cka)
    for i, left_name in enumerate(MODEL_NAMES):
        for j, right_name in enumerate(MODEL_NAMES):
            cka[i, j] = linear_cka(normalized[left_name], normalized[right_name])
            rsa[i, j] = correlation(
                upper_triangle(distances[left_name]), upper_triangle(distances[right_name])
            )
    return normalized, distances, cka, rsa


def plot_pca_spaces(representations, row_indices, output_path):
    import matplotlib.pyplot as plt
    import numpy as np

    normalized, _, _, _ = representation_metrics(representations)
    reference = normalized["Bonito"]
    aligned = {"Bonito": reference}
    residuals = {"Bonito": 0.0}
    for name in MODEL_NAMES[1:]:
        aligned[name], residuals[name] = procrustes_align(normalized[name], reference)
    combined = np.concatenate([aligned[name] for name in MODEL_NAMES], axis=0)
    common_coordinates, common_explained = pca_2d(combined)
    n = len(row_indices)
    common_coordinates = {
        name: common_coordinates[i * n:(i + 1) * n]
        for i, name in enumerate(MODEL_NAMES)
    }

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 11), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    for axis, name in zip(axes.flat[:3], MODEL_NAMES):
        coordinates, explained = pca_2d(normalized[name])
        points = axis.scatter(
            coordinates[:, 0], coordinates[:, 1], c=row_indices, cmap=cmap, s=34,
            edgecolor="white", linewidth=0.25, alpha=0.92,
        )
        axis.set_title(f"{name}: own PCA ({explained[0] * 100:.1f}% + {explained[1] * 100:.1f}%)")
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        axis.grid(alpha=0.2)
        colorbar = fig.colorbar(points, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label("Original chunk row")

    axis = axes[1, 1]
    for name in MODEL_NAMES:
        coordinates = common_coordinates[name]
        axis.scatter(
            coordinates[:, 0], coordinates[:, 1], label=name, marker=MODEL_MARKERS[name],
            c=MODEL_COLORS[name], s=42, alpha=0.70, edgecolor="white", linewidth=0.25,
        )
    # Connecting only a sparse, even subset keeps sample correspondence visible.
    for index in np.linspace(0, n - 1, 16, dtype=int):
        polyline = np.stack([common_coordinates[name][index] for name in MODEL_NAMES])
        axis.plot(polyline[:, 0], polyline[:, 1], color="#666666", alpha=0.23, linewidth=0.8, zorder=0)
    axis.set_title(
        "Common PCA after orthogonal alignment to Bonito\n"
        f"(PC1 + PC2 = {(common_explained.sum() * 100):.1f}%; "
        f"BERT residual {residuals['V543 BERT final']:.3f}, DLM residual {residuals['DLM ODE']:.3f})"
    )
    axis.set_xlabel("Common PC1")
    axis.set_ylabel("Common PC2")
    axis.grid(alpha=0.2)
    axis.legend(loc="best", frameon=True)
    fig.suptitle(
        "Three model pooled mean-representation spaces: 128 matched chunks\n"
        "Each top-panel PCA has its own axes; lower-right alignment is only for visualizing correspondence.",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return residuals, common_explained


def plot_rsa(representations, output_path):
    import matplotlib.pyplot as plt
    import numpy as np

    _, distances, cka, rsa = representation_metrics(representations)
    all_values = np.concatenate([upper_triangle(distances[name]) for name in MODEL_NAMES])
    vmin, vmax = float(np.quantile(all_values, 0.01)), float(np.quantile(all_values, 0.99))
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    images = []
    for axis, name in zip(axes.flat[:3], MODEL_NAMES):
        image = axis.imshow(distances[name], cmap="magma", vmin=vmin, vmax=vmax, interpolation="nearest")
        images.append(image)
        axis.set_title(f"{name}: pairwise chunk distance")
        axis.set_xlabel("Matched chunk index")
        axis.set_ylabel("Matched chunk index")
    colorbar = fig.colorbar(images[-1], ax=axes.flat[:3], shrink=0.82)
    colorbar.set_label("Distance after centering and global Frobenius normalization")

    axis = axes[1, 1]
    image = axis.imshow(rsa, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    axis.set_xticks(range(len(MODEL_NAMES)), MODEL_NAMES, rotation=20, ha="right")
    axis.set_yticks(range(len(MODEL_NAMES)), MODEL_NAMES)
    axis.set_title("RSA: correlation of pairwise distances")
    for i in range(len(MODEL_NAMES)):
        for j in range(len(MODEL_NAMES)):
            axis.text(j, i, f"{rsa[i, j]:.3f}", ha="center", va="center",
                      color="white" if rsa[i, j] > 0.58 else "black", fontsize=11)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Pearson r")
    fig.suptitle(
        "Representation similarity analysis of matched chunks\n"
        "Distance structure, unlike raw coordinate axes, can be compared between independently trained models.",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return cka, rsa


def matrix_dict(matrix):
    return {
        left: {right: float(matrix[i, j]) for j, right in enumerate(MODEL_NAMES)}
        for i, left in enumerate(MODEL_NAMES)
    }


def main() -> None:
    args = parse_args()
    import numpy as np

    paired = np.load(args.paired_npz)
    row_indices = paired["row_indices"].astype(np.int64)
    mean_representations = {
        "Bonito": paired["bonito_mean"],
        "V543 BERT final": paired["foundation_last_hidden_mean"],
        "DLM ODE": paired["dlm_ode_hidden_mean"],
    }
    std_representations = {
        "Bonito": paired["bonito_std"],
        "V543 BERT final": paired["foundation_last_hidden_std"],
        "DLM ODE": paired["dlm_ode_hidden_std"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mean_spaces_path = args.output_dir / "three_model_mean_representation_spaces.png"
    rsa_path = args.output_dir / "three_model_representation_rsa.png"
    residuals, common_explained = plot_pca_spaces(mean_representations, row_indices, mean_spaces_path)
    mean_normalized, _, mean_cka, mean_rsa = representation_metrics(mean_representations)
    _, _, std_cka, std_rsa = representation_metrics(std_representations)
    cka_from_rsa, rsa_from_rsa = plot_rsa(mean_representations, rsa_path)
    if not np.allclose(mean_cka, cka_from_rsa) or not np.allclose(mean_rsa, rsa_from_rsa):
        raise RuntimeError("Internal visualization metrics disagree.")
    result = {
        "sample_count": int(len(row_indices)),
        "row_indices": row_indices.tolist(),
        "representation": "Temporal mean/std pooled features across the 1,500 signal-token positions; all models use identical chunks.",
        "ode_parameters": {
            "ode_steps": 1,
            "ode_start_t": 0.97,
            "ode_self_cond_cfg_scale": 0.5,
        },
        "mean_linear_cka": matrix_dict(mean_cka),
        "std_linear_cka": matrix_dict(std_cka),
        "mean_distance_rsa_pearson": matrix_dict(mean_rsa),
        "std_distance_rsa_pearson": matrix_dict(std_rsa),
        "procrustes_residual_to_bonito_mean": residuals,
        "common_pca_explained_variance_ratio": [float(value) for value in common_explained],
        "figures": {
            "mean_representation_spaces": mean_spaces_path.name,
            "mean_representation_rsa": rsa_path.name,
        },
        "interpretation": (
            "Own-PCA panels are separately projected and their axes must not be compared numerically. "
            "The aligned overlay uses an orthogonal Procrustes rotation only to show corresponding chunks. "
            "RSA compares pairwise chunk distances and is coordinate-rotation invariant; it is still a pooled-feature analysis, not basecalling accuracy."
        ),
    }
    summary_path = args.output_dir / "three_model_visualization_summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
