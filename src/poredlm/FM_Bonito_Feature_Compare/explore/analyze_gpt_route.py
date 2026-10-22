#!/usr/bin/env python3
"""Compare V615 GPT/OLMo features and training logs with Bonito, BERT, and DLM.

All feature comparisons use the exact 128 chunks already paired for the
Bonito/BERT/DLM analysis.  The BERT feature used in the four-way basecall plot
is ``transformer_layer_12`` because Stage-4's ``hidden_layer=-1`` selects
``outputs.hidden_states[-1]``.  This avoids silently comparing GPT's actual
basecall feature against a different BERT output representation.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paired-npz", type=Path,
        default=SCRIPT_DIR.parent / "feature_analysis/paired_features_n128/paired_pooled_features.npz",
    )
    parser.add_argument(
        "--bert-layerwise-npz", type=Path,
        default=SCRIPT_DIR / "bert_layerwise_n128/foundation_layerwise_pooled_features.npz",
    )
    parser.add_argument(
        "--bert-layerwise-summary", type=Path,
        default=SCRIPT_DIR / "bert_layerwise_n128/foundation_layerwise_summary.json",
    )
    parser.add_argument(
        "--gpt-layerwise-npz", type=Path,
        default=SCRIPT_DIR / "gpt_layerwise_n128/gpt_layerwise_pooled_features.npz",
    )
    parser.add_argument(
        "--gpt-layerwise-summary", type=Path,
        default=SCRIPT_DIR / "gpt_layerwise_n128/gpt_layerwise_summary.json",
    )
    parser.add_argument(
        "--bonito-temporal-summary", type=Path,
        default=SCRIPT_DIR / "bonito_temporal_smoothness_n128.json",
    )
    parser.add_argument(
        "--gpt-v615-log", type=Path,
        default=SCRIPT_DIR.parent.parent / "training_public/stage4_basecall_for_RNA/runs/"
                "HF_RSQ741C12A529_MNAOLMO_V615_DNA_part/train.log",
    )
    parser.add_argument(
        "--gpt-v613-log", type=Path,
        default=SCRIPT_DIR.parent.parent / "training_public/stage4_basecall_for_RNA/runs/"
                "HF_RSQ741C12A529_DNAOLMO_V613/nohup.out",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    return parser.parse_args()


def center_and_normalize(x):
    import numpy as np

    result = np.asarray(x, dtype=np.float64)
    result = result - result.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(result)
    if norm == 0:
        raise ValueError("Cannot normalize a zero representation")
    return result / norm


def linear_cka(x, y) -> float:
    x = center_and_normalize(x)
    y = center_and_normalize(y)
    k = x @ x.T
    l = y @ y.T
    denom = float(((k * k).sum() * (l * l).sum()) ** 0.5)
    return float((k * l).sum() / denom) if denom else float("nan")


def pairwise_distance(x):
    import numpy as np

    x = center_and_normalize(x)
    squared = (x * x).sum(axis=1, keepdims=True)
    return np.sqrt(np.maximum(squared + squared.T - 2 * (x @ x.T), 0.0))


def distance_rsa(x, y) -> float:
    import numpy as np

    dx = pairwise_distance(x)
    dy = pairwise_distance(y)
    tri = np.triu_indices_from(dx, k=1)
    return float(np.corrcoef(dx[tri], dy[tri])[0, 1])


def procrustes_residual(source, target) -> float:
    import numpy as np

    source = center_and_normalize(source)
    target = center_and_normalize(target)
    u, _, vt = np.linalg.svd(source.T @ target, full_matrices=False)
    return float(np.linalg.norm(source @ (u @ vt) - target))


def pca_2d(x):
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    u, singular_values, _ = np.linalg.svd(x, full_matrices=False)
    coords = u[:, :2] * singular_values[:2]
    total = float((singular_values ** 2).sum())
    explained = singular_values[:2] ** 2 / total if total else np.zeros(2)
    return coords, explained


def procrustes_coordinates(representations):
    import numpy as np

    names = list(representations)
    reference = center_and_normalize(representations[names[0]])
    aligned = [reference]
    residuals = {names[0]: 0.0}
    for name in names[1:]:
        source = center_and_normalize(representations[name])
        u, _, vt = np.linalg.svd(source.T @ reference, full_matrices=False)
        mapped = source @ (u @ vt)
        aligned.append(mapped)
        residuals[name] = float(np.linalg.norm(mapped - reference))
    all_coords, explained = pca_2d(np.concatenate(aligned, axis=0))
    n = reference.shape[0]
    return {name: all_coords[i * n:(i + 1) * n] for i, name in enumerate(names)}, residuals, explained


METRIC_RE = re.compile(
    r"\[(Train|Val|Test)\] epoch=(\d+) loss=([0-9.eE+-]+) acc=([0-9.eE+-]+) "
    r"coverage=([0-9.eE+-]+) blank=([0-9.eE+-]+) nonzero_len=([0-9.eE+-]+)"
)


def latest_log_segment(path: Path) -> tuple[dict | None, list[dict]]:
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    args_rows: list[tuple[int, dict]] = []
    for index, line in enumerate(text.splitlines()):
        if "[Args]" not in line:
            continue
        try:
            args_rows.append((index, ast.literal_eval(line.split("[Args]", 1)[1].strip())))
        except (SyntaxError, ValueError):
            continue
    lines = text.splitlines()
    start = args_rows[-1][0] if args_rows else 0
    args = args_rows[-1][1] if args_rows else None
    metrics = []
    for match in METRIC_RE.finditer("\n".join(lines[start:])):
        kind, epoch, loss, accuracy, coverage, blank, nonzero = match.groups()
        metrics.append({
            "split": kind.lower(), "epoch": int(epoch), "loss": float(loss),
            "accuracy": float(accuracy), "coverage": float(coverage),
            "blank": float(blank), "nonzero_length": float(nonzero),
        })
    return args, metrics


def summarize_log(path: Path) -> dict:
    args, metrics = latest_log_segment(path)
    grouped: dict[int, dict[str, dict]] = {}
    for metric in metrics:
        grouped.setdefault(metric["epoch"], {})[metric["split"]] = metric
    paired = [values for _, values in sorted(grouped.items()) if "train" in values and "val" in values]
    rows = []
    for values in paired:
        train, val = values["train"], values["val"]
        rows.append({
            "epoch": train["epoch"],
            "train_loss": train["loss"], "val_loss": val["loss"],
            "loss_gap_val_minus_train": val["loss"] - train["loss"],
            "train_accuracy": train["accuracy"], "val_accuracy": val["accuracy"],
            "train_coverage": train["coverage"], "val_coverage": val["coverage"],
            "val_selection_score": val["accuracy"] * min(max(val["coverage"], 0.0), 1.0),
        })
    result = {
        "path": str(path),
        "arguments": {
            key: args.get(key) if args else None
            for key in (
                "jsonl_paths", "batch_size", "num_epochs", "backbone_type", "hidden_layer",
                "unfreeze_last_n_layers", "head_lr", "backbone_lr", "warmup_ratio", "group_by", "split_seed",
            )
        },
        "completed_train_val_epochs": rows,
        "completed_epoch_count": len(rows),
    }
    if rows:
        result["first_epoch"] = rows[0]
        result["last_epoch"] = rows[-1]
        result["best_validation_selection"] = max(rows, key=lambda row: row["val_selection_score"])
        result["minimum_validation_loss"] = min(rows, key=lambda row: row["val_loss"])
        result["maximum_loss_gap_val_minus_train"] = max(row["loss_gap_val_minus_train"] for row in rows)
    return result


def plot_four_model_spaces(representations, row_indices, output_path):
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(representations)
    colors = {"Bonito": "#2878B5", "BERT layer 12": "#E07A32", "DLM ODE": "#8467AF", "GPT V615": "#2A9D62"}
    markers = {"Bonito": "o", "BERT layer 12": "X", "DLM ODE": "s", "GPT V615": "^"}
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    plot_handle = None
    for axis, name in zip(axes.flat[:4], names):
        coords, explained = pca_2d(center_and_normalize(representations[name]))
        plot_handle = axis.scatter(coords[:, 0], coords[:, 1], c=row_indices, cmap=cmap, s=27, alpha=0.88,
                                   edgecolor="white", linewidth=0.2)
        axis.set_title(f"{name}: own PCA ({explained.sum() * 100:.1f}% in PC1–2)")
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        axis.grid(alpha=0.2)
    fig.colorbar(plot_handle, ax=axes.flat[:4], shrink=0.8, label="Original chunk row")

    common_coords, residuals, explained = procrustes_coordinates(representations)
    axis = axes[1, 1]
    for name in names:
        coordinates = common_coords[name]
        axis.scatter(coordinates[:, 0], coordinates[:, 1], label=name, c=colors[name], marker=markers[name],
                     s=35, alpha=0.68, edgecolor="white", linewidth=0.2)
    for index in np.linspace(0, len(row_indices) - 1, 12, dtype=int):
        path = np.stack([common_coords[name][index] for name in names])
        axis.plot(path[:, 0], path[:, 1], color="#555555", alpha=0.18, linewidth=0.7, zorder=0)
    axis.set_title("Common PCA after Procrustes alignment to Bonito\n"
                   f"PC1–2={explained.sum() * 100:.1f}%; GPT residual={residuals['GPT V615']:.3f}")
    axis.set_xlabel("Common PC1")
    axis.set_ylabel("Common PC2")
    axis.grid(alpha=0.2)
    axis.legend(loc="best", fontsize=9)

    cka = np.empty((len(names), len(names)))
    rsa = np.empty_like(cka)
    for i, left in enumerate(names):
        for j, right in enumerate(names):
            cka[i, j] = linear_cka(representations[left], representations[right])
            rsa[i, j] = distance_rsa(representations[left], representations[right])
    axis = axes[1, 2]
    image = axis.imshow(rsa, cmap="Blues", vmin=0, vmax=1)
    axis.set_title("RSA: paired chunk-distance correlation")
    axis.set_xticks(range(len(names)), names, rotation=25, ha="right")
    axis.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            axis.text(j, i, f"{rsa[i, j]:.3f}", ha="center", va="center",
                      color="white" if rsa[i, j] > 0.62 else "black", fontsize=9)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Pearson r")
    fig.suptitle("Bonito, BERT, DLM, and GPT pooled mean-representation spaces: 128 matched chunks\n"
                 "Own PCA axes are independent; the bottom-row alignment/RSA panels support cross-model comparison.",
                 fontsize=14)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)
    return cka, rsa, residuals, explained


def plot_gpt_layers(gpt_rows, bert_summary, bonito_summary, output_path):
    import matplotlib.pyplot as plt
    import numpy as np

    labels = ["emb"] + [str(index) for index in range(1, 13)] + ["final"]
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.3), constrained_layout=True)
    axis = axes[0]
    axis.plot(range(len(gpt_rows)), [row["bonito_mean_cka"] for row in gpt_rows], "o-", label="mean vs Bonito")
    axis.plot(range(len(gpt_rows)), [row["bonito_std_cka"] for row in gpt_rows], "s--", label="std vs Bonito")
    axis.plot(range(len(gpt_rows)), [row["bert_layer12_mean_cka"] for row in gpt_rows], "^-.", label="mean vs BERT layer 12")
    axis.set_xticks(range(len(labels)), labels)
    axis.set_ylim(0, 1)
    axis.set_xlabel("GPT representation")
    axis.set_ylabel("Linear CKA")
    axis.set_title("GPT layers: feature geometry alignment")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=9)

    axis = axes[1]
    gpt_cosines = [row["mean_adjacent_token_cosine"] for row in gpt_rows]
    bert_cosines = [row["mean_adjacent_token_cosine"] for row in bert_summary["layers"]]
    axis.plot(range(len(labels)), bert_cosines, "o-", color="#E07A32", label="V543 BERT")
    axis.plot(range(len(labels)), gpt_cosines, "^-", color="#2A9D62", label="GPT V615")
    bonito_mean = bonito_summary["mean_adjacent_position_cosine"]
    axis.axhline(bonito_mean, color="#3E5C76", linestyle="--", linewidth=2,
                 label=f"Bonito recurrent feature ({bonito_mean:.3f})")
    axis.axhspan(bonito_summary["p05_across_chunks"], bonito_summary["p95_across_chunks"],
                 color="#3E5C76", alpha=0.12, label="Bonito chunk p05–p95")
    axis.set_xticks(range(len(labels)), labels)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Representation layer")
    axis.set_ylabel("Mean adjacent-token cosine")
    axis.set_title("Local temporal smoothness")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(output_path, dpi=190)
    plt.close(figure)


def matrix_to_dict(names, matrix):
    return {
        left: {right: float(matrix[i, j]) for j, right in enumerate(names)}
        for i, left in enumerate(names)
    }


def main() -> None:
    args = parse_args()
    import numpy as np

    paired = np.load(args.paired_npz)
    bert = np.load(args.bert_layerwise_npz)
    gpt = np.load(args.gpt_layerwise_npz)
    if not (np.array_equal(paired["row_indices"], bert["row_indices"]) and np.array_equal(paired["row_indices"], gpt["row_indices"])):
        raise ValueError("Bonito/BERT/DLM/GPT representation rows are not exactly aligned")
    bert_names = bert["layer_names"].astype(str).tolist()
    gpt_names = gpt["layer_names"].astype(str).tolist()
    bert_layer12_index = bert_names.index("transformer_layer_12")
    gpt_final_index = gpt_names.index("final_normalized_hidden")
    base_representations = {
        "Bonito": paired["bonito_mean"],
        "BERT layer 12": bert["mean_features"][:, bert_layer12_index, :],
        "DLM ODE": paired["dlm_ode_hidden_mean"],
        "GPT V615": gpt["mean_features"][:, gpt_final_index, :],
    }
    base_std_representations = {
        "Bonito": paired["bonito_std"],
        "BERT layer 12": bert["std_features"][:, bert_layer12_index, :],
        "DLM ODE": paired["dlm_ode_hidden_std"],
        "GPT V615": gpt["std_features"][:, gpt_final_index, :],
    }
    gpt_rows = []
    for index, name in enumerate(gpt_names):
        mean = gpt["mean_features"][:, index, :]
        std = gpt["std_features"][:, index, :]
        gpt_rows.append({
            "layer_index": index,
            "layer_name": name,
            "bonito_mean_cka": linear_cka(mean, paired["bonito_mean"]),
            "bonito_std_cka": linear_cka(std, paired["bonito_std"]),
            "bert_layer12_mean_cka": linear_cka(mean, base_representations["BERT layer 12"]),
            "dlm_ode_mean_cka": linear_cka(mean, paired["dlm_ode_hidden_mean"]),
            "mean_adjacent_token_cosine": None,
        })
    gpt_summary = json.loads(args.gpt_layerwise_summary.read_text(encoding="utf-8"))
    for row, local_summary in zip(gpt_rows, gpt_summary["layers"]):
        row["mean_adjacent_token_cosine"] = local_summary["mean_adjacent_token_cosine"]
    bert_summary = json.loads(args.bert_layerwise_summary.read_text(encoding="utf-8"))
    bonito_summary = json.loads(args.bonito_temporal_summary.read_text(encoding="utf-8"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "gpt_layer_alignment.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gpt_rows[0]))
        writer.writeheader()
        writer.writerows(gpt_rows)
    cka, rsa, residuals, common_explained = plot_four_model_spaces(
        base_representations, paired["row_indices"], args.output_dir / "four_model_mean_representation_spaces.png"
    )
    plot_gpt_layers(gpt_rows, bert_summary, bonito_summary, args.output_dir / "gpt_layer_alignment.png")
    std_cka = np.empty_like(cka)
    std_rsa = np.empty_like(rsa)
    names = list(base_representations)
    for i, left in enumerate(names):
        for j, right in enumerate(names):
            std_cka[i, j] = linear_cka(base_std_representations[left], base_std_representations[right])
            std_rsa[i, j] = distance_rsa(base_std_representations[left], base_std_representations[right])
    result = {
        "sample_count": int(len(paired["row_indices"])),
        "row_alignment_verified": True,
        "gpt_model": gpt_summary,
        "bonito_temporal_smoothness": bonito_summary,
        "basecall_feature_definition": {
            "gpt": "V615 outputs.hidden_states[-1]; verified equal to outputs.last_hidden_state",
            "bert": "V543 outputs.hidden_states[-1], represented by transformer_layer_12",
            "dlm": "V011 ode_hidden_state with 1 step, start_t=0.97, CFG=0.5",
        },
        "four_model_mean_linear_cka": matrix_to_dict(names, cka),
        "four_model_std_linear_cka": matrix_to_dict(names, std_cka),
        "four_model_mean_distance_rsa": matrix_to_dict(names, rsa),
        "four_model_std_distance_rsa": matrix_to_dict(names, std_rsa),
        "procrustes_residual_to_bonito_mean": residuals,
        "common_pca_explained_variance_ratio": [float(value) for value in common_explained],
        "gpt_layers": gpt_rows,
        "best_gpt_layer_vs_bonito_mean": max(gpt_rows, key=lambda row: row["bonito_mean_cka"]),
        "best_gpt_layer_vs_bonito_std": max(gpt_rows, key=lambda row: row["bonito_std_cka"]),
        "gpt_training_logs": {
            "V615_active_DNA_run": summarize_log(args.gpt_v615_log),
            "V613_DNA_run": summarize_log(args.gpt_v613_log),
        },
        "caveat": (
            "The active GPT runs use DNA 150k chunks. Prior BERT/DLM logs use different model checkpoints, "
            "data sizes, and in the cited runs RNA data; their losses or validation scores are not direct controls."
        ),
    }
    (args.output_dir / "gpt_route_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "best_gpt_layer_vs_bonito_mean": result["best_gpt_layer_vs_bonito_mean"],
        "best_gpt_layer_vs_bonito_std": result["best_gpt_layer_vs_bonito_std"],
        "mean_cka": result["four_model_mean_linear_cka"],
        "mean_rsa": result["four_model_mean_distance_rsa"],
        "v613_completed_epochs": result["gpt_training_logs"]["V613_DNA_run"]["completed_epoch_count"],
        "v615_completed_epochs": result["gpt_training_logs"]["V615_active_DNA_run"]["completed_epoch_count"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
