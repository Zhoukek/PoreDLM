#!/usr/bin/env python3
"""Compare BERT representation choices and temporal smoothing by adjacent cosine.

This is a representation-only diagnostic. It does not train a basecaller and
does not use the reference sequence. Bonito's existing recurrent-feature
measurement is preserved as a reference in the result table and plot.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens", type=Path,
        default=PROJECT_DIR / "data/tokens/eval00001_chunks.jsonl.gz",
    )
    parser.add_argument(
        "--paired-npz", type=Path,
        default=PROJECT_DIR / "feature_analysis/paired_features_n128/paired_pooled_features.npz",
    )
    parser.add_argument(
        "--model-dir", type=Path,
        default=Path("/mnt/zzbnew/poregpt/models/HF_RSQ741C12V523_DNABERT_V543/base"),
    )
    parser.add_argument(
        "--bonito-summary", type=Path,
        default=PROJECT_DIR / "explore/bonito_temporal_smoothness_n128.json",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--plot-only", action="store_true", help="Regenerate the plot from the saved JSON without loading the model")
    return parser.parse_args()


def load_selected_tokens(path: Path, selected_rows: list[int]) -> list[list[int]]:
    token_re = re.compile(r"<\|bwav:(\d+)\|>")
    selected = set(selected_rows)
    rows: list[int] = []
    token_rows: list[list[int]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index not in selected:
                continue
            record = json.loads(line)
            codes = [int(value) for value in token_re.findall(record["text"])]
            if len(codes) != 1500 or record.get("id") != f"chunks_chunk_{index}":
                raise ValueError(f"Unexpected token row at index {index}")
            if min(codes) < 0 or max(codes) > 2400:
                raise ValueError(f"Signal code outside [0, 2400] at row {index}")
            # Match V543 basecaller convention: [CLS, BOS, signal tokens, EOS].
            rows.append(index)
            token_rows.append([5, 2, *(code + 128 for code in codes), 3])
    if rows != selected_rows:
        raise ValueError("Token rows do not match paired-feature row indices")
    return token_rows


def adjacent_cosine(features):
    import torch.nn.functional as F

    return F.cosine_similarity(features[:, :-1, :], features[:, 1:, :], dim=-1).mean(dim=1)


def smooth(features, alpha: float, window: int = 3):
    """Blend features with a centered box-filtered version along time."""
    import torch
    import torch.nn.functional as F

    if window % 2 != 1:
        raise ValueError("Smoothing window must be odd")
    channels_first = features.transpose(1, 2)
    padding = window // 2
    # Replicate edge positions so filtering does not introduce artificial zeros.
    padded = F.pad(channels_first, (padding, padding), mode="replicate")
    kernel = torch.full(
        (features.shape[-1], 1, window), 1.0 / window,
        dtype=features.dtype, device=features.device,
    )
    average = F.conv1d(padded, kernel, groups=features.shape[-1]).transpose(1, 2)
    return features.lerp(average, alpha)


def layer_norm(features):
    import torch.nn.functional as F

    return F.layer_norm(features, (features.shape[-1],))


def main() -> None:
    args = parse_args()
    import numpy as np

    if args.plot_only:
        saved = json.loads((args.output_dir / "bert_cosine_method_results.json").read_text(encoding="utf-8"))
        make_plot(saved["methods"], saved["bonito_reference"]["mean_adjacent_position_cosine"], saved["bonito_reference"], args.output_dir / "bert_cosine_methods.png")
        make_layerwise_smoothing_plot(saved["methods"], saved["bonito_reference"], args.output_dir / "bert_all_layers_box3.png")
        return

    import torch
    from transformers import AutoModel

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    paired = np.load(args.paired_npz)
    selected_rows = paired["row_indices"].astype(np.int64).tolist()
    token_rows = load_selected_tokens(args.tokens, selected_rows)
    bonito = json.loads(args.bonito_summary.read_text(encoding="utf-8"))
    if not bonito.get("row_indices_match_paired_features"):
        raise ValueError("Bonito summary does not certify paired-row alignment")

    device = torch.device(args.device)
    print(f"Loading V543 BERT on {device}; paired chunks={len(selected_rows)}", flush=True)
    model = AutoModel.from_pretrained(
        str(args.model_dir), trust_remote_code=True, local_files_only=True,
    ).to(device).eval()

    # Keep only per-chunk scalar metrics; no large token activation archive is written.
    metric_names = [
        "embedding_output", *[f"transformer_layer_{i}" for i in range(1, 13)],
        "final_normalized_hidden",
        "embedding_output_box3",
        *[f"transformer_layer_{i}_box3" for i in range(1, 13)],
        "final_normalized_hidden_box3",
        "layer8_layer12_normed_mean",
        "layer8_layer10_layer12_normed_mean",
        "layer8_layer12_normed_mean_then_box3",
        "layer12_layernorm",
        "layer12_box3_alpha_0.25",
        "layer12_box3_alpha_0.50",
        "layer12_box3_alpha_0.75",
        "layer12_box3_alpha_1.00",
        "layer12_box5_alpha_0.50",
        "layer12_layernorm_box3_alpha_0.50",
        "layer9_box3_alpha_0.50",
    ]
    values_by_name: dict[str, list[float]] = {name: [] for name in metric_names}

    with torch.inference_mode():
        for row_index, token_ids in zip(selected_rows, token_rows):
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            output = model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
            states = [state[:, 2:-1, :].float() for state in output.hidden_states]
            final_norm = output.last_hidden_state[:, 2:-1, :].float()
            state_by_name = {
                "embedding_output": states[0],
                **{f"transformer_layer_{i}": states[i] for i in range(1, 13)},
                "final_normalized_hidden": final_norm,
            }

            # Per-token normalization prevents larger layer norms from dominating fusion.
            l8n, l10n, l12n = (layer_norm(state_by_name[f"transformer_layer_{i}"]) for i in (8, 10, 12))
            fused_8_12 = 0.5 * (l8n + l12n)
            fused_8_10_12 = (l8n + l10n + l12n) / 3.0
            l12 = state_by_name["transformer_layer_12"]
            candidates = {
                **state_by_name,
                **{f"{name}_box3": smooth(state, 1.0, 3) for name, state in state_by_name.items()},
                "layer8_layer12_normed_mean": fused_8_12,
                "layer8_layer10_layer12_normed_mean": fused_8_10_12,
                "layer8_layer12_normed_mean_then_box3": smooth(fused_8_12, 1.0, 3),
                "layer12_layernorm": layer_norm(l12),
                "layer12_box3_alpha_0.25": smooth(l12, 0.25, 3),
                "layer12_box3_alpha_0.50": smooth(l12, 0.50, 3),
                "layer12_box3_alpha_0.75": smooth(l12, 0.75, 3),
                "layer12_box3_alpha_1.00": smooth(l12, 1.00, 3),
                "layer12_box5_alpha_0.50": smooth(l12, 0.50, 5),
                "layer12_layernorm_box3_alpha_0.50": smooth(layer_norm(l12), 0.50, 3),
                "layer9_box3_alpha_0.50": smooth(state_by_name["transformer_layer_9"], 0.50, 3),
            }
            for name in metric_names:
                values_by_name[name].append(float(adjacent_cosine(candidates[name]).item()))
            print(f"Processed paired row {row_index}", flush=True)

    baseline = float(bonito["mean_adjacent_position_cosine"])
    result_rows = []
    for name, chunk_values in values_by_name.items():
        array = np.asarray(chunk_values, dtype=np.float64)
        result_rows.append({
            "method": name,
            "mean_adjacent_cosine": float(array.mean()),
            "std_across_chunks": float(array.std(ddof=0)),
            "p05_across_chunks": float(np.quantile(array, 0.05)),
            "p95_across_chunks": float(np.quantile(array, 0.95)),
            "delta_vs_bert_layer12": float(array.mean() - np.mean(values_by_name["transformer_layer_12"])),
            "gap_to_bonito_reference": float(baseline - array.mean()),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "bert_cosine_method_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    output = {
        "model_dir": str(args.model_dir),
        "sample_count": len(selected_rows),
        "row_indices_match_paired_features": True,
        "device": str(device),
        "signal_tokens_per_chunk": 1500,
        "special_tokens_removed_before_cosine": ["CLS", "BOS", "EOS"],
        "bonito_reference": bonito,
        "methods": result_rows,
        "warning": (
            "These transformations are selected/evaluated only by adjacent cosine, which they can increase by construction. "
            "This does not establish improved basecalling or boundary discrimination."
        ),
    }
    json_path = args.output_dir / "bert_cosine_method_results.json"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    make_plot(result_rows, baseline, bonito, args.output_dir / "bert_cosine_methods.png")
    make_layerwise_smoothing_plot(result_rows, bonito, args.output_dir / "bert_all_layers_box3.png")
    print(json.dumps({"bonito_reference": baseline, "top_methods": sorted(result_rows, key=lambda row: row["mean_adjacent_cosine"], reverse=True)[:8]}, ensure_ascii=False, indent=2))


def make_plot(rows, bonito_mean: float, bonito: dict, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    # Show a small set of useful candidates; full results remain in CSV/JSON.
    preferred = [
        "transformer_layer_8", "transformer_layer_9", "transformer_layer_10",
        "transformer_layer_11", "transformer_layer_12", "final_normalized_hidden",
        "layer8_layer12_normed_mean", "layer8_layer10_layer12_normed_mean",
        "layer8_layer12_normed_mean_then_box3", "layer12_layernorm",
        "layer12_box3_alpha_0.25", "layer12_box3_alpha_0.50",
        "layer12_box3_alpha_0.75", "layer12_box3_alpha_1.00",
        "layer12_box5_alpha_0.50", "layer12_layernorm_box3_alpha_0.50",
        "layer9_box3_alpha_0.50",
    ]
    by_name = {row["method"]: row for row in rows}
    shown = [by_name[name] for name in preferred]
    label_map = {
        "transformer_layer_8": "BERT L8",
        "transformer_layer_9": "BERT L9",
        "transformer_layer_10": "BERT L10",
        "transformer_layer_11": "BERT L11",
        "transformer_layer_12": "BERT L12",
        "final_normalized_hidden": "Final norm",
        "layer8_layer12_normed_mean": "Mix L8+L12",
        "layer8_layer10_layer12_normed_mean": "Mix L8+10+12",
        "layer8_layer12_normed_mean_then_box3": "Mix 8+12 + box3",
        "layer12_layernorm": "L12 norm",
        "layer12_box3_alpha_0.25": "L12 box3 α=.25",
        "layer12_box3_alpha_0.50": "L12 box3 α=.50",
        "layer12_box3_alpha_0.75": "L12 box3 α=.75",
        "layer12_box3_alpha_1.00": "L12 box3 α=1",
        "layer12_box5_alpha_0.50": "L12 box5 α=.50",
        "layer12_layernorm_box3_alpha_0.50": "L12 norm+box3",
        "layer9_box3_alpha_0.50": "L9 box3 α=.50",
    }
    labels = [label_map[row["method"]] for row in shown]
    means = np.asarray([row["mean_adjacent_cosine"] for row in shown])
    low = means - np.asarray([row["p05_across_chunks"] for row in shown])
    high = np.asarray([row["p95_across_chunks"] for row in shown]) - means
    colors = [
        "#777777" if row["method"].startswith("transformer") or row["method"] == "final_normalized_hidden"
        else "#D58A32" if row["method"].startswith("layer8_")
        else "#2878B5"
        for row in shown
    ]

    fig, axis = plt.subplots(figsize=(17, 7), constrained_layout=True)
    positions = np.arange(len(shown))
    axis.bar(positions, means, color=colors, alpha=0.88)
    axis.errorbar(positions, means, yerr=np.vstack([low, high]), fmt="none", ecolor="#333333", capsize=3, linewidth=1)
    axis.axhline(bonito_mean, color="#C44536", linestyle="--", linewidth=2.2, label=f"Bonito mean = {bonito_mean:.3f}")
    axis.axhspan(bonito["p05_across_chunks"], bonito["p95_across_chunks"], color="#C44536", alpha=0.10, label="Bonito chunk p05–p95")
    axis.set_xticks(positions, labels, rotation=35, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Mean adjacent-position cosine")
    axis.set_title("BERT layer and temporal-filter effects on adjacent cosine\nError bars show chunk p05–p95; Bonito retained as a reference")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def make_layerwise_smoothing_plot(rows, bonito: dict, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    by_name = {row["method"]: row for row in rows}
    layers = [("embedding_output", "Emb")]
    layers.extend((f"transformer_layer_{i}", f"L{i}") for i in range(1, 13))
    layers.append(("final_normalized_hidden", "Final norm"))
    raw = [by_name[name] for name, _ in layers]
    filtered = [by_name[f"{name}_box3"] for name, _ in layers]
    x = np.arange(len(layers))
    width = 0.38
    fig, axis = plt.subplots(figsize=(15, 7), constrained_layout=True)
    for positions, group, color, label in (
        (x - width / 2, raw, "#777777", "Raw layer"),
        (x + width / 2, filtered, "#2878B5", "3-point mean"),
    ):
        means = np.asarray([row["mean_adjacent_cosine"] for row in group])
        lower = means - np.asarray([row["p05_across_chunks"] for row in group])
        upper = np.asarray([row["p95_across_chunks"] for row in group]) - means
        axis.bar(positions, means, width=width, color=color, alpha=0.88, label=label)
        axis.errorbar(positions, means, yerr=np.vstack([lower, upper]), fmt="none", ecolor="#333333", capsize=2, linewidth=0.8)
    baseline = bonito["mean_adjacent_position_cosine"]
    axis.axhline(baseline, color="#C44536", linestyle="--", linewidth=2, label=f"Bonito mean = {baseline:.3f}")
    axis.axhspan(bonito["p05_across_chunks"], bonito["p95_across_chunks"], color="#C44536", alpha=0.10, label="Bonito chunk p05–p95")
    axis.set_xticks(x, [label for _, label in layers])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Mean adjacent-position cosine")
    axis.set_title("Effect of direct 3-point averaging at every BERT representation layer\nError bars show chunk p05–p95")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=2)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    main()
