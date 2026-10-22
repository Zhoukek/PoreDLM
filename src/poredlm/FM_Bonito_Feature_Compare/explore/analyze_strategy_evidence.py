#!/usr/bin/env python3
"""Summarize layerwise feature geometry and comparable Stage-4 training runs."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from datetime import datetime
from pathlib import Path


def linear_cka(x, y) -> float:
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    k = x @ x.T
    l = y @ y.T
    denominator = float(((k * k).sum() * (l * l).sum()) ** 0.5)
    return float((k * l).sum() / denominator) if denominator else float("nan")


def parse_timestamp(line: str) -> datetime | None:
    match = re.match(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]", line)
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S") if match else None


def parse_latest_run(log_path: Path) -> dict | None:
    import numpy as np

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    arg_rows: list[tuple[int, dict]] = []
    for line_number, line in enumerate(lines):
        if "[Args]" not in line:
            continue
        try:
            args = ast.literal_eval(line.split("[Args]", 1)[1].strip())
        except (SyntaxError, ValueError):
            continue
        arg_rows.append((line_number, args))
    if not arg_rows:
        return None
    start, args = arg_rows[-1]
    segment = lines[start:]
    metric_re = re.compile(
        r"\[(Train|Val)\] epoch=(\d+) loss=([0-9.eE+-]+) acc=([0-9.eE+-]+) "
        r"coverage=([0-9.eE+-]+) blank=([0-9.eE+-]+) nonzero_len=([0-9.eE+-]+)"
    )
    metrics: dict[str, list[dict]] = {"Train": [], "Val": []}
    for line in segment:
        match = metric_re.search(line)
        if not match:
            continue
        kind = match.group(1)
        metrics[kind].append(
            {
                "epoch": int(match.group(2)),
                "loss": float(match.group(3)),
                "accuracy": float(match.group(4)),
                "coverage": float(match.group(5)),
                "blank": float(match.group(6)),
                "nonzero_length": float(match.group(7)),
                "selection_score": float(match.group(4)) * min(max(float(match.group(5)), 0.0), 1.0),
                "timestamp": parse_timestamp(line),
            }
        )
    val = metrics["Val"]
    if not val:
        return None
    best = max(val, key=lambda row: row["selection_score"])
    last = val[-1]
    val_times = [row["timestamp"] for row in val if row["timestamp"] is not None]
    epoch_seconds = [
        (right - left).total_seconds()
        for left, right in zip(val_times, val_times[1:])
        if 0 < (right - left).total_seconds() < 86400
    ]
    scheduler_match = None
    for line in segment:
        match = re.search(r"steps_per_epoch=(\d+) total_steps=(\d+) warmup_steps=(\d+)", line)
        if match:
            scheduler_match = tuple(int(value) for value in match.groups())
    train_by_epoch = {row["epoch"]: row for row in metrics["Train"]}
    best_train = train_by_epoch.get(best["epoch"])
    data_path = args.get("jsonl_paths") or args.get("npy_paths") or ""
    result = {
        "run_directory": str(log_path.parent),
        "latest_segment_start": lines[start][:21],
        "data_path": data_path,
        "data_name": Path(data_path).name if data_path else None,
        "model_name_or_path": args.get("model_name_or_path"),
        "backbone_type": args.get("backbone_type"),
        "feature_source": args.get("feature_source"),
        "hidden_layer": args.get("hidden_layer"),
        "learnable_fuse_last_n_layers": args.get("learnable_fuse_last_n_layers"),
        "unfreeze_context_last_n_layers": args.get("unfreeze_context_last_n_layers"),
        "unfreeze_elf_last_n_layers": args.get("unfreeze_elf_last_n_layers"),
        "unfreeze_last_n_layers": args.get("unfreeze_last_n_layers"),
        "head_type": args.get("head_type"),
        "pre_head_type": args.get("pre_head_type"),
        "batch_size_per_process": args.get("batch_size"),
        "head_lr": args.get("head_lr"),
        "backbone_lr": args.get("backbone_lr"),
        "warmup_ratio": args.get("warmup_ratio"),
        "num_epochs_planned": args.get("num_epochs"),
        "group_by": args.get("group_by"),
        "split_seed": args.get("split_seed"),
        "ode_steps": args.get("elf_ode_steps"),
        "ode_start_t": args.get("elf_ode_start_t"),
        "ode_self_cond_cfg_scale": args.get("elf_self_cond_cfg_scale"),
        "epochs_observed": len(val),
        "best_val": {key: value for key, value in best.items() if key != "timestamp"},
        "last_val": {key: value for key, value in last.items() if key != "timestamp"},
        "train_at_best_epoch": (
            {key: value for key, value in best_train.items() if key != "timestamp"} if best_train else None
        ),
        "median_epoch_seconds": float(np.median(epoch_seconds)) if epoch_seconds else None,
        "steps_per_epoch": scheduler_match[0] if scheduler_match else None,
        "total_steps": scheduler_match[1] if scheduler_match else None,
        "warmup_steps": scheduler_match[2] if scheduler_match else None,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layerwise-npz", type=Path, default=Path("bert_layerwise_n128/foundation_layerwise_pooled_features.npz")
    )
    parser.add_argument(
        "--paired-npz", type=Path, default=Path("../feature_analysis/paired_features_n128/paired_pooled_features.npz")
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("../../training_public/stage4_basecall_for_RNA/runs")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import numpy as np

    layerwise = np.load(args.layerwise_npz)
    paired = np.load(args.paired_npz)
    if not np.array_equal(layerwise["row_indices"], paired["row_indices"]):
        raise ValueError("Layerwise BERT and paired Bonito/DLM sample rows do not match")
    layer_names = layerwise["layer_names"].astype(str).tolist()
    layer_rows: list[dict] = []
    for index, name in enumerate(layer_names):
        mean_feature = layerwise["mean_features"][:, index, :]
        std_feature = layerwise["std_features"][:, index, :]
        layer_rows.append(
            {
                "layer_index": index,
                "layer_name": name,
                "bonito_mean_cka": linear_cka(mean_feature, paired["bonito_mean"]),
                "bonito_std_cka": linear_cka(std_feature, paired["bonito_std"]),
                "dlm_ode_mean_cka": linear_cka(mean_feature, paired["dlm_ode_hidden_mean"]),
                "dlm_ode_std_cka": linear_cka(std_feature, paired["dlm_ode_hidden_std"]),
                "temporal_mean_l2": float(np.linalg.norm(mean_feature, axis=1).mean()),
                "temporal_std_l2": float(np.linalg.norm(std_feature, axis=1).mean()),
            }
        )

    run_rows = []
    for log_path in sorted(args.runs_dir.glob("*/train.log")):
        parsed = parse_latest_run(log_path)
        if parsed is not None:
            run_rows.append(parsed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "bert_layer_strategy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(layer_rows[0]))
        writer.writeheader()
        writer.writerows(layer_rows)

    comparable_20k = [
        run
        for run in run_rows
        if run["data_name"] == "basecall_20000_chunks"
        and run["pre_head_type"] == "tcn"
        and run["head_type"] == "ctc"
    ]
    summary = {
        "sample_count": int(len(paired["row_indices"])),
        "layerwise_sample_alignment_verified": True,
        "ode_parameters": json.loads(
            (args.paired_npz.parent / "dlm_ode_feature_summary.json").read_text(encoding="utf-8")
        )["call_parameters"],
        "layers": layer_rows,
        "best_layer_by_bonito_mean_cka": max(layer_rows, key=lambda row: row["bonito_mean_cka"]),
        "best_layer_by_bonito_std_cka": max(layer_rows, key=lambda row: row["bonito_std_cka"]),
        "training_runs_latest_segments": run_rows,
        "comparable_20k_tcn_ctc_runs": comparable_20k,
        "caveat": "Layerwise CKA is a screening proxy after temporal pooling, not a basecalling probe or accuracy measurement.",
    }
    (args.output_dir / "strategy_evidence.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    x = np.arange(len(layer_rows))
    labels = ["emb"] + [str(i) for i in range(1, 13)] + ["final"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.3), constrained_layout=True)
    axes[0].plot(x, [row["bonito_mean_cka"] for row in layer_rows], "o-", label="mean vs Bonito")
    axes[0].plot(x, [row["bonito_std_cka"] for row in layer_rows], "s--", label="std vs Bonito")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("V543 BERT representation")
    axes[0].set_ylabel("Linear CKA")
    axes[0].set_title("Layerwise similarity to Bonito (128 chunks)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(x, [row["dlm_ode_mean_cka"] for row in layer_rows], "o-", label="mean vs DLM ODE")
    axes[1].plot(x, [row["dlm_ode_std_cka"] for row in layer_rows], "s--", label="std vs DLM ODE")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("V543 BERT representation")
    axes[1].set_ylabel("Linear CKA")
    axes[1].set_title("Layerwise similarity to DLM ODE (1 step, t=0.97)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.savefig(args.output_dir / "bert_layer_strategy.png", dpi=180)
    plt.close(fig)
    print(json.dumps({
        "best_mean": summary["best_layer_by_bonito_mean_cka"],
        "best_std": summary["best_layer_by_bonito_std_cka"],
        "run_count": len(run_rows),
        "comparable_20k_count": len(comparable_20k),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
