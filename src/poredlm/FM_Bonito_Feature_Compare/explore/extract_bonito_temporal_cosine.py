#!/usr/bin/env python3
"""Measure adjacent-position cosine in Bonito's pre-CRF recurrent features."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SCRIPT_DIR.parent / "feature_analysis"
REPO_ROOT = SCRIPT_DIR.parents[4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-npy", type=Path, default=SCRIPT_DIR.parent / "data/signal/eval00001_chunks.npy")
    parser.add_argument(
        "--paired-npz", type=Path,
        default=ANALYSIS_DIR / "paired_features_n128/paired_pooled_features.npz",
    )
    parser.add_argument(
        "--bonito-model-dir", type=Path,
        default=Path("/mnt/zzbnew/rnamodel/model/bonito/dna_basic_0121"),
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import numpy as np
    import torch

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing an accidental CPU run")
    if not args.bonito_model_dir.is_dir():
        raise FileNotFoundError(f"Bonito model directory does not exist: {args.bonito_model_dir}")
    if not (args.bonito_model_dir / "config.toml").is_file():
        raise FileNotFoundError(f"Bonito config.toml not found: {args.bonito_model_dir}")

    bonito_src = REPO_ROOT / "data" / "bonito"
    sys.path.insert(0, str(bonito_src.parent))
    from bonito.util import load_model

    torch.set_num_threads(args.threads)
    paired = np.load(args.paired_npz)
    rows = paired["row_indices"].astype(np.int64)
    signals = np.load(args.signal_npy, mmap_mode="r", allow_pickle=False)
    if signals.ndim != 2 or (len(rows) and int(rows.max()) >= len(signals)):
        raise ValueError(f"Signal array {signals.shape} cannot serve paired rows")

    device = torch.device(args.device)
    model = load_model(str(args.bonito_model_dir), str(device), half=False, use_koi=False)
    model.eval()
    cosine_per_chunk: list[float] = []
    n_time_positions: list[int] = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start:start + args.batch_size]
            signal_batch = torch.as_tensor(
                np.asarray(signals[batch_rows], dtype=np.float32),
                device=device,
            ).unsqueeze(1)
            _, feature_maps = model.encoder(signal_batch, return_features=True)
            features = feature_maps[-2]
            if features.ndim != 3:
                raise ValueError(f"Expected [T,B,C] or [B,T,C], got {tuple(features.shape)}")
            if features.shape[1] == len(batch_rows):
                # Bonito CRF encoder returns [T, B, C].
                features = features.transpose(0, 1)
            elif features.shape[0] != len(batch_rows):
                raise ValueError(f"Cannot identify batch dimension: {tuple(features.shape)}")
            features = features.float()
            left, right = features[:, :-1, :], features[:, 1:, :]
            cosine = torch.nn.functional.cosine_similarity(left, right, dim=-1)
            cosine_per_chunk.extend(cosine.mean(dim=1).cpu().tolist())
            n_time_positions.extend([int(features.shape[1])] * len(batch_rows))
            print(f"Processed {min(start + args.batch_size, len(rows))}/{len(rows)} chunks", flush=True)

    values = np.asarray(cosine_per_chunk, dtype=np.float64)
    if len(values) != len(rows):
        raise ValueError(
            f"Bonito cosine rows do not match selected samples: got {len(values)}, expected {len(rows)}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "bonito_temporal_smoothness_n128.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_index", "sample_id", "time_positions", "mean_adjacent_cosine"])
        for row, time_count, value in zip(rows, n_time_positions, values):
            writer.writerow([int(row), f"chunks_chunk_{int(row)}", time_count, f"{value:.9f}"])
    summary = {
        "model": "Bonito dna_basic_0121",
        "feature": "last recurrent encoder output immediately before LinearCRF projection (feature_maps[-2])",
        "sample_count": int(len(rows)),
        "row_indices_match_paired_features": True,
        "device": str(device),
        "input_signal_samples": int(signals.shape[1]),
        "output_time_positions": {
            "min": int(min(n_time_positions)),
            "max": int(max(n_time_positions)),
        },
        "mean_adjacent_position_cosine": float(values.mean()),
        "std_across_chunks": float(values.std(ddof=0)),
        "median_across_chunks": float(np.median(values)),
        "p05_across_chunks": float(np.quantile(values, 0.05)),
        "p95_across_chunks": float(np.quantile(values, 0.95)),
        "per_chunk_csv": csv_path.name,
        "definition": "For each chunk, cosine(feature[t], feature[t+1]) is averaged over adjacent Bonito encoder time positions; summary is the mean over paired chunks.",
        "resolution_note": "Bonito's recurrent feature has a different temporal stride from the Transformer signal-token stream; the cosine compares adjacent positions in each model's own native time grid.",
    }
    json_path = args.output_dir / "bonito_temporal_smoothness_n128.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
