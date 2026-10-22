#!/usr/bin/env python3
"""Compare Bonito recurrent-feature cosine with reference-conditioned CRF moves."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[3]
SIGNAL_DIR = PROJECT_DIR / "data/signal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-npy", type=Path, default=SIGNAL_DIR / "eval00001_chunks.npy")
    parser.add_argument("--references-npy", type=Path, default=SIGNAL_DIR / "eval00001_references.npy")
    parser.add_argument("--reference-lengths-npy", type=Path, default=SIGNAL_DIR / "reference_lengths.npy")
    parser.add_argument(
        "--paired-npz", type=Path,
        default=PROJECT_DIR / "feature_analysis/paired_features_n128/paired_pooled_features.npz",
    )
    parser.add_argument(
        "--bonito-model-dir", type=Path,
        default=Path("/mnt/zzbnew/rnamodel/model/bonito/dna_basic_0121"),
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def viterbi_target_path(stay_scores, move_scores):
    """Viterbi path on Bonito's target-specific stay/move lattice, in NumPy."""
    import numpy as np

    time_steps, n_states = stay_scores.shape
    if move_scores.shape != (time_steps, n_states - 1):
        raise ValueError(
            f"Invalid target lattice: stay={stay_scores.shape}, move={move_scores.shape}"
        )

    neg_inf = -np.inf
    path_score = np.full(n_states, neg_inf, dtype=np.float64)
    path_score[0] = 0.0
    came_from_move = np.zeros((time_steps, n_states), dtype=np.bool_)
    for t in range(time_steps):
        stay = path_score + stay_scores[t]
        move = np.full(n_states, neg_inf, dtype=np.float64)
        move[1:] = path_score[:-1] + move_scores[t]
        take_move = move > stay
        came_from_move[t] = take_move
        path_score = np.maximum(stay, move)

    if not np.isfinite(path_score[-1]):
        raise ValueError("No valid CRF path reaches the last reference k-mer state")

    state = n_states - 1
    move_frames = np.zeros(time_steps, dtype=np.bool_)
    state_path = np.zeros(time_steps, dtype=np.int32)
    for t in range(time_steps - 1, -1, -1):
        state_path[t] = state
        if came_from_move[t, state]:
            move_frames[t] = True
            state -= 1
    if state != 0:
        raise RuntimeError(f"CRF alignment traceback ended at state {state}, expected 0")
    return move_frames, state_path


def rank_auc(labels, scores) -> float:
    """Mann-Whitney AUC; scores should be larger for boundary edges."""
    import numpy as np

    labels = np.asarray(labels, dtype=np.bool_)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if not positives or not negatives:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # Average ranks for tied scores.
    sorted_scores = scores[order]
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = float(ranks[labels].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def average_precision(labels, scores) -> float:
    import numpy as np

    labels = np.asarray(labels, dtype=np.bool_)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    if not positives:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ordered = labels[order].astype(np.float64)
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float((precision * ordered).sum() / positives)


def build_summary(arrays: dict, metadata: dict) -> dict:
    import numpy as np

    cosine = arrays["cosine"]
    boundary = arrays["boundary_edges"].astype(bool)
    valid = arrays["valid_edges"].astype(bool)
    vals_at_boundary = cosine[boundary & valid]
    vals_elsewhere = cosine[(~boundary) & valid]
    change_score = 1.0 - cosine

    offsets = np.arange(-8, 9)
    event_curves = []
    for row in range(len(cosine)):
        centers = np.flatnonzero(boundary[row] & valid[row])
        samples = []
        for center in centers:
            indices = center + offsets
            good = (indices >= 0) & (indices < cosine.shape[1])
            if good.all():
                samples.append(cosine[row, indices])
        if samples:
            event_curves.append(np.mean(samples, axis=0))
    event_curves = np.asarray(event_curves, dtype=np.float64)

    row_deltas = []
    for row in range(len(cosine)):
        row_mask = valid[row]
        row_boundary = boundary[row] & row_mask
        row_nonboundary = (~boundary[row]) & row_mask
        if row_boundary.any() and row_nonboundary.any():
            row_deltas.append(
                float(cosine[row, row_boundary].mean() - cosine[row, row_nonboundary].mean())
            )
    row_deltas = np.asarray(row_deltas, dtype=np.float64)
    rng = np.random.default_rng(20261020)
    bootstrap_means = np.asarray([
        rng.choice(row_deltas, size=len(row_deltas), replace=True).mean()
        for _ in range(2000)
    ])

    summary = {
        **metadata,
        "sample_count": int(len(cosine)),
        "time_positions_per_chunk": int(cosine.shape[1] + 1),
        "mean_adjacent_cosine_all_edges": float(cosine[valid].mean()),
        "reference_conditioned_boundary_rate": float(boundary[valid].mean()),
        "mean_cosine_at_forced_boundary_edges": float(vals_at_boundary.mean()),
        "mean_cosine_at_nonboundary_edges": float(vals_elsewhere.mean()),
        "boundary_minus_nonboundary_cosine": float(vals_at_boundary.mean() - vals_elsewhere.mean()),
        "mean_per_chunk_boundary_minus_nonboundary_cosine": float(row_deltas.mean()),
        "mean_per_chunk_boundary_minus_nonboundary_bootstrap_95pct_ci": [
            float(np.quantile(bootstrap_means, 0.025)),
            float(np.quantile(bootstrap_means, 0.975)),
        ],
        "auc_using_one_minus_cosine": rank_auc(boundary[valid], change_score[valid]),
        "average_precision_using_one_minus_cosine": average_precision(boundary[valid], change_score[valid]),
        "average_precision_baseline_boundary_rate": float(boundary[valid].mean()),
        "per_chunk_boundary_minus_nonboundary_p05": float(np.quantile(row_deltas, 0.05)),
        "per_chunk_boundary_minus_nonboundary_median": float(np.median(row_deltas)),
        "per_chunk_boundary_minus_nonboundary_p95": float(np.quantile(row_deltas, 0.95)),
        "event_offsets_in_encoder_frames": offsets.tolist(),
        "cosine_around_boundary_mean": event_curves.mean(axis=0).tolist(),
        "cosine_around_boundary_std_across_chunks": event_curves.std(axis=0).tolist(),
        "note": (
            "Boundary locations are Viterbi forced-alignment move frames conditioned on the reference sequence, "
            "not independently observed physical base transitions. The cosine feature is Bonito's recurrent "
            "encoder output immediately before its LinearCRF projection."
        ),
    }
    return summary


def write_plots(arrays: dict, summary: dict, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cosine = arrays["cosine"]
    boundary = arrays["boundary_edges"].astype(bool)
    valid = arrays["valid_edges"].astype(bool)
    rows = arrays["row_indices"]
    lengths = arrays["reference_lengths"]

    examples_path = output_dir / "bonito_cosine_boundary_timelines.png"
    example_positions = np.unique(np.linspace(0, len(rows) - 1, min(4, len(rows)), dtype=int))
    fig, axes = plt.subplots(len(example_positions), 1, figsize=(16, 10), sharex=True, squeeze=False)
    for ax, sample_pos in zip(axes[:, 0], example_positions):
        n_edges = int(valid[sample_pos].sum())
        x = np.arange(n_edges, dtype=np.float32) + 0.5
        ax.plot(x, cosine[sample_pos, :n_edges], linewidth=0.65, color="#276FBF", alpha=0.9)
        event_edges = np.flatnonzero(boundary[sample_pos, :n_edges])
        ax.vlines(event_edges + 0.5, 0.03, 0.12, transform=ax.get_xaxis_transform(),
                  color="#D1495B", linewidth=0.55, alpha=0.28)
        ax.set_ylabel("cosine")
        ax.set_ylim(-0.05, 1.02)
        ax.grid(alpha=0.18)
        ax.set_title(
            f"chunks_chunk_{int(rows[sample_pos])} | ref length={int(lengths[sample_pos])} | "
            f"forced moves={len(event_edges)}",
            loc="left", fontsize=10,
        )
    axes[-1, 0].set_xlabel("Bonito encoder time position (one step ≈ 5 signal samples)")
    fig.suptitle("Bonito adjacent-feature cosine and reference-conditioned CRF base moves", y=0.995)
    fig.tight_layout()
    fig.savefig(examples_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    zoom_path = output_dir / "bonito_cosine_boundary_zoom.png"
    sample_pos = 0
    zoom_end = min(240, int(valid[sample_pos].sum()))
    zoom_events = np.flatnonzero(boundary[sample_pos, :zoom_end])
    fig, ax = plt.subplots(figsize=(15, 4.8))
    ax.plot(np.arange(zoom_end) + 0.5, cosine[sample_pos, :zoom_end], color="#276FBF", linewidth=1.0)
    ax.vlines(zoom_events + 0.5, 0.02, 0.13, transform=ax.get_xaxis_transform(),
              color="#D1495B", linewidth=1.0, alpha=0.62, label="reference-aligned CRF move")
    for event_number in range(0, len(zoom_events), 10):
        edge = int(zoom_events[event_number])
        ax.annotate(
            f"move {event_number + 1}", (edge + 0.5, 0.15),
            xycoords=("data", "axes fraction"), xytext=(0, 5), textcoords="offset points",
            rotation=90, fontsize=7, color="#D1495B", ha="center", va="bottom",
        )
    ax.set_xlim(0, zoom_end)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Bonito encoder time position (one step ≈ 5 signal samples)")
    ax.set_ylabel("Adjacent-feature cosine")
    ax.set_title(
        f"Reference-aligned moves on the first {zoom_end} frames of chunks_chunk_{int(rows[sample_pos])}"
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(zoom_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    stats_path = output_dir / "bonito_boundary_cosine_summary.png"
    boundary_values = cosine[boundary & valid]
    other_values = cosine[(~boundary) & valid]
    offsets = np.asarray(summary["event_offsets_in_encoder_frames"])
    event_mean = np.asarray(summary["cosine_around_boundary_mean"])
    event_std = np.asarray(summary["cosine_around_boundary_std_across_chunks"])
    n_chunks = len(rows)
    event_sem = event_std / np.sqrt(max(1, n_chunks))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 4.8))
    ax0.boxplot([other_values, boundary_values], tick_labels=["Other edges", "Forced base moves"],
                showfliers=False, widths=0.55)
    ax0.set_ylabel("Adjacent Bonito feature cosine")
    ax0.set_title("Cosine at aligned moves vs other time edges")
    ax0.grid(axis="y", alpha=0.2)
    ax1.plot(offsets, event_mean, color="#276FBF", marker="o", markersize=3, linewidth=1.7)
    ax1.fill_between(offsets, event_mean - event_sem, event_mean + event_sem, color="#276FBF", alpha=0.18,
                     label="±1 SEM across chunks")
    ax1.axvline(0, color="#D1495B", linestyle="--", linewidth=1.2, label="forced move edge")
    ax1.set_xlabel("Cosine edge offset from forced move (encoder frames)")
    ax1.set_ylabel("Mean adjacent-feature cosine")
    ax1.set_title("Cosine profile around aligned base moves")
    ax1.grid(alpha=0.2)
    ax1.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(stats_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_only(output_dir: Path) -> None:
    import numpy as np

    arrays_npz = np.load(output_dir / "bonito_boundary_cosine.npz", allow_pickle=False)
    arrays = {name: arrays_npz[name] for name in arrays_npz.files}
    previous = json.loads((output_dir / "bonito_boundary_cosine_summary.json").read_text(encoding="utf-8"))
    arrays.pop("boundary_base_ids", None)
    arrays.pop("boundary_base_indices", None)
    np.savez_compressed(output_dir / "bonito_boundary_cosine.npz", **arrays)

    metadata_keys = (
        "model", "feature", "signal_file", "reference_file", "paired_rows_file", "device",
        "signal_samples_per_chunk", "bonito_stride", "cosine_definition", "alignment_method",
        "boundary_edge_mapping",
    )
    metadata = {key: previous[key] for key in metadata_keys}
    summary = build_summary(arrays, metadata)
    summary["mean_reference_length"] = float(arrays["reference_lengths"].mean())
    summary["mean_forced_move_count"] = float(arrays["forced_move_counts"].mean())
    (output_dir / "bonito_boundary_cosine_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "bonito_boundary_cosine_per_chunk.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "row_index", "sample_id", "reference_length", "forced_move_count",
            "mean_cosine", "mean_cosine_at_forced_boundary_edges", "mean_cosine_at_other_edges",
        ])
        for i, row in enumerate(arrays["row_indices"]):
            bmask = arrays["boundary_edges"][i] & arrays["valid_edges"][i]
            nmask = (~arrays["boundary_edges"][i]) & arrays["valid_edges"][i]
            writer.writerow([
                int(row), f"chunks_chunk_{int(row)}", int(arrays["reference_lengths"][i]),
                int(arrays["forced_move_counts"][i]), f"{arrays['cosine'][i].mean():.7f}",
                f"{arrays['cosine'][i, bmask].mean():.7f}", f"{arrays['cosine'][i, nmask].mean():.7f}",
            ])
    write_plots(arrays, summary, output_dir)


def main() -> None:
    args = parse_args()
    if args.max_samples <= 0 or args.batch_size <= 0 or args.threads <= 0:
        raise ValueError("--max-samples, --batch-size, and --threads must be positive")
    if args.plot_only:
        plot_only(args.output_dir)
        return

    import numpy as np
    import torch
    import torch.nn.functional as F

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    if not args.bonito_model_dir.is_dir():
        raise FileNotFoundError(f"Bonito model directory not found: {args.bonito_model_dir}")
    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(REPO_ROOT / "data"))
    from bonito.util import load_model

    paired = np.load(args.paired_npz, allow_pickle=False)
    row_indices = paired["row_indices"].astype(np.int64)
    n_samples = min(args.max_samples, len(row_indices))
    if n_samples != len(row_indices):
        selected = np.linspace(0, len(row_indices) - 1, n_samples, dtype=np.int64)
        row_indices = row_indices[selected]

    signals = np.load(args.signal_npy, mmap_mode="r", allow_pickle=False)
    references = np.load(args.references_npy, mmap_mode="r", allow_pickle=False)
    reference_lengths = np.load(args.reference_lengths_npy, mmap_mode="r", allow_pickle=False)
    if signals.ndim != 2 or references.ndim != 2 or reference_lengths.ndim != 1:
        raise ValueError("Expected signals/references as 2D arrays and lengths as a 1D array")
    if max(row_indices) >= len(signals) or max(row_indices) >= len(references):
        raise ValueError("Selected paired rows exceed signal/reference array bounds")

    device = torch.device(args.device)
    print(f"Loading Bonito {args.bonito_model_dir} on {device}; chunks={len(row_indices)}", flush=True)
    model = load_model(str(args.bonito_model_dir), str(device), half=False, use_koi=False)
    model.eval()
    stride = int(model.stride)
    cosines = []
    boundary_edges = []
    valid_edges = []
    state_paths = []
    observed_lengths = []
    move_counts = []
    alphabet = "NACGT"

    with torch.inference_mode():
        for start in range(0, len(row_indices), args.batch_size):
            batch_rows = row_indices[start:start + args.batch_size]
            batch = torch.as_tensor(
                np.asarray(signals[batch_rows], dtype=np.float32), device=device
            ).unsqueeze(1)
            _, feature_maps = model.encoder(batch, return_features=True)
            features = feature_maps[-2]
            scores = feature_maps[-1]
            if features.ndim != 3 or scores.ndim != 3:
                raise ValueError(f"Unexpected Bonito outputs: features={features.shape}, scores={scores.shape}")
            if features.shape[1] == len(batch_rows):
                features = features.transpose(0, 1)
                scores = scores.transpose(0, 1)
            elif features.shape[0] != len(batch_rows):
                raise ValueError(f"Cannot identify Bonito batch axis: {features.shape}")

            for local, row in enumerate(batch_rows):
                feature = features[local].float()
                score = scores[local:local + 1].transpose(0, 1).float()  # [T, 1, C]
                time_steps = int(feature.shape[0])
                cosine = F.cosine_similarity(feature[:-1], feature[1:], dim=-1).cpu().numpy()

                target_len = int(reference_lengths[row])
                target_np = np.asarray(references[row, :target_len], dtype=np.int64)
                if target_len <= model.seqdist.state_len:
                    raise ValueError(f"Reference too short at row {row}: {target_len}")
                if target_np.min() < 1 or target_np.max() >= len(alphabet):
                    raise ValueError(f"Unexpected base ids at row {row}: {np.unique(target_np)}")
                target = torch.as_tensor(target_np[None, :], dtype=torch.long, device=device)
                stay, move = model.seqdist.prepare_ctc_scores(score, target)
                forced_moves, state_path = viterbi_target_path(
                    stay[:, 0, :].detach().cpu().numpy(),
                    move[:, 0, :].detach().cpu().numpy(),
                )

                # A move into a new 5-mer state emits the next reference base.
                # Place it on the cosine edge immediately before that output frame.
                edge_events = np.zeros(time_steps - 1, dtype=np.bool_)
                for frame in np.flatnonzero(forced_moves):
                    edge = int(frame) - 1
                    if 0 <= edge < len(edge_events):
                        edge_events[edge] = True
                edge_valid = np.ones(time_steps - 1, dtype=np.bool_)
                cosines.append(cosine.astype(np.float32))
                boundary_edges.append(edge_events)
                valid_edges.append(edge_valid)
                state_paths.append(state_path.astype(np.int32))
                observed_lengths.append(target_len)
                move_counts.append(int(forced_moves.sum()))
                print(
                    f"Processed {min(start + local + 1, len(row_indices))}/{len(row_indices)} "
                    f"chunks (row={int(row)}, T={time_steps}, ref={target_len}, moves={int(forced_moves.sum())})",
                    flush=True,
                )

    # Bonito emits a fixed number of frames for these equally sized chunks.
    if len({len(x) for x in cosines}) != 1 or len({len(x) for x in state_paths}) != 1:
        raise ValueError("Expected equal Bonito output lengths for fixed-length signal chunks")
    cosine_array = np.stack(cosines)
    boundary_array = np.stack(boundary_edges)
    valid_array = np.stack(valid_edges)
    state_array = np.stack(state_paths)
    lengths_array = np.asarray(observed_lengths, dtype=np.int32)
    move_count_array = np.asarray(move_counts, dtype=np.int32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "bonito_boundary_cosine.npz",
        row_indices=row_indices,
        cosine=cosine_array,
        boundary_edges=boundary_array,
        valid_edges=valid_array,
        state_path=state_array,
        reference_lengths=lengths_array,
        forced_move_counts=move_count_array,
        stride=np.asarray(stride, dtype=np.int32),
    )

    metadata = {
        "model": str(args.bonito_model_dir),
        "feature": "feature_maps[-2], last recurrent encoder output immediately before LinearCRF projection",
        "signal_file": str(args.signal_npy),
        "reference_file": str(args.references_npy),
        "paired_rows_file": str(args.paired_npz),
        "device": str(device),
        "signal_samples_per_chunk": int(signals.shape[1]),
        "bonito_stride": stride,
        "cosine_definition": "cos(feature[t], feature[t+1]) for adjacent recurrent encoder positions",
        "alignment_method": "Viterbi max path through Bonito CRF stay/move scores constrained to the known reference sequence",
        "boundary_edge_mapping": "A forced move at encoder frame t is plotted on cosine edge t-1, between feature frames t-1 and t.",
    }
    summary = build_summary(
        {
            "cosine": cosine_array,
            "boundary_edges": boundary_array,
            "valid_edges": valid_array,
        },
        metadata,
    )
    summary["mean_reference_length"] = float(lengths_array.mean())
    summary["mean_forced_move_count"] = float(move_count_array.mean())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "bonito_boundary_cosine_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "bonito_boundary_cosine_per_chunk.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "row_index", "sample_id", "reference_length", "forced_move_count",
            "mean_cosine", "mean_cosine_at_forced_boundary_edges", "mean_cosine_at_other_edges",
        ])
        for i, row in enumerate(row_indices):
            bmask = boundary_array[i] & valid_array[i]
            nmask = (~boundary_array[i]) & valid_array[i]
            writer.writerow([
                int(row), f"chunks_chunk_{int(row)}", int(lengths_array[i]), int(move_count_array[i]),
                f"{cosine_array[i].mean():.7f}", f"{cosine_array[i, bmask].mean():.7f}",
                f"{cosine_array[i, nmask].mean():.7f}",
            ])
    write_plots(
        {
            "cosine": cosine_array,
            "boundary_edges": boundary_array,
            "valid_edges": valid_array,
            "row_indices": row_indices,
            "reference_lengths": lengths_array,
        },
        summary,
        args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
