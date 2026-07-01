#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np


def summarize(name: str, values: np.ndarray) -> str:
    if values.size == 0:
        return f"{name}: empty"
    return (
        f"{name}: len={values.size}, mean={values.mean():.6g}, "
        f"p95={np.quantile(values, 0.95):.6g}, max={values.max():.6g}"
    )


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def load_jsonl_row(path: Path, row_index: int) -> dict | None:
    with open_text(path) as handle:
        for current_index, line in enumerate(handle):
            if current_index == row_index:
                return json.loads(line)
    return None


def find_score_record(score_jsonl: Path, npz_path: Path) -> dict | None:
    target = str(npz_path)
    target_name = npz_path.name
    with open_text(score_jsonl) as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            score_array_path = item.get("score_array_path")
            if not score_array_path:
                continue
            score_path = Path(score_array_path)
            if score_array_path == target or score_path.name == target_name:
                return item
    return None


def infer_lengths_from_score_jsonl(score_jsonl: Path | None, npz_path: Path) -> dict[str, int | str] | None:
    if score_jsonl is None:
        return None
    score_record = find_score_record(score_jsonl, npz_path)
    if score_record is None:
        print(f"Warning: did not find {npz_path.name} in score jsonl: {score_jsonl}")
        return None

    info: dict[str, int | str] = {}
    for key in ("id", "valid_len", "seq_len"):
        if score_record.get(key) is not None:
            info[key] = score_record[key]

    source_path = score_record.get("source_path")
    row_index = score_record.get("row_index")
    if source_path is None or row_index is None:
        return info

    source_row = load_jsonl_row(Path(source_path), int(row_index))
    if source_row is None:
        print(f"Warning: could not read source row {row_index} from {source_path}")
        return info

    meta = source_row.get("meta") or {}
    for key in ("raw_signal_len", "processed_signal_len", "original_token_len"):
        if meta.get(key) is not None:
            info[key] = int(meta[key])
    return info


def format_length_info(length_info: dict[str, int | str] | None, signal_length: int | None) -> str:
    if length_info is None:
        length_info = {}
    pieces = []
    if length_info.get("id") is not None:
        pieces.append(f"read={length_info['id']}")
    if signal_length is not None:
        pieces.append(f"raw signal length={signal_length}")
    elif length_info.get("raw_signal_len") is not None:
        pieces.append(f"raw signal length={length_info['raw_signal_len']}")
    if length_info.get("processed_signal_len") is not None:
        pieces.append(f"processed signal length={length_info['processed_signal_len']}")
    if length_info.get("original_token_len") is not None:
        pieces.append(f"original token length={length_info['original_token_len']}")
    if length_info.get("valid_len") is not None:
        pieces.append(f"valid token length={length_info['valid_len']}")
    return " | ".join(pieces)


def plot_npz(
    npz_path: Path,
    output_path: Path | None,
    top_k: int,
    title: str | None,
    annotate_top_k: bool,
    length_info: dict[str, int | str] | None,
    signal_length: int | None,
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("This plotting script needs matplotlib. Install it in the current environment first.") from exc

    data = np.load(npz_path)
    if "l2" not in data:
        raise KeyError(f"{npz_path} does not contain array 'l2'. Available keys: {list(data.keys())}")

    l2 = np.asarray(data["l2"], dtype=np.float32)
    cosine_distance = np.asarray(data["cosine_distance"], dtype=np.float32) if "cosine_distance" in data else None
    x = np.arange(l2.size)

    if output_path is None:
        output_path = npz_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nrows = 2 if cosine_distance is not None else 1
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(14, 6 if nrows == 1 else 9), sharex=True)
    if nrows == 1:
        axes = [axes]

    fig_title = title or npz_path.name
    fig.suptitle(fig_title)
    length_text = format_length_info(length_info, signal_length)
    if length_text:
        fig.text(0.5, 0.935, length_text, ha="center", va="top", fontsize=9)
        print(length_text)

    axes[0].plot(x, l2, linewidth=0.9, color="#1f77b4")
    axes[0].set_ylabel("L2")
    axes[0].set_title(summarize("l2", l2))
    axes[0].grid(True, alpha=0.25)

    k = min(max(0, int(top_k)), l2.size)
    if k > 0:
        top_indices = np.argpartition(l2, -k)[-k:]
        top_indices = top_indices[np.argsort(l2[top_indices])[::-1]]
        axes[0].scatter(top_indices, l2[top_indices], s=24, color="#d62728", zorder=3, label=f"top {k}")
        if annotate_top_k:
            for rank, pos in enumerate(top_indices, start=1):
                axes[0].annotate(
                    f"{rank}:{int(pos)}",
                    xy=(int(pos), float(l2[pos])),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    rotation=45,
                    color="#7f1d1d",
                )
        axes[0].legend(loc="upper right")

        print("Top L2 positions:")
        print("rank\tposition\tl2\tcosine_distance")
        for rank, pos in enumerate(top_indices, start=1):
            cos_value = float(cosine_distance[pos]) if cosine_distance is not None and pos < cosine_distance.size else float("nan")
            print(f"{rank}\t{int(pos)}\t{float(l2[pos]):.8g}\t{cos_value:.8g}")

    if cosine_distance is not None:
        axes[1].plot(np.arange(cosine_distance.size), cosine_distance, linewidth=0.9, color="#2ca02c")
        axes[1].set_ylabel("Cosine distance")
        axes[1].set_title(summarize("cosine_distance", cosine_distance))
        axes[1].grid(True, alpha=0.25)

    axes[-1].set_xlabel("Token position")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot one embedding-shift score .npz file.")
    parser.add_argument("--npz", required=True, help="Path to one file under LB06_score_arrays/ or LB07_score_arrays/.")
    parser.add_argument("--output", default=None, help="Output png path. Default: same path with .png suffix.")
    parser.add_argument("--top-k", type=int, default=20, help="Highlight top-k L2 positions.")
    parser.add_argument("--no-annotate-top-k", action="store_true", help="Do not write top-k token positions on the plot.")
    parser.add_argument("--title", default=None, help="Optional figure title.")
    parser.add_argument("--signal-length", type=int, default=None, help="Manually provide raw signal length for this read.")
    parser.add_argument("--score-jsonl", default=None, help="Optional LB06/LB07 *_embedding_shift_scores.jsonl.gz used to infer raw signal length.")
    args = parser.parse_args()

    npz_path = Path(args.npz).expanduser()
    output_path = Path(args.output).expanduser() if args.output else None
    score_jsonl = Path(args.score_jsonl).expanduser() if args.score_jsonl else None
    length_info = infer_lengths_from_score_jsonl(score_jsonl, npz_path)
    saved_path = plot_npz(
        npz_path=npz_path,
        output_path=output_path,
        top_k=args.top_k,
        title=args.title,
        annotate_top_k=not args.no_annotate_top_k,
        length_info=length_info,
        signal_length=args.signal_length,
    )
    print(f"Saved plot to: {saved_path}")


if __name__ == "__main__":
    main()
