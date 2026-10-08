#!/usr/bin/env python3
"""Tokenize selected S18 windows for V600 Apple, V610 Apple and V003 Stone."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


S15_DIR = Path(__file__).resolve().parents[1] / "s15"
if str(S15_DIR) not in sys.path:
    sys.path.insert(0, str(S15_DIR))
from s15_common import (  # noqa: E402
    BOS_ID,
    EOS_ID,
    POREGPT_ROOT,
    TOKEN_OFFSET,
    patch_transformers,
    register_local_codec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefix", default="s18_v600_v610_v003_token_corpus")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def wrapped_ids(tokens: np.ndarray) -> list[int]:
    return [BOS_ID, *(tokens.astype(np.int64) + TOKEN_OFFSET).tolist(), EOS_ID]


def main() -> int:
    args = parse_args()
    if args.device != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("S18 tokenization is restricted to physical GPU Device 0")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    import torch

    patch_transformers(torch)
    from transformers import AutoModel

    if not torch.cuda.is_available():
        raise RuntimeError("GPU Device 0 is unavailable")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    corpus_path = out / f"{args.prefix}.parquet"
    partial_path = out / f"{args.prefix}.parquet.partial"
    summary_path = out / f"{args.prefix}_summary.json"
    if corpus_path.exists() or summary_path.exists():
        raise FileExistsError(f"S18 token outputs already exist: {corpus_path}")

    manifest = pq.read_table(args.manifest).to_pylist()
    wanted = {str(row["window_id"]) for row in manifest}
    if len(wanted) != len(manifest):
        raise RuntimeError("Selection manifest contains duplicate window IDs")
    if not wanted:
        raise RuntimeError("Selection manifest is empty")

    device = torch.device(args.device)
    v600_dir = POREGPT_ROOT / "models/HF_RSQ742C12A511_DNAOLMO_V600/encoder"
    v610_dir = POREGPT_ROOT / "models/HF_RSQ741C12V523_MIXOLMO_V610/encoder"
    v003_dir = POREGPT_ROOT / "models/HF_VQE768C08A001_DNADLLM_V003/encoder"
    register_local_codec(v600_dir, "s18_v600_codec")
    register_local_codec(v610_dir, "s18_v610_codec")
    register_local_codec(v003_dir, "s18_v003_codec")
    v600_codec = AutoModel.from_pretrained(str(v600_dir), trust_remote_code=True, local_files_only=True).to(device).eval()
    v610_codec = AutoModel.from_pretrained(str(v610_dir), trust_remote_code=True, local_files_only=True).to(device).eval()
    v003_codec = AutoModel.from_pretrained(str(v003_dir), trust_remote_code=True, local_files_only=True).to(device).eval()

    writer: pq.ParquetWriter | None = None
    pending_by_length: dict[int, list[dict[str, object]]] = defaultdict(list)
    pending_output: list[dict[str, object]] = []
    seen: set[str] = set()
    token_lengths = {"v600": [], "v610": [], "v003": []}
    files = sorted(Path(args.candidate_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No candidate parquet shards found in {args.candidate_dir}")

    def write_pending() -> None:
        nonlocal writer
        if not pending_output:
            return
        table = pa.Table.from_pylist(pending_output)
        if writer is None:
            writer = pq.ParquetWriter(
                partial_path,
                table.schema,
                compression="zstd",
                compression_level=6,
                use_dictionary=True,
            )
        writer.write_table(table)
        pending_output.clear()

    def tokenize_rows(rows: list[dict[str, object]]) -> None:
        if not rows:
            return
        stone = torch.tensor(
            [row["signal_stone"] for row in rows], dtype=torch.float32, device=device
        ).unsqueeze(1)
        apple = torch.tensor(
            [row["signal_apple"] for row in rows], dtype=torch.float32, device=device
        ).unsqueeze(1)
        with torch.inference_mode():
            v600_raw = v600_codec.encode_signal(apple, layer=1).detach().cpu().tolist()
            v610_raw = v610_codec.encode_signal(apple, layer=1).detach().cpu().tolist()
            v003_raw = v003_codec.encode_signal(stone).detach().cpu().tolist()

        for row, v600_values, v610_values, v003_values in zip(rows, v600_raw, v610_raw, v003_raw):
            v600_tokens = np.asarray(v600_values, dtype=np.int32)
            v610_tokens = np.asarray(v610_values, dtype=np.int32)
            v003_tokens = np.asarray(v003_values, dtype=np.int32)
            if v600_tokens.size == 0 or v600_tokens.min() < 0 or v600_tokens.max() >= 2401:
                raise RuntimeError(f"Invalid V600 token IDs for {row['window_id']}")
            if v610_tokens.size == 0 or v610_tokens.min() < 0 or v610_tokens.max() >= 2401:
                raise RuntimeError(f"Invalid V610 token IDs for {row['window_id']}")
            if v003_tokens.size == 0 or v003_tokens.min() < 0 or v003_tokens.max() >= 65536:
                raise RuntimeError(f"Invalid V003 token IDs for {row['window_id']}")
            token_lengths["v600"].append(int(v600_tokens.size))
            token_lengths["v610"].append(int(v610_tokens.size))
            token_lengths["v003"].append(int(v003_tokens.size))
            pending_output.append({
                "window_id": str(row["window_id"]),
                "site_pos0": int(row["site_pos0"]),
                "label": int(row["label"]),
                "bed_percent": float(row["bed_percent"]),
                "bed_coverage": int(row["bed_coverage"]),
                "block_id": int(row["block_id"]),
                "ref_7mer": str(row["ref_7mer"]),
                "fast5_raw_read_id": str(row["fast5_raw_read_id"]),
                "source_part": str(row["source_part"]),
                "mapq": int(row["mapq"]),
                "signal_len": int(row["signal_len"]),
                "v600_apple_tokens_raw": v600_tokens.tolist(),
                "v600_apple_input_ids": wrapped_ids(v600_tokens),
                "v610_apple_tokens_raw": v610_tokens.tolist(),
                "v610_apple_input_ids": wrapped_ids(v610_tokens),
                "v003_stone_tokens_raw": v003_tokens.tolist(),
                "v003_stone_input_ids": wrapped_ids(v003_tokens),
            })
        if len(pending_output) >= 2048:
            write_pending()

    columns = [
        "window_id", "site_pos0", "label", "bed_coverage", "bed_percent", "block_id",
        "ref_7mer", "fast5_raw_read_id", "source_part", "mapq", "signal_len",
        "signal_apple", "signal_stone",
    ]
    started = time.time()
    for file_index, path in enumerate(files, 1):
        table = pq.ParquetFile(path).read(columns=columns)
        data = table.to_pydict()
        for index, value in enumerate(data["window_id"]):
            window_id = str(value)
            if window_id not in wanted or window_id in seen:
                continue
            seen.add(window_id)
            row = {name: data[name][index] for name in columns}
            pending_by_length[int(row["signal_len"])].append(row)
            queue = pending_by_length[int(row["signal_len"])]
            while len(queue) >= args.batch_size:
                batch = queue[:args.batch_size]
                pending_by_length[int(row["signal_len"])] = queue[args.batch_size:]
                queue = pending_by_length[int(row["signal_len"])]
                tokenize_rows(batch)
        if file_index % 500 == 0 or file_index == len(files):
            print(json.dumps({
                "stage": "tokenize_scan",
                "shards": file_index,
                "total_shards": len(files),
                "selected_seen": len(seen),
                "selected_total": len(wanted),
            }), flush=True)

    for length in sorted(pending_by_length):
        tokenize_rows(pending_by_length[length])
    write_pending()
    if writer is not None:
        writer.close()
    if seen != wanted:
        missing = len(wanted - seen)
        raise RuntimeError(f"Failed to recover {missing} selected windows from candidate shards")
    os.replace(partial_path, corpus_path)

    del v600_codec, v610_codec, v003_codec, writer
    gc.collect()
    torch.cuda.empty_cache()
    summary = {
        "records": len(seen),
        "seconds": time.time() - started,
        "device": "physical GPU 0 via CUDA_VISIBLE_DEVICES=0",
        "models": {
            "V600_Apple": "HF_RSQ742C12A511_DNAOLMO_V600 encoder; Apple-normalized signal; layer=1",
            "V610_Apple": "HF_RSQ741C12V523_MIXOLMO_V610 encoder; Apple-normalized signal; layer=1",
            "V003_Stone": "HF_VQE768C08A001_DNADLLM_V003 encoder; Stone-normalized signal; default layer",
        },
        "special_ids": {"bos": BOS_ID, "eos": EOS_ID, "token_offset": TOKEN_OFFSET},
        "raw_token_length_min_max": {
            name: [int(min(values)), int(max(values))] for name, values in token_lengths.items()
        },
        "apple_signal_source": "V600 Apple extractor; V600/V610 Apple preprocessor agreement checked during S18 extraction",
        "output": str(corpus_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
