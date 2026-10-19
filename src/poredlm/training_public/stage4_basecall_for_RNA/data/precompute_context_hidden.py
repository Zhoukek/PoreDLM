#!/usr/bin/env python3
"""Precompute frozen DLM context_encoder outputs for Stage 4 basecalling."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE4_ROOT = SCRIPT_DIR.parent
SRC_ROOT = STAGE4_ROOT.parents[2]
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(STAGE4_ROOT))

from poredlm.training_public.stage4_basecall_for_RNA.Basecalling.basecaller_v8_0420.data_multifolder import (  # noqa: E402
    MultiJsonlSignalRefDataset,
    MultiNpySignalRefDataset,
    create_collate_fn,
    scan_jsonl_files,
    scan_npy_pairs,
)
from poredlm.training_public.stage4_basecall_for_RNA.Basecalling.basecaller_v8_0420.model_dlm import BasecallModel  # noqa: E402
from poredlm.training_public.stage4_basecall_for_RNA.data.context_hidden_cache import (  # noqa: E402
    model_source_signature,
    token_sequence_key,
    valid_sequence_lengths,
)


def _split_paths(raw: str | None) -> list[str]:
    return [part.strip() for part in (raw or "").split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--jsonl_paths", help="Comma-separated .jsonl.gz files or folders.")
    data_group.add_argument("--npy_paths", help="Comma-separated folders or tokens/reference .npy files.")
    parser.add_argument("--output_dir", required=True, help="Directory for manifest, index, and fp16 features.")
    parser.add_argument("--model_name_or_path", required=True, help="The same DLM checkpoint used by basecall training.")
    parser.add_argument("--tokenizer_type", choices=["bwav"], default="bwav")
    parser.add_argument("--tokenizer_token_offset", type=int, default=128)
    parser.add_argument("--token_offset", type=int, default=0,
                        help="Offset added to raw signal token IDs before BwavTokenizer encoding.")
    parser.add_argument("--add_cls_token", action="store_true")
    parser.add_argument("--cls_token_id", type=int, default=5)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--backbone_chunk_size", type=int, default=1540)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing cache directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.token_offset < 0 or args.tokenizer_token_offset < 0:
        raise ValueError("Token offsets must be nonnegative.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"Requested device {args.device}, but CUDA is unavailable.")

    if args.jsonl_paths:
        files = scan_jsonl_files(_split_paths(args.jsonl_paths), group_by="file", recursive=args.recursive)
        dataset = MultiJsonlSignalRefDataset(files, token_offset=args.token_offset)
    else:
        pairs = scan_npy_pairs(_split_paths(args.npy_paths), group_by="file", recursive=args.recursive)
        dataset = MultiNpySignalRefDataset(pairs, token_offset=args.token_offset)
    if len(dataset) == 0:
        raise ValueError("No labeled records were found to cache.")

    model = BasecallModel(
        model_path=args.model_name_or_path,
        backbone_type="dlm",
        tokenizer_type=args.tokenizer_type,
        tokenizer_token_offset=args.tokenizer_token_offset,
        add_cls_token=args.add_cls_token,
        cls_token_id=args.cls_token_id,
        feature_source="context_hidden",
        freeze_backbone=True,
        head_type="ctc",
        pre_head_type="none",
        backbone_chunk_size=args.backbone_chunk_size,
    )
    model.to(torch.device(args.device))
    model.eval()
    hidden_size = int(model._infer_hidden_size())

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Cache output path is not a directory: {output_dir}")
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace it.")
        for name in (
            "manifest.json",
            "manifest.json.tmp",
            "index.sqlite3",
            "index.sqlite3-wal",
            "index.sqlite3-shm",
            "context_hidden.fp16",
        ):
            stale_file = output_dir / name
            if stale_file.exists():
                stale_file.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "context_hidden.fp16"
    index_path = output_dir / "index.sqlite3"

    connection = sqlite3.connect(index_path)
    connection.execute(
        "CREATE TABLE features (sequence_key TEXT PRIMARY KEY, token_offset INTEGER NOT NULL, token_length INTEGER NOT NULL)"
    )
    connection.commit()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
        collate_fn=create_collate_fn(model.tokenizer),
    )

    total_tokens = 0
    unique_sequences = 0
    device = torch.device(args.device)
    with features_path.open("wb") as feature_file:
        with torch.inference_mode():
            for step, batch in enumerate(loader, start=1):
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device, non_blocking=True)
                hidden = model._forward_backbone_hidden(input_ids, attention_mask)
                lengths = valid_sequence_lengths(input_ids, attention_mask)
                ids_cpu = input_ids.detach().to(device="cpu", dtype=torch.int32)

                for row, length in enumerate(lengths):
                    key = token_sequence_key(ids_cpu[row, :length])
                    if connection.execute(
                        "SELECT 1 FROM features WHERE sequence_key = ?", (key,)
                    ).fetchone() is not None:
                        continue
                    values = hidden[row, :length].detach().to(device="cpu", dtype=torch.float32).numpy()
                    values = np.asarray(values, dtype=np.float16, order="C")
                    byte_offset = total_tokens
                    feature_file.write(values.tobytes(order="C"))
                    connection.execute(
                        "INSERT INTO features(sequence_key, token_offset, token_length) VALUES (?, ?, ?)",
                        (key, byte_offset, length),
                    )
                    total_tokens += length
                    unique_sequences += 1

                connection.commit()
                if step % 100 == 0:
                    print(
                        f"[Cache] batches={step}/{len(loader)} unique_sequences={unique_sequences} "
                        f"cached_tokens={total_tokens}",
                        flush=True,
                    )
        feature_file.flush()
        os.fsync(feature_file.fileno())
    connection.close()

    manifest = {
        "format_version": 1,
        "model_name_or_path": str(Path(args.model_name_or_path).expanduser().resolve()),
        "model_signature": model_source_signature(args.model_name_or_path),
        "tokenizer_type": args.tokenizer_type,
        "tokenizer_token_offset": int(args.tokenizer_token_offset),
        "token_offset": int(args.token_offset),
        "add_cls_token": bool(args.add_cls_token),
        "cls_token_id": int(args.cls_token_id),
        "hidden_size": hidden_size,
        "storage_dtype": "float16",
        "total_tokens": total_tokens,
        "unique_sequences": unique_sequences,
        "source_records": len(dataset),
    }
    temp_manifest = output_dir / "manifest.json.tmp"
    with temp_manifest.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    temp_manifest.replace(output_dir / "manifest.json")
    print(f"[Cache] complete: {output_dir} ({unique_sequences} unique sequences, {total_tokens} tokens)")


if __name__ == "__main__":
    main()

