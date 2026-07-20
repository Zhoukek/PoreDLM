#!/usr/bin/env python3
"""Tokenize Apple-normalized windows with the 64K VQE codec and embed with Stage-2 BERT."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

from tokenize_embed import patch_torch_transformers_compatibility


TOKEN_OFFSET = 128
CODEBOOK_SIZE = 65536
BOS_TOKEN_ID = 2
EOS_TOKEN_ID = 3
PAD_TOKEN_ID = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--codec-dir", required=True)
    parser.add_argument("--bert-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def register_local_codec(codec_dir: Path) -> None:
    modeling_path = codec_dir / "modeling_pore_vq_codec.py"
    spec = importlib.util.spec_from_file_location("local_modeling_pore_vq_codec", modeling_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load codec module from {modeling_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(out_dir / "hf_cache"))
    os.environ.setdefault("HF_MODULES_CACHE", str(out_dir / "hf_cache" / "modules"))

    import torch

    patch_torch_transformers_compatibility(torch)
    from transformers import AutoModel

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    records = [json.loads(line) for line in Path(args.windows).open() if line.strip()]
    if not records:
        raise RuntimeError("No windows were loaded")
    if not all(row["normal_mode"] == "apple" for row in records):
        raise RuntimeError("At least one signal window is not marked Apple-normalized")
    if not all(int(row["signal_base_shift"]) == -4 for row in records):
        raise RuntimeError("At least one signal window does not use shift=-4")

    register_local_codec(Path(args.codec_dir))
    print(json.dumps({"stage": "load_codec", "path": args.codec_dir}), flush=True)
    codec = AutoModel.from_pretrained(args.codec_dir, trust_remote_code=True)
    codec.to(device).eval()
    raw_tokens: list[np.ndarray] = []
    with torch.inference_mode():
        for index, row in enumerate(records, start=1):
            # Corpus values are already Apple-normalized. Do not call the feature extractor.
            signal = torch.as_tensor(row["signal"], dtype=torch.float32, device=device).view(1, 1, -1)
            token_ids = codec.encode_signal(signal).squeeze(0)
            tokens = token_ids.detach().cpu().numpy().astype(np.int64, copy=False)
            if tokens.size == 0 or tokens.min() < 0 or tokens.max() >= CODEBOOK_SIZE:
                raise ValueError(f"Invalid VQE token range for record {index}")
            raw_tokens.append(tokens)
    del codec
    if device.type == "cuda":
        torch.cuda.empty_cache()

    sequence_lengths = np.asarray([len(tokens) + 2 for tokens in raw_tokens], dtype=np.int32)
    max_length = int(sequence_lengths.max())
    input_ids = torch.full((len(records), max_length), PAD_TOKEN_ID, dtype=torch.long)
    attention_mask = torch.zeros((len(records), max_length), dtype=torch.long)
    signal_token_mask = torch.zeros((len(records), max_length), dtype=torch.bool)
    for index, tokens in enumerate(raw_tokens):
        shifted = torch.from_numpy(tokens + TOKEN_OFFSET)
        length = len(tokens)
        input_ids[index, 0] = BOS_TOKEN_ID
        input_ids[index, 1 : length + 1] = shifted
        input_ids[index, length + 1] = EOS_TOKEN_ID
        attention_mask[index, : length + 2] = 1
        signal_token_mask[index, 1 : length + 1] = True
    active_ids = input_ids[attention_mask.bool()]
    if active_ids.min() < 2 or active_ids.max() >= CODEBOOK_SIZE + TOKEN_OFFSET:
        raise ValueError("BERT input IDs are outside the configured vocabulary")

    print(
        json.dumps(
            {
                "stage": "load_bert", "path": args.bert_dir, "dtype": str(dtype),
                "records": len(records), "max_sequence_length": max_length,
            }
        ),
        flush=True,
    )
    bert = AutoModel.from_pretrained(args.bert_dir, trust_remote_code=True, torch_dtype=dtype)
    bert.to(device).eval()
    for parameter in bert.parameters():
        parameter.requires_grad_(False)

    pooled: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(records), args.batch_size):
            end = min(start + args.batch_size, len(records))
            ids = input_ids[start:end].to(device)
            mask = attention_mask[start:end].to(device)
            token_mask = signal_token_mask[start:end].to(device).unsqueeze(-1)
            hidden = bert.encode(input_ids=ids, attention_mask=mask).float()
            weights = token_mask.to(hidden.dtype)
            mean_hidden = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
            pooled.append(mean_hidden.cpu().numpy())
            print(json.dumps({"stage": "embed", "completed": end, "total": len(records)}), flush=True)
    embeddings = np.concatenate(pooled, axis=0).astype(np.float32, copy=False)

    token_path = out_dir / "signal_windows.vqe_tokens.jsonl"
    with token_path.open("w") as handle:
        for row, tokens in zip(records, raw_tokens):
            output = {key: value for key, value in row.items() if key != "signal"}
            output.update(
                {
                    "tokenizer": "VQE768C08A001 64K codec",
                    "raw_signal_tokens": tokens.tolist(),
                    "bert_input_ids": [BOS_TOKEN_ID, *(tokens + TOKEN_OFFSET).tolist(), EOS_TOKEN_ID],
                    "token_count": int(tokens.size),
                    "token_offset": TOKEN_OFFSET,
                }
            )
            handle.write(json.dumps(output, separators=(",", ":")) + "\n")

    embedding_path = out_dir / "signal_windows.vqe_bert_embeddings.npz"
    np.savez_compressed(
        embedding_path,
        embeddings=embeddings,
        labels=np.asarray([row["label"] for row in records], dtype=np.int8),
        datasets=np.asarray([row["dataset"] for row in records]),
        site_ids=np.asarray([row["site_id"] for row in records]),
        read_ids=np.asarray([row["read_id"] for row in records]),
        site_pos0=np.asarray([row["site_pos0"] for row in records], dtype=np.int64),
        strands=np.asarray([row["strand"] for row in records]),
        token_counts=np.asarray([len(tokens) for tokens in raw_tokens], dtype=np.int32),
        signal_lengths=np.asarray([row["signal_len"] for row in records], dtype=np.int32),
    )
    summary = {
        "codec_dir": args.codec_dir,
        "bert_dir": args.bert_dir,
        "records": len(records),
        "codebook_size": CODEBOOK_SIZE,
        "token_offset": TOKEN_OFFSET,
        "special_tokens": {"bos": BOS_TOKEN_ID, "eos": EOS_TOKEN_ID, "pad": PAD_TOKEN_ID},
        "token_count_min": int(min(len(tokens) for tokens in raw_tokens)),
        "token_count_max": int(max(len(tokens) for tokens in raw_tokens)),
        "token_count_mean": float(np.mean([len(tokens) for tokens in raw_tokens])),
        "raw_token_min": int(min(tokens.min() for tokens in raw_tokens)),
        "raw_token_max": int(max(tokens.max() for tokens in raw_tokens)),
        "embedding_source": "mean of final BERT hidden states over VQE signal-token positions; BOS/EOS excluded",
        "embedding_dim": int(embeddings.shape[1]),
        "apple_normalization_reapplied": False,
        "signal_base_shift": -4,
        "device": str(device),
        "dtype": str(dtype),
        "torch_version": torch.__version__,
    }
    (out_dir / "vqe_bert_token_embedding_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
