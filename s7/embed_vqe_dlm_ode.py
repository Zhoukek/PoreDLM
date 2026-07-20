#!/usr/bin/env python3
"""Embed existing 64K VQE signal tokens with the Stage-3 DLM ODE hidden state."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from tokenize_embed import patch_torch_transformers_compatibility


BOS_TOKEN_ID = 2
EOS_TOKEN_ID = 3
PAD_TOKEN_ID = 1
TOKEN_OFFSET = 128
CODEBOOK_SIZE = 65536


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/s7/vqe_bert/signal_windows.vqe_tokens.jsonl")
    parser.add_argument("--model-dir", default="/mnt/zzbnew/poregpt/models/HF_VQE768C08A001_DNADLLM_V001/hf_dlm")
    parser.add_argument("--out-dir", default="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/s7/vqe_dlm_zhou")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=8)
    # Defaults intentionally match hf_dlm/readme.md.
    parser.add_argument("--ode-steps", type=int, default=4)
    parser.add_argument("--ode-start-t", type=float, default=0.95)
    parser.add_argument("--ode-self-cond-cfg-scale", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(out_dir / "hf_cache"))
    os.environ.setdefault("HF_MODULES_CACHE", str(out_dir / "hf_cache" / "modules"))

    model_dir = Path(args.model_dir)
    elf_src = model_dir / "ELF-pytorch-port" / "src"
    if not elf_src.is_dir():
        raise RuntimeError(f"ELF source directory is missing: {elf_src}")
    if str(elf_src) not in sys.path:
        sys.path.insert(0, str(elf_src))

    import torch

    patch_torch_transformers_compatibility(torch)
    from transformers import AutoModel

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    rows = [json.loads(line) for line in Path(args.tokens).open() if line.strip()]
    if not rows:
        raise RuntimeError("No VQE token rows were loaded")

    for index, row in enumerate(rows, start=1):
        raw_tokens = np.asarray(row["raw_signal_tokens"], dtype=np.int64)
        expected_ids = [BOS_TOKEN_ID, *(raw_tokens + TOKEN_OFFSET).tolist(), EOS_TOKEN_ID]
        if raw_tokens.size == 0 or raw_tokens.min() < 0 or raw_tokens.max() >= CODEBOOK_SIZE:
            raise ValueError(f"Invalid 64K VQE token range in row {index}")
        if row.get("bert_input_ids") != expected_ids:
            raise ValueError(f"Unexpected shifted/special-token sequence in row {index}")
        if int(row.get("token_count", -1)) != raw_tokens.size:
            raise ValueError(f"Token-count mismatch in row {index}")

    sequence_lengths = np.asarray([len(row["bert_input_ids"]) for row in rows], dtype=np.int32)
    max_length = int(sequence_lengths.max())
    input_ids = torch.full((len(rows), max_length), PAD_TOKEN_ID, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), max_length), dtype=torch.long)
    signal_token_mask = torch.zeros((len(rows), max_length), dtype=torch.bool)
    for index, row in enumerate(rows):
        ids = torch.as_tensor(row["bert_input_ids"], dtype=torch.long)
        length = len(ids)
        input_ids[index, :length] = ids
        attention_mask[index, :length] = 1
        signal_token_mask[index, 1 : length - 1] = True

    print(
        json.dumps(
            {
                "stage": "load_dlm",
                "path": str(model_dir),
                "records": len(rows),
                "max_sequence_length": max_length,
                "ode_steps": args.ode_steps,
                "ode_start_t": args.ode_start_t,
                "ode_self_cond_cfg_scale": args.ode_self_cond_cfg_scale,
            }
        ),
        flush=True,
    )
    # Do not override torch_dtype: the model README loads the checkpoint at its
    # configured dtype and then moves it to the selected device.
    model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    pooled: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            end = min(start + args.batch_size, len(rows))
            ids = input_ids[start:end].to(device)
            mask = attention_mask[start:end].to(device)
            output = model(
                input_ids=ids,
                attention_mask=mask,
                return_context=True,
                return_ode_hidden=True,
                ode_steps=args.ode_steps,
                ode_start_t=args.ode_start_t,
                ode_self_cond_cfg_scale=args.ode_self_cond_cfg_scale,
            )
            ode_hidden = output["ode_hidden_state"].float()
            weights = signal_token_mask[start:end].to(device).unsqueeze(-1).to(ode_hidden.dtype)
            mean_hidden = (ode_hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
            pooled.append(mean_hidden.cpu().numpy())
            print(json.dumps({"stage": "embed", "completed": end, "total": len(rows)}), flush=True)

    embeddings = np.concatenate(pooled, axis=0).astype(np.float32, copy=False)
    if not np.isfinite(embeddings).all():
        raise RuntimeError("ODE embeddings contain non-finite values")

    embedding_path = out_dir / "signal_windows.vqe_dlm_ode_embeddings.npz"
    np.savez_compressed(
        embedding_path,
        embeddings=embeddings,
        labels=np.asarray([row["label"] for row in rows], dtype=np.int8),
        datasets=np.asarray([row["dataset"] for row in rows]),
        site_ids=np.asarray([row["site_id"] for row in rows]),
        read_ids=np.asarray([row["read_id"] for row in rows]),
        site_pos0=np.asarray([row["site_pos0"] for row in rows], dtype=np.int64),
        strands=np.asarray([row["strand"] for row in rows]),
        token_counts=np.asarray([row["token_count"] for row in rows], dtype=np.int32),
        signal_lengths=np.asarray([row["signal_len"] for row in rows], dtype=np.int32),
    )

    parameter_dtype = str(next(model.parameters()).dtype)
    summary = {
        "model_dir": str(model_dir),
        "records": len(rows),
        "input_tokens": "64K VQE tokens with +128 offset, BOS=2, EOS=3, PAD=1",
        "ode_steps": args.ode_steps,
        "ode_start_t": args.ode_start_t,
        "ode_self_cond_cfg_scale": args.ode_self_cond_cfg_scale,
        "embedding_source": (
            "mean of ode_hidden_state over VQE signal-token positions; BOS/EOS/PAD excluded"
        ),
        "embedding_dim": int(embeddings.shape[1]),
        "token_count_min": int(min(row["token_count"] for row in rows)),
        "token_count_max": int(max(row["token_count"] for row in rows)),
        "token_count_mean": float(np.mean([row["token_count"] for row in rows])),
        "device": str(device),
        "parameter_dtype": parameter_dtype,
        "output_dtype": str(embeddings.dtype),
        "torch_version": torch.__version__,
        "embedding_path": str(embedding_path),
    }
    (out_dir / "vqe_dlm_ode_embedding_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
