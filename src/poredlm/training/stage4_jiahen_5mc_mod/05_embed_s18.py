#!/usr/bin/env python3
"""Extract comparable frozen embeddings for the S18 three-model benchmark."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROOT = Path("/mnt/zzbnew/rnamodel")
POREGPT_ROOT = Path("/mnt/zzbnew/poregpt")
PAD_ID = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("chr19", "chr16"), required=True)
    parser.add_argument("--model", choices=("v600", "v610", "v003"), required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--attention-token-budget",
        type=int,
        default=256 * 256 * 256,
        help="Approximate batch*sequence_length^2 budget used to avoid long-sequence OOM",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def make_inputs(sequences: list[list[int]], device, torch):
    max_len = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), max_len), PAD_ID, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for row_index, sequence in enumerate(sequences):
        values = torch.as_tensor(sequence, dtype=torch.long, device=device)
        input_ids[row_index, :len(sequence)] = values
        attention_mask[row_index, :len(sequence)] = 1
        if len(sequence) > 2:
            token_mask[row_index, 1:len(sequence) - 1] = True
    return input_ids, attention_mask, token_mask


def load_model(model_name: str, device, torch):
    sys.path.insert(0, str(ROOT / "liujiaheng/genome/analysis/s15"))
    from s15_common import patch_transformers, register_local_codec

    patch_transformers(torch)
    from transformers import AutoModel, AutoModelForCausalLM

    if model_name == "v003":
        model_dir = POREGPT_ROOT / "models/HF_VQE768C08A001_DNADLLM_V003/hf_dlm"
        elf = model_dir / "ELF-pytorch-port" / "src"
        if str(elf) not in sys.path:
            sys.path.insert(0, str(elf))
        model = AutoModel.from_pretrained(
            str(model_dir), trust_remote_code=True, local_files_only=True
        ).to(device).eval()
        return model
    root = {
        "v600": POREGPT_ROOT / "models/HF_RSQ742C12A511_DNAOLMO_V600",
        "v610": POREGPT_ROOT / "models/HF_RSQ741C12V523_MIXOLMO_V610",
    }[model_name]
    return AutoModelForCausalLM.from_pretrained(
        str(root / "base"), torch_dtype=torch.bfloat16,
        trust_remote_code=True, local_files_only=True,
    ).to(device).eval()


def encode(model_name: str, model, input_ids, attention_mask, token_mask, torch):
    if model_name == "v003":
        result = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_context=False,
            return_ode_hidden=True,
            return_sde_hidden=False,
            ode_steps=2,
            ode_start_t=0.98,
            ode_self_cond_cfg_scale=0.5,
        )
        hidden = result["ode_hidden_state"]
    else:
        result = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden = result.hidden_states[-1]
    if hidden.ndim != 3 or hidden.shape[-1] != 768:
        raise RuntimeError(f"Unexpected {model_name} hidden shape: {tuple(hidden.shape)}")
    weights = token_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


def main() -> int:
    args = parse_args()
    if args.device != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("S18 embedding requires physical GPU Device 0 via CUDA_VISIBLE_DEVICES=0")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device(args.device)
    table = pq.read_table(args.corpus, columns=[
        "window_id", "v003_stone_input_ids", "v600_apple_input_ids", "v610_apple_input_ids",
    ])
    data = table.to_pydict()
    total = table.num_rows if not args.limit else min(args.limit, table.num_rows)
    key = {
        "v003": "v003_stone_input_ids",
        "v600": "v600_apple_input_ids",
        "v610": "v610_apple_input_ids",
    }[args.model]
    sequences = [[int(value) for value in sequence] for sequence in data[key][:total]]
    if any(len(sequence) < 3 or sequence[0] != 2 or sequence[-1] != 3 for sequence in sequences):
        raise RuntimeError(f"{args.model} input IDs do not have valid BOS/EOS")

    out = Path(args.out_dir)
    embedding_dir = out / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.tag or ("smoke" if args.limit else "full")
    path = embedding_dir / f"{args.dataset}_{args.model}_{suffix}.npy"
    summary_path = embedding_dir / f"{args.dataset}_{args.model}_{suffix}_summary.json"
    if path.exists() and summary_path.exists():
        raise FileExistsError(f"Embedding output already exists: {path}")

    started = time.time()
    model = load_model(args.model, device, torch)
    output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(total, 768))
    # Sort by encoded length so padding is small. Long reads are processed in
    # smaller batches because attention memory grows quadratically with length.
    order = sorted(range(total), key=lambda index: len(sequences[index]))
    embedded = 0
    with torch.inference_mode():
        cursor = 0
        while cursor < total:
            tentative_end = min(cursor + args.batch_size, total)
            max_len = len(sequences[order[tentative_end - 1]])
            allowed = max(1, args.attention_token_budget // max(max_len * max_len, 1))
            end = min(tentative_end, cursor + allowed)
            batch_indices = order[cursor:end]
            batch_sequences = [sequences[index] for index in batch_indices]
            input_ids, attention_mask, token_mask = make_inputs(batch_sequences, device, torch)
            encoded = encode(args.model, model, input_ids, attention_mask, token_mask, torch)
            values = encoded.float().cpu().numpy()
            if values.shape != (end - cursor, 768) or not np.isfinite(values).all():
                raise RuntimeError(f"Invalid {args.model} embedding batch {cursor}:{end}")
            output[np.asarray(batch_indices, dtype=np.int64)] = values.astype(np.float16)
            cursor = end
            embedded = cursor
            if embedded == total or embedded % (args.batch_size * 10) == 0 or len(batch_sequences) == 1:
                output.flush()
                print(json.dumps({"dataset": args.dataset, "model": args.model, "embedded": embedded, "total": total, "batch_size": len(batch_sequences), "max_sequence_length": max_len}), flush=True)
    output.flush()
    del model, output
    gc.collect()
    torch.cuda.empty_cache()
    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "corpus": str(args.corpus),
        "records": total,
        "embedding_shape": [total, 768],
        "dtype": "float16",
        "device": "physical GPU 0 via CUDA_VISIBLE_DEVICES=0",
        "v003_hidden": "ode_hidden_state, ode_steps=2, ode_start_t=0.98, self_cond_cfg_scale=0.5" if args.model == "v003" else None,
        "v600_hidden": "last hidden state of base OLMo2; mean over non-special tokens" if args.model == "v600" else None,
        "v610_hidden": "last hidden state of base OLMo2; mean over non-special tokens" if args.model == "v610" else None,
        "seconds": time.time() - started,
        "output": str(path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
