#!/usr/bin/env python3
"""Extract the DLM checkpoint's documented ODE-refined hidden features."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens-jsonl-gz", type=Path, default=Path("data/tokens/eval00001_chunks.jsonl.gz"))
    parser.add_argument(
        "--dlm-model-dir",
        type=Path,
        default=Path("/mnt/zzbnew/poregpt/models/HF_RSQ741C12V523_DNADLLM_V011/hf_dlm_elf_mixed_with_cls_token_step65000-unsharded"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("feature_analysis/paired_features_n128"))
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ode-steps", type=int, default=1)
    parser.add_argument("--ode-start-t", type=float, default=0.97)
    parser.add_argument("--ode-self-cond-cfg-scale", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_dir = args.dlm_model_dir
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"DLM config not found: {model_dir}")
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    elf_src = model_dir / "ELF-pytorch-port" / "src"
    if elf_src.is_dir():
        sys.path.insert(0, str(elf_src))
    else:
        configured_src = config.get("elf_src_path")
        if configured_src and Path(configured_src).is_dir():
            sys.path.insert(0, configured_src)

    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; refusing a silent CPU fallback.")
    device = torch.device(args.device)
    paired_path = args.output_dir / "paired_pooled_features.npz"
    paired = np.load(paired_path)
    row_indices = paired["row_indices"].astype(np.int64)
    if len(row_indices) > args.max_samples:
        row_indices = row_indices[: args.max_samples]
    row_set = set(int(i) for i in row_indices)
    token_re = re.compile(r"<\|bwav:(\d+)\|>")
    all_rows: dict[int, list[int]] = {}
    with gzip.open(args.tokens_jsonl_gz, "rt", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx not in row_set:
                continue
            record = json.loads(line)
            code_matches = token_re.findall(record["text"])
            if len(code_matches) != 1500:
                raise ValueError(f"Expected 1,500 signal codes at row {idx}; got {len(code_matches)}")
            if record.get("id") != f"chunks_chunk_{idx}":
                raise ValueError(f"Unexpected row ID at {idx}: {record.get('id')}")
            all_rows[idx] = [int(code) for code in code_matches]
    if set(all_rows) != row_set:
        raise ValueError(f"Found {len(all_rows)} token rows, expected {len(row_set)}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
    vocab = tokenizer.get_vocab()
    token_rows: list[list[int]] = []
    for idx in row_indices:
        codes = all_rows[int(idx)]
        row = [5, 2]
        for code in codes:
            token = f"<|bwav:{code}|>"
            token_id = vocab.get(token, tokenizer.unk_token_id)
            if token_id == tokenizer.unk_token_id:
                raise ValueError(f"DLM tokenizer has no token {token}")
            row.append(int(token_id))
        row.append(int(tokenizer.eos_token_id))
        token_rows.append(row)
    expected_ids = [[5, 2, *(code + 128 for code in all_rows[int(idx)]), 3] for idx in row_indices]
    if token_rows != expected_ids:
        raise ValueError("DLM tokenizer mapping differs from the paired V543 token sequence")

    print(f"Loading DLM from {model_dir}", flush=True)
    model = AutoModel.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True, torch_dtype="auto"
    ).to(device).eval()
    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(row_indices), args.batch_size):
            end = min(start + args.batch_size, len(row_indices))
            ids = torch.tensor(token_rows[start:end], dtype=torch.long, device=device)
            mask = torch.ones_like(ids)
            outputs = model(
                input_ids=ids,
                attention_mask=mask,
                return_ode_hidden=True,
                ode_steps=args.ode_steps,
                ode_start_t=args.ode_start_t,
                ode_self_cond_cfg_scale=args.ode_self_cond_cfg_scale,
            )
            hidden = outputs["ode_hidden_state"][:, 2:-1, :].float()
            means.append(hidden.mean(dim=1).cpu().numpy())
            stds.append(hidden.var(dim=1, unbiased=False).clamp_min(0).sqrt().cpu().numpy())
            print(f"Processed {end}/{len(row_indices)} chunks", flush=True)

    ode_mean = np.concatenate(means).astype(np.float32)
    ode_std = np.concatenate(stds).astype(np.float32)
    arrays = {key: paired[key] for key in paired.files}
    arrays["dlm_ode_hidden_mean"] = ode_mean
    arrays["dlm_ode_hidden_std"] = ode_std
    np.savez_compressed(paired_path, **arrays)
    metadata = {
        "model_dir": str(model_dir),
        "derived_from": config.get("dlm_config", {}).get("context_encoder_path"),
        "sample_count": len(row_indices),
        "row_indices_match_existing_paired_features": bool(np.array_equal(row_indices, paired["row_indices"])),
        "token_mapping": "DLM tokenizer.json: <|bwav:n|> -> n+128; explicit CLS=5, BOS=2, EOS=3; verified against paired V543 inputs",
        "output": "ode_hidden_state (deterministic ODE refinement of context_encoder hidden state)",
        "call_parameters": {
            "ode_steps": args.ode_steps,
            "ode_start_t": args.ode_start_t,
            "ode_self_cond_cfg_scale": args.ode_self_cond_cfg_scale,
        },
        "feature_pooling": "mean and population standard deviation over the 1,500 signal-token positions only",
        "hidden_size": int(ode_mean.shape[-1]),
        "device": str(device),
    }
    out_path = args.output_dir / "dlm_ode_feature_summary.json"
    out_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
