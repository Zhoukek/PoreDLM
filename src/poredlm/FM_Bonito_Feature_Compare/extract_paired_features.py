#!/usr/bin/env python3
"""Extract paired Bonito and foundation-model features on the same chunks.

Run from a project environment with numpy, torch, transformers, and the Bonito
runtime dependencies installed. Outputs per-chunk pooled features (not full
activation tensors) and linear CKA between sample-level representations.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-npy", type=Path, default=Path("data/signal/eval00001_chunks.npy"))
    parser.add_argument("--tokens-jsonl-gz", type=Path, default=Path("data/tokens/eval00001_chunks.jsonl.gz"))
    parser.add_argument("--bonito-model-dir", type=Path, default=Path("/mnt/zzbnew/rnamodel/model/bonito/dna_basic_0121"))
    parser.add_argument("--foundation-base-dir", type=Path, default=Path("/mnt/zzbnew/poregpt/models/HF_RSQ741C12V523_DNABERT_V543/base"))
    parser.add_argument("--output-dir", type=Path, default=Path("feature_analysis/paired_features"))
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def load_token_rows(path: Path, selected_indices: set[int]) -> tuple[list[int], list[list[int]]]:
    """Read selected rows in file order; map codec ids to BERT ids + specials."""
    import re

    token_re = re.compile(r"<\|bwav:(\d+)\|>")
    row_indices: list[int] = []
    token_rows: list[list[int]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index not in selected_indices:
                continue
            record = json.loads(line)
            codes = [int(value) for value in token_re.findall(record["text"])]
            if len(codes) != 1500:
                raise ValueError(f"Expected 1,500 signal codes at row {row_index}, got {len(codes)}")
            if min(codes) < 0 or max(codes) > 2400:
                raise ValueError(f"Signal code outside [0, 2400] at row {row_index}")
            if record.get("id") != f"chunks_chunk_{row_index}":
                raise ValueError(f"Unexpected ID/order at row {row_index}: {record.get('id')}")
            # Match the project's base model convention: [CLS=5, BOS=2], code+128, [EOS=3].
            token_rows.append([5, 2, *(code + 128 for code in codes), 3])
            row_indices.append(row_index)
    if len(row_indices) != len(selected_indices):
        raise ValueError(f"Found {len(row_indices)} selected token rows; expected {len(selected_indices)}")
    return row_indices, token_rows


def pooled_mean_std(features, batch_size: int):
    """Return mean and standard deviation over time for [B,T,C] features."""
    import torch

    if features.ndim != 3:
        raise ValueError(f"Expected 3D activation [B,T,C], got {tuple(features.shape)}")
    if features.shape[0] != batch_size:
        if features.shape[1] == batch_size:  # Bonito CTC/CRF encoders commonly return [T,B,C].
            features = features.transpose(0, 1)
        else:
            raise ValueError(f"Cannot identify batch axis in feature shape {tuple(features.shape)}")
    features = features.float()
    return features.mean(dim=1), features.var(dim=1, unbiased=False).clamp_min(0).sqrt()


def linear_cka(x, y) -> float:
    """Linear CKA between sample-by-feature matrices, using sample Gram matrices."""
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise ValueError("CKA needs at least two paired samples")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    k = x @ x.T
    l = y @ y.T
    numerator = float((k * l).sum())
    denominator = float(((k * k).sum() * (l * l).sum()) ** 0.5)
    return numerator / denominator if denominator else float("nan")


def main() -> None:
    args = parse_args()
    import numpy as np
    import torch
    from transformers import AutoModel

    repo_root = Path(__file__).resolve().parents[3]
    bonito_src = repo_root / "data" / "bonito"
    sys.path.insert(0, str(bonito_src.parent))
    from bonito.util import load_model

    if not args.bonito_model_dir.is_dir():
        raise FileNotFoundError(f"Bonito model directory is not available: {args.bonito_model_dir}")
    if not (args.bonito_model_dir / "config.toml").is_file():
        raise FileNotFoundError(f"Bonito config.toml not found under {args.bonito_model_dir}")
    if not args.foundation_base_dir.is_dir():
        raise FileNotFoundError(f"Foundation model directory not found: {args.foundation_base_dir}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    signals = np.load(args.signal_npy, mmap_mode="r", allow_pickle=False)
    if signals.ndim != 2:
        raise ValueError(f"Signal array must have shape [N,T], got {signals.shape}")
    n_rows = int(signals.shape[0])
    n_samples = min(int(args.max_samples), n_rows)
    selected = np.linspace(0, n_rows - 1, n_samples, dtype=np.int64)
    selected_set = set(int(i) for i in selected)
    row_indices, token_rows = load_token_rows(args.tokens_jsonl_gz, selected_set)
    if row_indices != selected.tolist():
        raise ValueError("Selected token rows do not match selected signal indices")

    print(f"Loading Bonito from {args.bonito_model_dir}", flush=True)
    bonito = load_model(str(args.bonito_model_dir), str(device), half=False, use_koi=False)
    bonito.eval()
    print(f"Loading foundation base model from {args.foundation_base_dir}", flush=True)
    foundation = AutoModel.from_pretrained(
        str(args.foundation_base_dir),
        trust_remote_code=True,
        local_files_only=True,
    ).to(device)
    foundation.eval()

    bonito_means = []
    bonito_stds = []
    foundation_input_means = []
    foundation_input_stds = []
    foundation_last_means = []
    foundation_last_stds = []
    sample_ids = []
    code_offset = 128

    with torch.inference_mode():
        for start in range(0, n_samples, args.batch_size):
            end = min(start + args.batch_size, n_samples)
            batch_rows = row_indices[start:end]
            batch_size = len(batch_rows)
            signal_batch = torch.as_tensor(
                np.asarray(signals[batch_rows], dtype=np.float32),
                dtype=torch.float32,
                device=device,
            ).unsqueeze(1)

            # This checkpoint is Bonito CRF: its final encoder module is a
            # LinearCRF projection. Capture the preceding recurrent features.
            _, bonito_feature_maps = bonito.encoder(signal_batch, return_features=True)
            bonito_features = bonito_feature_maps[-2]
            b_mean, b_std = pooled_mean_std(bonito_features, batch_size)
            bonito_means.append(b_mean.cpu().numpy())
            bonito_stds.append(b_std.cpu().numpy())

            input_ids = torch.tensor(token_rows[start:end], dtype=torch.long, device=device)
            positions = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
            bert = foundation.bert
            input_embeddings = bert.token_embedding(input_ids) + bert.position_embedding(positions)
            input_embeddings = bert.embedding_dropout(bert.embedding_norm(input_embeddings))
            outputs = foundation(input_ids=input_ids, output_hidden_states=False, return_dict=True)
            last_hidden = outputs.last_hidden_state
            # Keep only signal token positions; omit CLS, BOS and EOS special tokens.
            input_signal = input_embeddings[:, 2:-1, :]
            last_signal = last_hidden[:, 2:-1, :]
            i_mean, i_std = pooled_mean_std(input_signal, batch_size)
            h_mean, h_std = pooled_mean_std(last_signal, batch_size)
            foundation_input_means.append(i_mean.cpu().numpy())
            foundation_input_stds.append(i_std.cpu().numpy())
            foundation_last_means.append(h_mean.cpu().numpy())
            foundation_last_stds.append(h_std.cpu().numpy())
            sample_ids.extend(f"chunks_chunk_{i}" for i in batch_rows)
            print(f"Processed {end}/{n_samples} chunks", flush=True)

    bm = np.concatenate(bonito_means).astype(np.float32)
    bs = np.concatenate(bonito_stds).astype(np.float32)
    im = np.concatenate(foundation_input_means).astype(np.float32)
    ist = np.concatenate(foundation_input_stds).astype(np.float32)
    hm = np.concatenate(foundation_last_means).astype(np.float32)
    hs = np.concatenate(foundation_last_stds).astype(np.float32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "paired_pooled_features.npz",
        row_indices=np.asarray(row_indices, dtype=np.int64),
        sample_ids=np.asarray(sample_ids),
        bonito_mean=bm,
        bonito_std=bs,
        foundation_input_embedding_mean=im,
        foundation_input_embedding_std=ist,
        foundation_last_hidden_mean=hm,
        foundation_last_hidden_std=hs,
    )
    summary = {
        "signal_file": str(args.signal_npy),
        "token_file": str(args.tokens_jsonl_gz),
        "bonito_model_dir": str(args.bonito_model_dir),
        "foundation_base_dir": str(args.foundation_base_dir),
        "device": str(device),
        "sample_count": n_samples,
        "sampling": "evenly spaced chunk indices across the file",
        "bonito_input_preprocessing": "signal rows passed directly as float32; Bonito config has no explicit scaling block, and the saved chunks are already near zero mean/unit-scale",
        "bonito_feature": "last recurrent encoder output immediately before LinearCRFEncoder; 768 channels",
        "feature_pooling": "per-chunk temporal mean and standard deviation",
        "foundation_signal_token_offset": code_offset,
        "linear_cka_on_chunk_mean_features": {
            "bonito_vs_foundation_input_embedding": linear_cka(bm, im),
            "bonito_vs_foundation_last_hidden": linear_cka(bm, hm),
            "foundation_input_embedding_vs_last_hidden": linear_cka(im, hm),
        },
        "note": "CKA measures similarity of sample-level Gram geometry after temporal mean pooling; it is not basecalling accuracy or tokenwise alignment.",
    }
    (args.output_dir / "paired_feature_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
