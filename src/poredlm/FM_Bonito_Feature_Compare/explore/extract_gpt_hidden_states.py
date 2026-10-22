#!/usr/bin/env python3
"""Extract V615 GPT/OLMo hidden states on the exact paired-feature chunks.

The Stage-4 GPT run uses BwavTokenizer without CLS, which yields
``[BOS=2, signal_code+128..., EOS=3]``.  This script verifies that mapping
against the V615 tokenizer before feature extraction and pools only the 1,500
signal-code positions.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens-jsonl-gz", type=Path,
        default=SCRIPT_DIR.parent / "data/tokens/eval00001_chunks.jsonl.gz",
    )
    parser.add_argument(
        "--paired-npz", type=Path,
        default=SCRIPT_DIR.parent / "feature_analysis/paired_features_n128/paired_pooled_features.npz",
    )
    parser.add_argument(
        "--model-dir", type=Path,
        default=Path("/mnt/zzbnew/poregpt/models/HF_RSQ741C12A529_MNAOLMO_V615/base"),
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "gpt_layerwise_n128")
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Optional diagnostic cap; omit to use every row from the paired feature archive.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def read_selected_rows(path: Path, selected_rows: set[int]) -> tuple[list[int], list[str], list[list[int]]]:
    token_re = re.compile(r"<\|bwav:(\d+)\|>")
    rows: list[int] = []
    ids: list[str] = []
    token_rows: list[list[int]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for row, line in enumerate(handle):
            if row not in selected_rows:
                continue
            record = json.loads(line)
            codes = [int(value) for value in token_re.findall(record["text"])]
            if len(codes) != 1500 or record.get("id") != f"chunks_chunk_{row}":
                raise ValueError(f"Unexpected token record at row {row}")
            if not codes or min(codes) < 0 or max(codes) > 2400:
                raise ValueError(f"Signal code outside expected range at row {row}")
            # Matches the active Stage-4 GPT run: add_cls_token=false.
            token_rows.append([2, *(code + 128 for code in codes), 3])
            rows.append(row)
            ids.append(record["id"])
    return rows, ids, token_rows


def main() -> None:
    args = parse_args()
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing an accidental CPU extraction")
    torch.set_num_threads(args.threads)
    paired = np.load(args.paired_npz)
    selected_rows = paired["row_indices"].astype(np.int64).tolist()
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive when provided")
        selected_rows = selected_rows[:args.max_samples]
    print(f"Reading {len(selected_rows)} matched token chunks", flush=True)
    rows, sample_ids, token_rows = read_selected_rows(args.tokens_jsonl_gz, set(selected_rows))
    if rows != selected_rows:
        raise ValueError("GPT inputs do not have the same row order as the paired Bonito/BERT/DLM features")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    mapping_checks = {}
    for code in (0, 1, 1200, 2399, 2400):
        token = f"<|bwav:{code}|>"
        token_id = int(tokenizer.convert_tokens_to_ids(token))
        expected = code + 128
        if token_id != expected:
            raise ValueError(f"V615 tokenizer mapping mismatch: {token} -> {token_id}, expected {expected}")
        mapping_checks[token] = token_id

    device = torch.device(args.device)
    print(f"Loading GPT backbone from {args.model_dir}", flush=True)
    model = AutoModel.from_pretrained(str(args.model_dir), local_files_only=True).to(device).eval()
    n_transformer_layers = int(model.config.num_hidden_layers)
    n_representations = n_transformer_layers + 2  # embedding, each block, final normalized output
    mean_parts: list[list[np.ndarray]] = [[] for _ in range(n_representations)]
    std_parts: list[list[np.ndarray]] = [[] for _ in range(n_representations)]
    mean_norms: list[list[float]] = [[] for _ in range(n_representations)]
    adjacent_cosines: list[list[float]] = [[] for _ in range(n_representations)]
    final_hidden_max_abs_diff = 0.0

    with torch.inference_mode():
        for start in range(0, len(token_rows), args.batch_size):
            end = min(start + args.batch_size, len(token_rows))
            input_ids = torch.tensor(token_rows[start:end], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            if outputs.hidden_states is None or len(outputs.hidden_states) != n_transformer_layers + 1:
                raise ValueError("Unexpected GPT hidden-state count")
            states = list(outputs.hidden_states)
            states.append(outputs.last_hidden_state)
            final_hidden_max_abs_diff = max(
                final_hidden_max_abs_diff,
                float((states[-1].float() - states[-2].float()).abs().max().cpu()),
            )
            for representation_index, state in enumerate(states):
                signal_state = state[:, 1:-1, :].float()
                mean_parts[representation_index].append(signal_state.mean(dim=1).cpu().numpy())
                std_parts[representation_index].append(
                    signal_state.var(dim=1, unbiased=False).clamp_min(0).sqrt().cpu().numpy()
                )
                mean_norms[representation_index].extend(signal_state.norm(dim=-1).mean(dim=1).cpu().tolist())
                adjacent_dot = (signal_state[:, 1:, :] * signal_state[:, :-1, :]).sum(dim=-1)
                adjacent_norm = signal_state[:, 1:, :].norm(dim=-1) * signal_state[:, :-1, :].norm(dim=-1)
                adjacent_cosines[representation_index].extend(
                    (adjacent_dot / adjacent_norm.clamp_min(1e-12)).mean(dim=1).cpu().tolist()
                )
            print(f"Processed {end}/{len(token_rows)} chunks", flush=True)

    mean_features = np.stack([np.concatenate(parts).astype(np.float32) for parts in mean_parts], axis=1)
    std_features = np.stack([np.concatenate(parts).astype(np.float32) for parts in std_parts], axis=1)
    layer_names = (
        ["embedding_output"]
        + [f"transformer_layer_{index}" for index in range(1, n_transformer_layers + 1)]
        + ["final_normalized_hidden"]
    )
    if mean_features.shape[1] != len(layer_names):
        raise RuntimeError("Layer feature count mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "gpt_layerwise_pooled_features.npz",
        row_indices=np.asarray(rows, dtype=np.int64),
        sample_ids=np.asarray(sample_ids),
        layer_names=np.asarray(layer_names),
        mean_features=mean_features,
        std_features=std_features,
    )
    np.savez_compressed(
        args.output_dir / "gpt_final_pooled_features.npz",
        row_indices=np.asarray(rows, dtype=np.int64),
        sample_ids=np.asarray(sample_ids),
        gpt_final_hidden_mean=mean_features[:, -1, :],
        gpt_final_hidden_std=std_features[:, -1, :],
        gpt_hidden_layer_minus_one_mean=mean_features[:, -2, :],
        gpt_hidden_layer_minus_one_std=std_features[:, -2, :],
    )
    summary = {
        "model_dir": str(args.model_dir),
        "model_class": type(model).__name__,
        "model_type": model.config.model_type,
        "architecture": list(getattr(model.config, "architectures", [])),
        "device": str(device),
        "sample_count": len(rows),
        "row_indices_match_paired_features": bool(np.array_equal(np.asarray(rows), paired["row_indices"][:len(rows)])),
        "input_layout": "[BOS=2, 1500 signal_code+128 positions, EOS=3]; no CLS, matching active GPT Stage-4 run",
        "tokenizer_mapping_checks": mapping_checks,
        "basecall_feature_source": "hidden_states[-1] (the current Stage-4 GPT run uses hidden_layer=-1)",
        "final_normalized_vs_hidden_states_minus_one_max_abs_difference": final_hidden_max_abs_diff,
        "layers": [
            {
                "name": name,
                "mean_token_l2_norm": float(np.mean(mean_norms[index])),
                "mean_adjacent_token_cosine": float(np.mean(adjacent_cosines[index])),
            }
            for index, name in enumerate(layer_names)
        ],
    }
    (args.output_dir / "gpt_layerwise_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
