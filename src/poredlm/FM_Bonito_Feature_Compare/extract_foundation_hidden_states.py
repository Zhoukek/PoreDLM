#!/usr/bin/env python3
"""Extract layerwise contextual representations from the PoreBERT base model.

The script pools signal-token positions within each selected chunk and saves
per-chunk summaries for the embedding output, all 12 Transformer layers, and
the model's final normalized hidden state. Requires the PoreDLM conda runtime.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, default=Path("data/tokens/eval00001_chunks.jsonl.gz"))
    parser.add_argument("--model-dir", type=Path, default=Path("/mnt/zzbnew/poregpt/models/HF_RSQ741C12V523_DNABERT_V543/base"))
    parser.add_argument("--output-dir", type=Path, default=Path("feature_analysis/foundation_hidden_states"))
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def read_selected_tokens(path: Path, selected: set[int]) -> tuple[list[int], list[str], list[list[int]]]:
    token_re = re.compile(r"<\|bwav:(\d+)\|>")
    rows: list[int] = []
    ids: list[str] = []
    tokens: list[list[int]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for row, line in enumerate(handle):
            if row not in selected:
                continue
            record = json.loads(line)
            codes = [int(x) for x in token_re.findall(record["text"])]
            if len(codes) != 1500 or record.get("id") != f"chunks_chunk_{row}":
                raise ValueError(f"Token row {row} has unexpected length or id")
            if not codes or min(codes) < 0 or max(codes) > 2400:
                raise ValueError(f"Token code out of range in row {row}")
            # Project convention: [CLS=5, BOS=2], signal code + 128, [EOS=3].
            tokens.append([5, 2, *(x + 128 for x in codes), 3])
            rows.append(row)
            ids.append(record["id"])
    if set(rows) != selected:
        raise ValueError(f"Selected {len(rows)} token rows; expected {len(selected)}")
    return rows, ids, tokens


def cka(x, y) -> float:
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    k = x @ x.T
    l = y @ y.T
    denominator = float(((k * k).sum() * (l * l).sum()) ** 0.5)
    return float((k * l).sum()) / denominator if denominator else float("nan")


def main() -> None:
    args = args_parser()
    import numpy as np
    import torch
    from transformers import AutoModel

    torch.set_num_threads(args.threads)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(
        str(args.model_dir),
        trust_remote_code=True,
        local_files_only=True,
    ).to(device)
    model.eval()

    # The token stream is verified to be row-aligned with signal/references.
    n_records = sum(1 for _ in gzip.open(args.tokens, "rt", encoding="utf-8"))
    n_samples = min(args.max_samples, n_records)
    selected_rows = np.linspace(0, n_records - 1, n_samples, dtype=np.int64).tolist()
    selected_set = set(selected_rows)
    rows, sample_ids, token_rows = read_selected_tokens(args.tokens, selected_set)
    if rows != selected_rows:
        raise ValueError("Selected token rows are not in expected sorted order")

    n_layers = int(model.config.num_hidden_layers)
    means: list[list] = [[] for _ in range(n_layers + 2)]
    stds: list[list] = [[] for _ in range(n_layers + 2)]
    mean_norms: list[list[float]] = [[] for _ in range(n_layers + 2)]
    adjacent_cosines: list[list[float]] = [[] for _ in range(n_layers + 2)]

    with torch.inference_mode():
        for start in range(0, n_samples, args.batch_size):
            end = min(start + args.batch_size, n_samples)
            input_ids = torch.tensor(token_rows[start:end], dtype=torch.long, device=device)
            outputs = model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
            signal_hidden = [state[:, 2:-1, :].float() for state in outputs.hidden_states]
            signal_hidden.append(outputs.last_hidden_state[:, 2:-1, :].float())
            for layer_index, state in enumerate(signal_hidden):
                means[layer_index].append(state.mean(dim=1).cpu().numpy())
                stds[layer_index].append(state.var(dim=1, unbiased=False).clamp_min(0).sqrt().cpu().numpy())
                mean_norms[layer_index].extend(state.norm(dim=-1).mean(dim=1).cpu().tolist())
                adjacent = (state[:, 1:, :] * state[:, :-1, :]).sum(dim=-1)
                norms = state[:, 1:, :].norm(dim=-1) * state[:, :-1, :].norm(dim=-1)
                adjacent_cosines[layer_index].extend((adjacent / norms.clamp_min(1e-12)).mean(dim=1).cpu().tolist())
            print(f"Processed {end}/{n_samples} chunks", flush=True)

    layer_means = np.stack([np.concatenate(item).astype(np.float32) for item in means], axis=1)
    layer_stds = np.stack([np.concatenate(item).astype(np.float32) for item in stds], axis=1)
    # index 0..12 are embedding output and transformer layers; 13 is final LayerNorm output.
    layer_names = ["embedding_output"] + [f"transformer_layer_{i}" for i in range(1, n_layers + 1)] + ["final_normalized_hidden"]
    cka_matrix = np.empty((len(layer_names), len(layer_names)), dtype=np.float32)
    for i in range(len(layer_names)):
        for j in range(len(layer_names)):
            cka_matrix[i, j] = cka(layer_means[:, i, :], layer_means[:, j, :])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "foundation_layerwise_pooled_features.npz",
        row_indices=np.asarray(rows, dtype=np.int64),
        sample_ids=np.asarray(sample_ids),
        layer_names=np.asarray(layer_names),
        mean_features=layer_means,
        std_features=layer_stds,
        layerwise_cka=cka_matrix,
    )
    summary = {
        "model_dir": str(args.model_dir),
        "tokens_file": str(args.tokens),
        "device": str(device),
        "sample_count": n_samples,
        "sample_policy": "evenly spaced rows across token file",
        "signal_tokens_per_chunk": 1500,
        "special_tokens": {"CLS": 5, "BOS": 2, "EOS": 3, "signal_code_offset": 128},
        "layers": [
            {
                "name": name,
                "mean_token_l2_norm": float(np.mean(mean_norms[i])),
                "mean_adjacent_token_cosine": float(np.mean(adjacent_cosines[i])),
            }
            for i, name in enumerate(layer_names)
        ],
        "interpretation": "Per-chunk temporal pooled embeddings and token-local geometry; no Bonito comparison or basecalling probe in this run.",
    }
    (args.output_dir / "foundation_layerwise_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
