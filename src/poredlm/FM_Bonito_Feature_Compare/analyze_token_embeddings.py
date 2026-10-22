#!/usr/bin/env python3
"""Analyze BERT token-embedding geometry using only Python's standard library.

This reads the token embedding matrix directly from a safetensors checkpoint.
It does not run the Transformer; outputs describe the learned input-token
vectors, before position embeddings, LayerNorm, or contextualization.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import mmap
import random
import re
import statistics
import struct
from pathlib import Path


TOKEN_RE = re.compile(r"<\|bwav:(\d+)\|>")


def read_safetensors_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    return header, 8 + header_size


def get_tensor_vectors(mapped: mmap.mmap, data_start: int, tensor_info: dict, rows: list[int], dim: int) -> dict[int, tuple[float, ...]]:
    start, end = tensor_info["data_offsets"]
    if tensor_info["shape"][1] != dim or end - start != tensor_info["shape"][0] * dim * 4:
        raise ValueError(f"Expected float32 matrix with width {dim}, got {tensor_info}")
    vectors = {}
    for row in rows:
        offset = data_start + start + row * dim * 4
        vectors[row] = struct.unpack_from(f"<{dim}f", mapped, offset)
    return vectors


def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {}

    def quantile(q: float) -> float:
        index = (len(ordered) - 1) * q
        lo = int(index)
        hi = min(lo + 1, len(ordered) - 1)
        frac = index - lo
        return ordered[lo] * (1 - frac) + ordered[hi] * frac

    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "p05": quantile(0.05),
        "median": quantile(0.50),
        "p95": quantile(0.95),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("/mnt/zzbnew/poregpt/models/HF_RSQ741C12V523_DNABERT_V543/base"))
    parser.add_argument("--tokens", type=Path, default=Path("data/tokens/eval00001_chunks.jsonl.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path("feature_analysis"))
    parser.add_argument("--sample-rows", type=int, default=256, help="Initial rows used for adjacent/far token cosine sampling")
    parser.add_argument("--pair-step", type=int, default=64, help="Sample one pair every N token positions")
    args = parser.parse_args()

    config = json.loads((args.model_dir / "config.json").read_text())
    dim = int(config["hidden_size"])
    offset = 128  # Verified against this checkpoint's tokenizer.json.
    model_file = args.model_dir / "model.safetensors"
    header, data_start = read_safetensors_header(model_file)
    tensor_info = header["bert.token_embedding.weight"]
    vocab_size = int(tensor_info["shape"][0])
    if int(tensor_info["shape"][1]) != dim:
        raise ValueError("Model config and token embedding width disagree")

    counts: collections.Counter[int] = collections.Counter()
    n_records = n_tokens = n_sample_records = 0
    local_pairs: list[tuple[int, int]] = []
    far_pairs: list[tuple[int, int]] = []
    rng = random.Random(20261019)

    with gzip.open(args.tokens, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            ids = [int(value) for value in TOKEN_RE.findall(record["text"])]
            if not ids:
                continue
            if min(ids) < 0 or max(ids) + offset >= vocab_size:
                raise ValueError(f"Signal code ID outside BERT vocabulary: {record.get('id')}")
            counts.update(ids)
            n_records += 1
            n_tokens += len(ids)
            if n_sample_records < args.sample_rows:
                for pos in range(0, len(ids) - 128, args.pair_step):
                    local_pairs.append((ids[pos], ids[pos + 1]))
                    far_pos = pos + rng.randint(64, min(512, len(ids) - pos - 1))
                    far_pairs.append((ids[pos], ids[far_pos]))
                n_sample_records += 1

    if set(counts) != set(range(2401)):
        raise ValueError(f"Expected 2,401 distinct signal codes, found {len(counts)}")

    with model_file.open("rb") as handle:
        mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        vectors = get_tensor_vectors(
            mapped,
            data_start,
            tensor_info,
            [code + offset for code in range(2401)],
            dim,
        )
        norms_by_id = {
            token_id: math.sqrt(sum(value * value for value in vec))
            for token_id, vec in vectors.items()
        }
        unweighted_norms = list(norms_by_id.values())
        weighted_norm_sum = sum(norms_by_id[code + offset] * count for code, count in counts.items())
        weighted_norm_mean = weighted_norm_sum / n_tokens
        adjacent_cos = [cosine(vectors[a + offset], vectors[b + offset]) for a, b in local_pairs]
        distinct_adjacent_cos = [
            cosine(vectors[a + offset], vectors[b + offset])
            for a, b in local_pairs
            if a != b
        ]
        far_cos = [cosine(vectors[a + offset], vectors[b + offset]) for a, b in far_pairs]

    entropy = -sum((count / n_tokens) * math.log2(count / n_tokens) for count in counts.values())
    effective_codes = 2 ** entropy
    top_codes = [
        {"code": code, "bert_token_id": code + offset, "count": count, "fraction": count / n_tokens}
        for code, count in counts.most_common(20)
    ]
    result = {
        "scope": "token_embedding.weight only; no position embeddings, embedding LayerNorm, Transformer layers, or Bonito features",
        "model": str(args.model_dir),
        "tokens_file": str(args.tokens),
        "num_records": n_records,
        "num_tokens": n_tokens,
        "num_unique_signal_codes": len(counts),
        "signal_code_to_bert_token_offset": offset,
        "embedding_dim": dim,
        "token_frequency_entropy_bits": entropy,
        "effective_codes_from_entropy": effective_codes,
        "embedding_norm_unweighted_code_types": summarize(unweighted_norms),
        "embedding_norm_corpus_frequency_weighted_mean": weighted_norm_mean,
        "adjacent_token_embedding_cosine": summarize(adjacent_cos),
        "adjacent_same_code_fraction": sum(a == b for a, b in local_pairs) / len(local_pairs),
        "adjacent_distinct_code_embedding_cosine": summarize(distinct_adjacent_cos),
        "same_chunk_distant_token_embedding_cosine": summarize(far_cos),
        "distinct_adjacent_minus_distant_mean_cosine": statistics.fmean(distinct_adjacent_cos) - statistics.fmean(far_cos),
        "pair_sampling": {
            "sample_rows": n_sample_records,
            "pair_step": args.pair_step,
            "far_offset_range": [64, 512],
            "pairs_per_comparison": len(adjacent_cos),
            "seed": 20261019,
        },
        "top_20_signal_codes": top_codes,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_dir / "token_embedding_analysis.json"
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    output_csv = args.output_dir / "token_embedding_code_frequencies.csv"
    with output_csv.open("w", encoding="utf-8") as handle:
        handle.write("signal_code,bert_token_id,count,fraction,embedding_l2_norm\n")
        for code in range(2401):
            count = counts[code]
            handle.write(f"{code},{code + offset},{count},{count / n_tokens:.10g},{norms_by_id[code + offset]:.9g}\n")

    print(f"records={n_records} tokens={n_tokens} unique_codes={len(counts)}")
    print(f"effective_codes={effective_codes:.2f} / 2401")
    print(f"embedding_norm_unweighted={summarize(unweighted_norms)}")
    print(f"embedding_norm_frequency_weighted_mean={weighted_norm_mean:.6f}")
    print(f"adjacent_cos={summarize(adjacent_cos)}")
    print(f"distant_cos={summarize(far_cos)}")
    print(f"wrote {output_json} and {output_csv}")


if __name__ == "__main__":
    main()
