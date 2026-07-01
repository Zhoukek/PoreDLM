#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[5]
TRAINING_DIR = REPO_ROOT / "src" / "poredlm" / "training"
STAGE4_DIR = TRAINING_DIR / "stage4_finetune"
for import_root in (STAGE4_DIR, TRAINING_DIR, REPO_ROOT / "src", REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from stage2_BERT_Encoder.dataset import (  # noqa: E402
    build_bwav_vocab_lookup,
    load_tokenizer_vocab,
    parse_bwav_token_text,
)
from Basecalling.basecaller_v8_0420.model_dlm import (  # noqa: E402
    BasecallModel,
)


@dataclass
class TokenRecord:
    record_id: str
    input_ids: list[int]
    valid_len: int | None
    source_path: str
    row_index: int
    label: str | None = None


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def open_jsonl_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")


def iter_input_files(paths: list[str], suffixes: tuple[str, ...]) -> Iterator[Path]:
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_dir():
            for suffix in suffixes:
                yield from sorted(path.rglob(f"*{suffix}"))
        elif path.exists():
            yield path
        else:
            raise FileNotFoundError(f"Input path does not exist: {path}")


def infer_valid_len_from_meta(item: dict, seq_len: int) -> int | None:
    meta = item.get("meta")
    if isinstance(meta, dict):
        for key in ("original_token_len", "valid_token_len", "unpadded_token_len"):
            value = meta.get(key)
            if value is not None:
                return max(0, min(int(value), seq_len))
    for key in ("original_token_len", "valid_token_len", "unpadded_token_len", "input_length"):
        value = item.get(key)
        if value is not None:
            return max(0, min(int(value), seq_len))
    return None


def iter_jsonl_records(
    path: Path,
    vocab: dict[str, int] | None,
    bwav_vocab_lookup: dict[int, int] | None,
    unk_token_id: int,
) -> Iterator[TokenRecord]:
    with open_text(path) as handle:
        for row_index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item.get("input_ids"), list):
                input_ids = [int(x) for x in item["input_ids"]]
            elif isinstance(item.get("tokens"), list):
                input_ids = [int(x) for x in item["tokens"]]
            elif isinstance(item.get("text"), str):
                if vocab is None:
                    raise ValueError(
                        "JSONL text input needs --tokenizer-json so <|bwav:*|> tokens "
                        "can be mapped to the stage2/stage3 vocabulary ids."
                    )
                input_ids = parse_bwav_token_text(
                    item["text"],
                    vocab=vocab,
                    unk_token_id=unk_token_id,
                    bwav_vocab_lookup=bwav_vocab_lookup,
                )
            else:
                raise ValueError(f"{path}:{row_index + 1} has neither input_ids/tokens nor text.")

            record_id = str(
                item.get("id")
                or item.get("read_id")
                or item.get("record_id")
                or f"{path.stem}:{row_index}"
            )
            valid_len = infer_valid_len_from_meta(item, len(input_ids))
            label = item.get("label") or item.get("sample") or item.get("dataset")
            yield TokenRecord(
                record_id=record_id,
                input_ids=input_ids,
                valid_len=valid_len,
                source_path=str(path),
                row_index=row_index,
                label=str(label) if label is not None else None,
            )


def iter_npy_records(path: Path, pad_token_id: int) -> Iterator[TokenRecord]:
    arr = np.load(path, allow_pickle=True)
    if arr.ndim == 1:
        rows = [arr]
    elif arr.ndim == 2:
        rows = arr
    else:
        raise ValueError(f"Only 1D or 2D token npy is supported, got shape={arr.shape} from {path}")

    for row_index, row in enumerate(rows):
        input_ids = [int(x) for x in np.asarray(row).tolist()]
        valid_len = int(np.count_nonzero(np.asarray(input_ids) != int(pad_token_id)))
        yield TokenRecord(
            record_id=f"{path.stem}:{row_index}",
            input_ids=input_ids,
            valid_len=valid_len,
            source_path=str(path),
            row_index=row_index,
        )


def iter_records(
    jsonl_inputs: list[str],
    npy_inputs: list[str],
    tokenizer_json: str | None,
    unk_token_id: int,
    pad_token_id: int,
) -> Iterator[TokenRecord]:
    vocab = load_tokenizer_vocab(tokenizer_json) if tokenizer_json else None
    bwav_vocab_lookup = build_bwav_vocab_lookup(vocab) if vocab is not None else None

    for path in iter_input_files(jsonl_inputs, (".jsonl", ".jsonl.gz")):
        yield from iter_jsonl_records(path, vocab, bwav_vocab_lookup, unk_token_id)

    for path in iter_input_files(npy_inputs, (".npy",)):
        yield from iter_npy_records(path, pad_token_id=pad_token_id)


def make_batch(
    records: list[TokenRecord],
    pad_token_id: int,
    max_length: int | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    lengths = [len(record.input_ids) for record in records]
    if max_length is not None:
        target_len = min(max(lengths), int(max_length))
    else:
        target_len = max(lengths)
    if target_len <= 0:
        raise ValueError("Encountered an empty batch.")

    input_ids = torch.full((len(records), target_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(records), target_len), dtype=torch.long)
    effective_lengths: list[int] = []

    for row, record in enumerate(records):
        ids = record.input_ids[:target_len]
        if ids:
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)

        if record.valid_len is None:
            valid = sum(1 for token_id in ids if int(token_id) != int(pad_token_id))
        else:
            valid = min(int(record.valid_len), len(ids))
        if valid > 0:
            attention_mask[row, :valid] = 1
        effective_lengths.append(valid)

    return input_ids.to(device), attention_mask.to(device), effective_lengths


def forward_context_and_ode(
    model: BasecallModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    backbone_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(model.backbone, "context_encoder"):
        raise ValueError("The loaded Stage3 model does not expose backbone.context_encoder.")
    if not hasattr(model.backbone, "elf_denoiser"):
        raise ValueError("The loaded Stage3 model does not expose backbone.elf_denoiser.")

    context_parts = []
    ode_parts = []
    chunk_size = max(0, int(backbone_chunk_size))
    if chunk_size <= 0 or input_ids.shape[1] <= chunk_size:
        ranges = [(0, input_ids.shape[1])]
    else:
        ranges = [(start, min(start + chunk_size, input_ids.shape[1])) for start in range(0, input_ids.shape[1], chunk_size)]

    old_feature_source = model.feature_source
    try:
        model.feature_source = "context_hidden"
        for start, end in ranges:
            chunk_ids = input_ids[:, start:end]
            chunk_mask = attention_mask[:, start:end]
            context = model._forward_backbone_hidden(chunk_ids, attention_mask=chunk_mask)
            ode = model._ode_from_context_hidden(context, attention_mask=chunk_mask)
            context_parts.append(context)
            ode_parts.append(ode)
    finally:
        model.feature_source = old_feature_source

    return torch.cat(context_parts, dim=1), torch.cat(ode_parts, dim=1)


def summarize_scores(
    l2: torch.Tensor,
    cosine_distance: torch.Tensor,
    valid_len: int,
    token_ids: torch.Tensor,
    top_k: int,
) -> dict:
    if valid_len <= 0:
        return {
            "valid_len": 0,
            "mean_l2": None,
            "max_l2": None,
            "p95_l2": None,
            "mean_normed_l2": None,
            "mean_cosine_distance": None,
            "max_cosine_distance": None,
            "top_positions": [],
        }

    l2_valid = l2[:valid_len].float().cpu()
    cos_valid = cosine_distance[:valid_len].float().cpu()
    token_valid = token_ids[:valid_len].detach().cpu()
    k = min(max(0, int(top_k)), valid_len)

    top_positions = []
    if k > 0:
        values, indices = torch.topk(l2_valid, k=k, largest=True, sorted=True)
        for score, pos in zip(values.tolist(), indices.tolist()):
            top_positions.append(
                {
                    "position": int(pos),
                    "token_id": int(token_valid[pos].item()),
                    "l2": float(score),
                    "cosine_distance": float(cos_valid[pos].item()),
                }
            )

    hidden_dim = 1
    if l2_valid.numel() > 0:
        # l2 is already reduced over hidden dim; caller stores sqrt(hidden_dim) scale separately.
        hidden_dim = int(getattr(summarize_scores, "_hidden_dim", 1))

    return {
        "valid_len": int(valid_len),
        "mean_l2": float(l2_valid.mean().item()),
        "max_l2": float(l2_valid.max().item()),
        "p95_l2": float(torch.quantile(l2_valid, 0.95).item()),
        "mean_normed_l2": float((l2_valid / math.sqrt(hidden_dim)).mean().item()),
        "mean_cosine_distance": float(cos_valid.mean().item()),
        "max_cosine_distance": float(cos_valid.max().item()),
        "top_positions": top_positions,
    }


def write_optional_score_arrays(
    outdir: Path | None,
    records: list[TokenRecord],
    l2: torch.Tensor,
    cosine_distance: torch.Tensor,
    effective_lengths: list[int],
) -> list[str | None]:
    if outdir is None:
        return [None for _ in records]
    outdir.mkdir(parents=True, exist_ok=True)
    paths: list[str | None] = []
    for idx, record in enumerate(records):
        safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in record.record_id)[:180]
        path = outdir / f"{safe_id}.npz"
        valid_len = effective_lengths[idx]
        np.savez_compressed(
            path,
            l2=l2[idx, :valid_len].detach().cpu().numpy().astype(np.float32),
            cosine_distance=cosine_distance[idx, :valid_len].detach().cpu().numpy().astype(np.float32),
        )
        paths.append(str(path))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score potential DNA modification by comparing BERT/context embeddings "
            "with deterministic ELF ODE-refined embeddings."
        )
    )
    parser.add_argument("--model-name-or-path", required=True, help="Stage3 HF DLM model directory.")
    parser.add_argument("--tokenizer-json", default=None, help="Tokenizer JSON used to map <|bwav:*|> text to vocab ids.")
    parser.add_argument("--jsonl", nargs="*", default=[], help="Input jsonl/jsonl.gz file(s) or directories.")
    parser.add_argument("--npy", nargs="*", default=[], help="Input token npy file(s) or directories.")
    parser.add_argument("--output-jsonl", required=True, help="Output score jsonl/jsonl.gz path.")
    parser.add_argument("--score-array-dir", default=None, help="Optional directory for per-read l2/cosine npz arrays.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=None, help="Optional truncate length, e.g. 1600.")
    parser.add_argument("--pad-token-id", type=int, default=1)
    parser.add_argument("--unk-token-id", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--backbone-chunk-size", type=int, default=600)
    parser.add_argument("--elf-ode-steps", type=int, default=4)
    parser.add_argument("--elf-ode-start-t", type=float, default=0.85)
    parser.add_argument("--elf-self-cond-cfg-scale", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not args.jsonl and not args.npy:
        raise ValueError("Provide at least one --jsonl or --npy input.")

    device = torch.device(args.device)
    model = BasecallModel(
        model_path=args.model_name_or_path,
        feature_source="context_hidden",
        freeze_backbone=True,
        pre_head_type="none",
        head_type="ctc",
        backbone_chunk_size=args.backbone_chunk_size,
        elf_ode_steps=args.elf_ode_steps,
        elf_ode_start_t=args.elf_ode_start_t,
        elf_self_cond_cfg_scale=args.elf_self_cond_cfg_scale,
    )
    model.eval().to(device)

    if args.dtype == "float16":
        model = model.half()
    elif args.dtype == "bfloat16":
        model = model.to(dtype=torch.bfloat16)
    elif args.dtype == "float32":
        model = model.float()

    score_array_dir = Path(args.score_array_dir).expanduser() if args.score_array_dir else None
    records_iter = iter_records(
        jsonl_inputs=args.jsonl,
        npy_inputs=args.npy,
        tokenizer_json=args.tokenizer_json,
        unk_token_id=args.unk_token_id,
        pad_token_id=args.pad_token_id,
    )

    total = 0
    with open_jsonl_writer(Path(args.output_jsonl).expanduser()) as writer:
        batch: list[TokenRecord] = []
        pbar = tqdm(desc="scoring reads", unit="read")
        for record in records_iter:
            if args.limit is not None and total >= args.limit:
                break
            batch.append(record)
            if len(batch) < args.batch_size:
                continue

            total += process_batch(args, model, device, batch, score_array_dir, writer)
            pbar.update(len(batch))
            batch = []

        if batch and (args.limit is None or total < args.limit):
            if args.limit is not None:
                batch = batch[: max(0, args.limit - total)]
            total += process_batch(args, model, device, batch, score_array_dir, writer)
            pbar.update(len(batch))
        pbar.close()

    print(f"Wrote {total} records to {args.output_jsonl}")


def process_batch(
    args: argparse.Namespace,
    model: BasecallModel,
    device: torch.device,
    batch: list[TokenRecord],
    score_array_dir: Path | None,
    writer,
) -> int:
    input_ids, attention_mask, effective_lengths = make_batch(
        batch,
        pad_token_id=args.pad_token_id,
        max_length=args.max_length,
        device=device,
    )

    with torch.inference_mode():
        context, ode = forward_context_and_ode(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            backbone_chunk_size=args.backbone_chunk_size,
        )
        context_f = context.float()
        ode_f = ode.float()
        delta = ode_f - context_f
        l2 = torch.linalg.vector_norm(delta, dim=-1)
        cosine_distance = 1.0 - F.cosine_similarity(context_f, ode_f, dim=-1, eps=1e-8)
        hidden_dim = int(context_f.shape[-1])

    setattr(summarize_scores, "_hidden_dim", hidden_dim)
    score_paths = write_optional_score_arrays(score_array_dir, batch, l2, cosine_distance, effective_lengths)

    for idx, record in enumerate(batch):
        summary = summarize_scores(
            l2=l2[idx],
            cosine_distance=cosine_distance[idx],
            valid_len=effective_lengths[idx],
            token_ids=input_ids[idx],
            top_k=args.top_k,
        )
        payload = {
            "id": record.record_id,
            "label": record.label,
            "source_path": record.source_path,
            "row_index": record.row_index,
            "seq_len": int(input_ids.shape[1]),
            "pad_token_id": int(args.pad_token_id),
            "backbone_chunk_size": int(args.backbone_chunk_size),
            "elf_ode_steps": int(args.elf_ode_steps),
            "elf_ode_start_t": float(args.elf_ode_start_t),
            **summary,
        }
        if score_paths[idx] is not None:
            payload["score_array_path"] = score_paths[idx]
        writer.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return len(batch)


if __name__ == "__main__":
    main()
