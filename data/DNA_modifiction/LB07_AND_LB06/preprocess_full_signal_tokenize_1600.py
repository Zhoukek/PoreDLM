from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(
    parent for parent in [SCRIPT_DIR, *SCRIPT_DIR.parents]
    if (parent / "src" / "poredlm").exists()
)
POREDLM_SRC = REPO_ROOT / "src" / "poredlm"
STAGE2_TOKENIZER_DIR = POREDLM_SRC / "data" / "stage2_BERT_Encoder"

for path in (REPO_ROOT / "src", POREDLM_SRC, STAGE2_TOKENIZER_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


DEFAULT_MODEL_CKPT = (
    "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/"
    "stage1_tokenizer/runs/"
    "03_S0_HG002_UNMOD_35g_model_type_1_cnn_type_0_distill_0.1_8k_vq_apple/"
    "models/porepgt_vqe_tokenizer.final.pth"
)


def tokens_to_text(
    token_ids,
    *,
    target_length: int,
    pad_token_text: str,
) -> str:
    import numpy as np

    token_ids = np.asarray(token_ids, dtype=np.int64)[:target_length]
    pieces = [f"<|bwav:{int(token_id)}|>" for token_id in token_ids]
    if len(pieces) < target_length:
        pieces.extend([pad_token_text] * (target_length - len(pieces)))
    return "".join(pieces)


def tokens_to_vocab_input_ids(
    token_ids,
    *,
    target_length: int,
    codebook_vocab_offset: int,
    pad_token_id: int,
) -> tuple[list[int], int, bool]:
    import numpy as np

    token_ids = np.asarray(token_ids, dtype=np.int64)
    original_length = int(token_ids.size)
    valid_length = min(original_length, target_length)
    was_truncated = original_length > target_length

    input_ids = [int(token_id) + int(codebook_vocab_offset) for token_id in token_ids[:valid_length]]
    if valid_length < target_length:
        input_ids.extend([int(pad_token_id)] * (target_length - valid_length))
    return input_ids, original_length, was_truncated


def get_record_id(obj: dict, fallback_index: int) -> str:
    for key in ("read_id", "id", "signal_key"):
        value = obj.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return f"record_{fallback_index:08d}"


def process_one_jsonl(
    input_jsonl: str | Path,
    output_jsonl_gz: str | Path,
    tokenizer,
    *,
    target_token_length: int = 1600,
    pad_token_id: int = 1,
    pad_token_text: str = "<|pad|>",
    codebook_vocab_offset: int = 5,
    strategy: str = "apple",
    spike_window_size: int = 6000,
    count_total_first: bool = True,
) -> dict:
    import numpy as np
    from signal_prepare import (
        count_sample_records,
        iter_sample_records,
        nanopore_process_signal,
        parse_maybe_list,
    )

    input_jsonl = Path(input_jsonl)
    output_jsonl_gz = Path(output_jsonl_gz)
    output_jsonl_gz.parent.mkdir(parents=True, exist_ok=True)

    if target_token_length <= 0:
        raise ValueError(f"target_token_length must be positive, got {target_token_length}")

    print("=" * 80)
    print(f"Input jsonl: {input_jsonl}")
    print(f"Output jsonl.gz: {output_jsonl_gz}")
    print("Process: full signal apple preprocessing -> VQE tokenizer -> fixed token length")
    print(f"strategy: {strategy}")
    print(f"spike_window_size: {spike_window_size}")
    print(f"target_token_length: {target_token_length}")
    print(f"pad_token_id: {pad_token_id} ({pad_token_text})")
    print(f"codebook_vocab_offset: {codebook_vocab_offset}")
    print("=" * 80)

    if count_total_first:
        total_sample_records = count_sample_records(input_jsonl)
        print(f"Sample records: {total_sample_records}")
    else:
        total_sample_records = None

    stats = {
        "input_jsonl": str(input_jsonl),
        "output_jsonl_gz": str(output_jsonl_gz),
        "process_order": "full_signal_apple_preprocess_then_tokenize_no_signal_chunk",
        "target_token_length": target_token_length,
        "pad_token_id": pad_token_id,
        "pad_token_text": pad_token_text,
        "codebook_vocab_offset": codebook_vocab_offset,
        "strategy": strategy,
        "spike_window_size": spike_window_size,
        "total_records": 0,
        "written_records": 0,
        "invalid_signal_records": 0,
        "preprocess_empty_records": 0,
        "tokenizer_empty_records": 0,
        "padded_records": 0,
        "truncated_records": 0,
        "min_raw_signal_len": None,
        "max_raw_signal_len": None,
        "min_processed_signal_len": None,
        "max_processed_signal_len": None,
        "min_original_token_len": None,
        "max_original_token_len": None,
    }

    start_time = time.time()
    iterator = iter_sample_records(input_jsonl)

    if tqdm is not None:
        iterator = tqdm(
            iterator,
            total=total_sample_records,
            desc=f"Tokenizing {input_jsonl.name}",
            ncols=120,
        )

    with gzip.open(output_jsonl_gz, "wt", encoding="utf-8") as fout:
        for record_index, obj in enumerate(iterator):
            stats["total_records"] += 1
            record_id = get_record_id(obj, record_index)

            signal = parse_maybe_list(obj.get("signal"))
            if not isinstance(signal, list):
                stats["invalid_signal_records"] += 1
                continue

            signal = np.asarray(signal, dtype=np.float32)
            if signal.ndim != 1:
                stats["invalid_signal_records"] += 1
                continue

            raw_signal_len = int(signal.size)
            stats["min_raw_signal_len"] = raw_signal_len if stats["min_raw_signal_len"] is None else min(stats["min_raw_signal_len"], raw_signal_len)
            stats["max_raw_signal_len"] = raw_signal_len if stats["max_raw_signal_len"] is None else max(stats["max_raw_signal_len"], raw_signal_len)

            processed_signal = nanopore_process_signal(
                signal,
                strategy=strategy,
                spike_window_size=spike_window_size,
            )

            processed_signal = np.asarray(processed_signal, dtype=np.float32)
            processed_signal_len = int(processed_signal.size)
            if processed_signal_len == 0:
                stats["preprocess_empty_records"] += 1
                continue

            stats["min_processed_signal_len"] = processed_signal_len if stats["min_processed_signal_len"] is None else min(stats["min_processed_signal_len"], processed_signal_len)
            stats["max_processed_signal_len"] = processed_signal_len if stats["max_processed_signal_len"] is None else max(stats["max_processed_signal_len"], processed_signal_len)

            token_ids = tokenizer._tokenize_chunked_signal(processed_signal)
            if token_ids.size == 0:
                stats["tokenizer_empty_records"] += 1

            input_ids, original_token_len, was_truncated = tokens_to_vocab_input_ids(
                token_ids=token_ids,
                target_length=target_token_length,
                codebook_vocab_offset=codebook_vocab_offset,
                pad_token_id=pad_token_id,
            )

            if original_token_len < target_token_length:
                stats["padded_records"] += 1
            if was_truncated:
                stats["truncated_records"] += 1

            stats["min_original_token_len"] = original_token_len if stats["min_original_token_len"] is None else min(stats["min_original_token_len"], original_token_len)
            stats["max_original_token_len"] = original_token_len if stats["max_original_token_len"] is None else max(stats["max_original_token_len"], original_token_len)

            out_item = {
                "id": record_id,
                "text": tokens_to_text(
                    token_ids,
                    target_length=target_token_length,
                    pad_token_text=pad_token_text,
                ),
                "input_ids": input_ids,
                "meta": {
                    "source_folder": obj.get("source_folder"),
                    "read_id": obj.get("read_id"),
                    "signal_key": obj.get("signal_key"),
                    "raw_signal_len": raw_signal_len,
                    "processed_signal_len": processed_signal_len,
                    "original_token_len": original_token_len,
                    "target_token_length": target_token_length,
                    "pad_token_id": pad_token_id,
                    "pad_token_text": pad_token_text,
                    "codebook_vocab_offset": codebook_vocab_offset,
                    "padded_token_count": max(0, target_token_length - original_token_len),
                    "truncated": was_truncated,
                    "strategy": strategy,
                    "base_sample_spans_rel": obj.get("base_sample_spans_rel"),
                },
            }
            fout.write(json.dumps(out_item, ensure_ascii=False) + "\n")
            stats["written_records"] += 1

            if tqdm is not None:
                iterator.set_postfix(
                    {
                        "written": stats["written_records"],
                        "tok": original_token_len,
                        "pad": max(0, target_token_length - original_token_len),
                    }
                )

    elapsed = time.time() - start_time
    stats["elapsed_seconds"] = elapsed
    stats["records_per_second"] = stats["total_records"] / max(elapsed, 1e-6)
    stats["written_per_second"] = stats["written_records"] / max(elapsed, 1e-6)

    stats_path = output_jsonl_gz.with_suffix("")
    if stats_path.suffix == ".jsonl":
        stats_path = stats_path.with_suffix(".stats.json")
    else:
        stats_path = output_jsonl_gz.with_name(output_jsonl_gz.name + ".stats.json")

    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(f"Done: {input_jsonl.name}")
    print(f"Written records: {stats['written_records']}")
    print(f"Padded records: {stats['padded_records']}")
    print(f"Truncated records: {stats['truncated_records']}")
    print(f"Stats: {stats_path}")
    print(f"Elapsed: {elapsed / 60:.2f} min")
    print("=" * 80)
    return stats


def parse_splits(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Apple-preprocess full LB06/LB07 signal records, tokenize without signal chunking, "
            "and write fixed-length token jsonl.gz files."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-jsonl", type=str, help="Single input jsonl file.")
    input_group.add_argument("--split-dir", type=str, help="Directory containing train/validation/test jsonl files.")
    parser.add_argument("--output-jsonl-gz", type=str, help="Output path for --input-jsonl mode.")
    parser.add_argument("--output-dir", type=str, help="Output directory for --split-dir mode.")
    parser.add_argument("--splits", type=str, default="train,validation,test", help="Comma-separated split names.")
    parser.add_argument("--model-ckpt", type=str, default=DEFAULT_MODEL_CKPT, help="VQE tokenizer checkpoint path.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for tokenizer inference, for example cuda:0 or cpu.")
    parser.add_argument("--target-token-length", type=int, default=1600, help="Final number of tokens per record.")
    parser.add_argument("--pad-token-id", type=int, default=1, help="Vocabulary id used for right padding, usually <|pad|>=1.")
    parser.add_argument("--pad-token-text", type=str, default="<|pad|>", help="Text token used for right padding.")
    parser.add_argument("--codebook-vocab-offset", type=int, default=5, help="Offset that maps codebook ids to BERT vocabulary ids.")
    parser.add_argument("--strategy", type=str, default="apple", help="Signal preprocessing strategy from signal_prepare.py.")
    parser.add_argument("--spike-window-size", type=int, default=6000, help="Apple spike median-filter window size.")
    parser.add_argument("--no-count-total-first", action="store_true", help="Skip a first pass for tqdm total counting.")
    args = parser.parse_args()

    from vqe_tokenizer import VQETokenizer

    tokenizer = VQETokenizer(model_ckpt=args.model_ckpt, device=args.device)

    if args.input_jsonl:
        if not args.output_jsonl_gz:
            raise ValueError("--output-jsonl-gz is required when using --input-jsonl.")
        process_one_jsonl(
            input_jsonl=args.input_jsonl,
            output_jsonl_gz=args.output_jsonl_gz,
            tokenizer=tokenizer,
            target_token_length=args.target_token_length,
            pad_token_id=args.pad_token_id,
            pad_token_text=args.pad_token_text,
            codebook_vocab_offset=args.codebook_vocab_offset,
            strategy=args.strategy,
            spike_window_size=args.spike_window_size,
            count_total_first=not args.no_count_total_first,
        )
        return

    if not args.output_dir:
        raise ValueError("--output-dir is required when using --split-dir.")

    split_dir = Path(args.split_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    for split_name in parse_splits(args.splits):
        input_jsonl = split_dir / f"{split_name}.jsonl"
        if not input_jsonl.exists():
            print(f"Skip missing split: {input_jsonl}")
            continue

        output_jsonl_gz = output_dir / (
            f"{split_name}_fullapple_token{args.target_token_length}.jsonl.gz"
        )
        all_stats[split_name] = process_one_jsonl(
            input_jsonl=input_jsonl,
            output_jsonl_gz=output_jsonl_gz,
            tokenizer=tokenizer,
            target_token_length=args.target_token_length,
            pad_token_id=args.pad_token_id,
            pad_token_text=args.pad_token_text,
            codebook_vocab_offset=args.codebook_vocab_offset,
            strategy=args.strategy,
            spike_window_size=args.spike_window_size,
            count_total_first=not args.no_count_total_first,
        )

    summary_path = output_dir / f"process_stats_fullapple_token{args.target_token_length}.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(all_stats, handle, ensure_ascii=False, indent=2)
    print(f"All split stats saved to: {summary_path}")


if __name__ == "__main__":
    main()
