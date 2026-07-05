from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

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
    "stage1_tokenizer/runs/LB07_AND_LB06_model_type_1_cnn_type_0_distill_0.1_8k_vq_apple/"
    "models/porepgt_vqe_tokenizer.final.pth"
)
DEFAULT_MODIFIED_BASE_POSITIONS = "14,33,52,71,90,109,128"


def open_text(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def count_jsonl_records(path: Path) -> int:
    with open_text(path, "rt") as fin:
        return sum(1 for line in fin if line.strip())


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with open_text(path, "rt") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            yield line_no, obj


def parse_maybe_list(value: Any) -> Any:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def parse_modified_positions(value: str) -> list[int]:
    positions = [int(part.strip()) for part in value.split(",") if part.strip()]
    if any(pos <= 0 for pos in positions):
        raise ValueError(f"modified base positions are 1-based and must be positive: {positions}")
    return positions


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


def get_record_id(obj: dict[str, Any], fallback_index: int) -> str:
    for key in ("read_id", "id", "signal_key"):
        value = obj.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return f"record_{fallback_index:08d}"


def normalize_base_spans(value: Any, *, ref_len: int, line_no: int) -> list[list[int | None]]:
    spans = parse_maybe_list(value)
    if not isinstance(spans, list):
        raise ValueError(f"line {line_no}: base_sample_span_ref must be a list")
    if len(spans) != ref_len:
        raise ValueError(
            f"line {line_no}: len(base_sample_span_ref)={len(spans)} does not match len(ref)={ref_len}"
        )

    normalized: list[list[int | None]] = []
    for index, span in enumerate(spans):
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            raise ValueError(f"line {line_no}: base_sample_span_ref[{index}] must be [start, end]")
        start, end = span
        if start is None and end is None:
            normalized.append([None, None])
            continue
        if start is None or end is None:
            raise ValueError(f"line {line_no}: base_sample_span_ref[{index}] has only one null side: {span}")
        start = int(start)
        end = int(end)
        if end <= start:
            raise ValueError(f"line {line_no}: invalid base_sample_span_ref[{index}] range: {span}")
        normalized.append([start, end])
    return normalized


def sample_span_to_token_span(
    start_sample: int,
    end_sample: int,
    *,
    samples_per_token: int,
    token_count: int,
) -> tuple[int, int]:
    token_start = max(0, int(start_sample) // samples_per_token)
    token_end = min(token_count, int(math.ceil(float(end_sample) / samples_per_token)))
    token_end = max(token_start, token_end)
    return token_start, token_end


def build_c_modification_labels(
    *,
    ref: str,
    base_sample_span_ref: list[list[int | None]],
    modified_base_positions_1based: list[int],
    samples_per_token: int,
    valid_token_count: int,
    target_token_length: int,
    pad_label_id: int,
) -> tuple[list[int], dict[str, Any]]:
    valid_labels = [0] * valid_token_count
    modified_base_set = set(modified_base_positions_1based)
    c_token_spans: list[list[int | str]] = []
    modified_c_token_spans: list[list[int | str]] = []
    null_c_base_positions: list[int] = []
    modified_positions_not_c: list[int] = []

    for base_index, base in enumerate(ref):
        base_position = base_index + 1
        is_c = base.upper() == "C"
        is_modified = base_position in modified_base_set
        if is_modified and not is_c:
            modified_positions_not_c.append(base_position)
        if not is_c:
            continue

        span = base_sample_span_ref[base_index]
        start_sample, end_sample = span
        if start_sample is None or end_sample is None:
            null_c_base_positions.append(base_position)
            continue

        token_start, token_end = sample_span_to_token_span(
            int(start_sample),
            int(end_sample),
            samples_per_token=samples_per_token,
            token_count=valid_token_count,
        )
        if token_end <= token_start:
            continue

        label_value = 2 if is_modified else 1
        for token_index in range(token_start, token_end):
            valid_labels[token_index] = max(valid_labels[token_index], label_value)

        span_item: list[int | str] = [base_position, int(start_sample), int(end_sample), token_start, token_end]
        if is_modified:
            modified_c_token_spans.append(span_item)
        else:
            c_token_spans.append(span_item)

    labels = valid_labels[:target_token_length]
    if len(labels) < target_token_length:
        labels.extend([int(pad_label_id)] * (target_token_length - len(labels)))

    summary = {
        "valid_token_label_len": valid_token_count,
        "label_0_count": int(sum(1 for value in valid_labels if value == 0)),
        "label_1_count": int(sum(1 for value in valid_labels if value == 1)),
        "label_2_count": int(sum(1 for value in valid_labels if value == 2)),
        "c_base_count": int(sum(1 for base in ref if base.upper() == "C")),
        "c_token_spans": c_token_spans,
        "modified_c_token_spans": modified_c_token_spans,
        "null_c_base_positions_1based": null_c_base_positions,
        "modified_positions_not_c_1based": modified_positions_not_c,
    }
    return labels, summary


def build_meta(
    obj: dict[str, Any],
    *,
    signal_len: int,
    original_token_len: int,
    target_token_length: int,
    pad_token_id: int,
    pad_token_text: str,
    codebook_vocab_offset: int,
    was_truncated: bool,
    min_signal_len: int,
    max_signal_len: int,
    samples_per_token: int,
    pad_label_id: int,
    modified_base_positions_1based: list[int],
    label_summary: dict[str, Any],
) -> dict[str, Any]:
    meta = {
        "read_id": obj.get("read_id"),
        "id": obj.get("id"),
        "signal_key": obj.get("signal_key"),
        "label": obj.get("label"),
        "seq": obj.get("seq"),
        "ref": obj.get("ref"),
        "align": obj.get("align"),
        "mv_stride": obj.get("mv_stride"),
        "kmer": obj.get("kmer"),
        "normal_mode": obj.get("normal_mode"),
        "signal_len": signal_len,
        "signal_crop_start": obj.get("signal_crop_start"),
        "signal_crop_end": obj.get("signal_crop_end"),
        "base_sample_span_ref": obj.get("base_sample_span_ref"),
        "base_sample_span_seq": obj.get("base_sample_span_seq"),
        "crop_info": obj.get("crop_info"),
        "original_token_len": original_token_len,
        "target_token_length": target_token_length,
        "pad_token_id": pad_token_id,
        "pad_token_text": pad_token_text,
        "codebook_vocab_offset": codebook_vocab_offset,
        "padded_token_count": max(0, target_token_length - original_token_len),
        "truncated": was_truncated,
        "min_signal_len": min_signal_len,
        "max_signal_len": max_signal_len,
        "signal_preprocessed": True,
        "samples_per_token": samples_per_token,
        "pad_label_id": pad_label_id,
        "modification_base_positions_1based": modified_base_positions_1based,
        "c_modification_label_schema": "0=non-C token, 1=unmodified-C token, 2=modified-C token, pad=-100 by default",
    }
    meta.update(label_summary)
    return {key: value for key, value in meta.items() if value is not None}


def process_one_jsonl(
    input_jsonl: str | Path,
    output_jsonl_gz: str | Path,
    tokenizer,
    *,
    target_token_length: int = 2000,
    min_signal_len: int = 0,
    max_signal_len: int = 1000000000,
    samples_per_token: int = 5,
    modified_base_positions_1based: list[int],
    pad_label_id: int = -100,
    pad_token_id: int = 1,
    pad_token_text: str = "<|pad|>",
    codebook_vocab_offset: int = 5,
    count_total_first: bool = True,
) -> dict[str, Any]:
    import numpy as np

    input_jsonl = Path(input_jsonl)
    output_jsonl_gz = Path(output_jsonl_gz)
    output_jsonl_gz.parent.mkdir(parents=True, exist_ok=True)

    if min_signal_len < 0 or max_signal_len < min_signal_len:
        raise ValueError(
            f"invalid signal length range: min_signal_len={min_signal_len}, max_signal_len={max_signal_len}"
        )
    if target_token_length <= 0:
        raise ValueError(f"target_token_length must be positive, got {target_token_length}")
    if samples_per_token <= 0:
        raise ValueError(f"samples_per_token must be positive, got {samples_per_token}")

    print("=" * 80)
    print(f"Input jsonl: {input_jsonl}")
    print(f"Output jsonl.gz: {output_jsonl_gz}")
    print("Process: preprocessed signal -> VQE tokenizer -> C/modification token labels")
    print(f"signal length range: [{min_signal_len}, {max_signal_len}]")
    print(f"target_token_length: {target_token_length}")
    print(f"samples_per_token: {samples_per_token}")
    print(f"modified_base_positions_1based: {modified_base_positions_1based}")
    print(f"pad_label_id: {pad_label_id}")
    print(f"pad_token_id: {pad_token_id} ({pad_token_text})")
    print(f"codebook_vocab_offset: {codebook_vocab_offset}")
    print("=" * 80)

    total_records = count_jsonl_records(input_jsonl) if count_total_first else None
    if total_records is not None:
        print(f"Input records: {total_records}")

    stats: dict[str, Any] = {
        "input_jsonl": str(input_jsonl),
        "output_jsonl_gz": str(output_jsonl_gz),
        "process_order": "preprocessed_signal_tokenize_then_c_modification_labels",
        "target_token_length": target_token_length,
        "min_signal_len": min_signal_len,
        "max_signal_len": max_signal_len,
        "samples_per_token": samples_per_token,
        "modified_base_positions_1based": modified_base_positions_1based,
        "pad_label_id": pad_label_id,
        "pad_token_id": pad_token_id,
        "pad_token_text": pad_token_text,
        "codebook_vocab_offset": codebook_vocab_offset,
        "total_records": 0,
        "written_records": 0,
        "invalid_signal_records": 0,
        "invalid_ref_or_span_records": 0,
        "too_short_signal_records": 0,
        "too_long_signal_records": 0,
        "tokenizer_empty_records": 0,
        "padded_records": 0,
        "truncated_records": 0,
        "records_with_null_c_bases": 0,
        "records_with_modified_positions_not_c": 0,
        "total_label_1_tokens": 0,
        "total_label_2_tokens": 0,
        "min_kept_signal_len": None,
        "max_kept_signal_len": None,
        "min_original_token_len": None,
        "max_original_token_len": None,
    }

    start_time = time.time()
    iterator = iter_jsonl(input_jsonl)
    if tqdm is not None:
        iterator = tqdm(iterator, total=total_records, desc=f"Tokenizing {input_jsonl.name}", ncols=120)

    with open_text(output_jsonl_gz, "wt") as fout:
        for record_index, (line_no, obj) in enumerate(iterator):
            stats["total_records"] += 1
            record_id = get_record_id(obj, record_index)

            ref = obj.get("ref")
            if not isinstance(ref, str) or not ref:
                stats["invalid_ref_or_span_records"] += 1
                continue
            try:
                base_sample_span_ref = normalize_base_spans(
                    obj.get("base_sample_span_ref"),
                    ref_len=len(ref),
                    line_no=line_no,
                )
            except ValueError:
                stats["invalid_ref_or_span_records"] += 1
                continue

            signal = parse_maybe_list(obj.get("signal"))
            if not isinstance(signal, list):
                stats["invalid_signal_records"] += 1
                continue

            signal = np.asarray(signal, dtype=np.float32)
            if signal.ndim != 1:
                stats["invalid_signal_records"] += 1
                continue

            signal_len = int(signal.size)
            if signal_len < min_signal_len:
                stats["too_short_signal_records"] += 1
                continue
            if signal_len > max_signal_len:
                stats["too_long_signal_records"] += 1
                continue

            stats["min_kept_signal_len"] = (
                signal_len if stats["min_kept_signal_len"] is None else min(stats["min_kept_signal_len"], signal_len)
            )
            stats["max_kept_signal_len"] = (
                signal_len if stats["max_kept_signal_len"] is None else max(stats["max_kept_signal_len"], signal_len)
            )

            token_ids = tokenizer._tokenize_chunked_signal(signal)
            if token_ids.size == 0:
                stats["tokenizer_empty_records"] += 1
                continue

            input_ids, original_token_len, was_truncated = tokens_to_vocab_input_ids(
                token_ids=token_ids,
                target_length=target_token_length,
                codebook_vocab_offset=codebook_vocab_offset,
                pad_token_id=pad_token_id,
            )
            valid_token_count = min(original_token_len, target_token_length)

            c_modification_label, label_summary = build_c_modification_labels(
                ref=ref,
                base_sample_span_ref=base_sample_span_ref,
                modified_base_positions_1based=modified_base_positions_1based,
                samples_per_token=samples_per_token,
                valid_token_count=valid_token_count,
                target_token_length=target_token_length,
                pad_label_id=pad_label_id,
            )

            if original_token_len < target_token_length:
                stats["padded_records"] += 1
            if was_truncated:
                stats["truncated_records"] += 1
            if label_summary["null_c_base_positions_1based"]:
                stats["records_with_null_c_bases"] += 1
            if label_summary["modified_positions_not_c_1based"]:
                stats["records_with_modified_positions_not_c"] += 1

            stats["total_label_1_tokens"] += int(label_summary["label_1_count"])
            stats["total_label_2_tokens"] += int(label_summary["label_2_count"])
            stats["min_original_token_len"] = (
                original_token_len
                if stats["min_original_token_len"] is None
                else min(stats["min_original_token_len"], original_token_len)
            )
            stats["max_original_token_len"] = (
                original_token_len
                if stats["max_original_token_len"] is None
                else max(stats["max_original_token_len"], original_token_len)
            )

            out_item = {
                "id": record_id,
                "text": tokens_to_text(
                    token_ids,
                    target_length=target_token_length,
                    pad_token_text=pad_token_text,
                ),
                "input_ids": input_ids,
                "c_modification_label": c_modification_label,
                "meta": build_meta(
                    obj,
                    signal_len=signal_len,
                    original_token_len=original_token_len,
                    target_token_length=target_token_length,
                    pad_token_id=pad_token_id,
                    pad_token_text=pad_token_text,
                    codebook_vocab_offset=codebook_vocab_offset,
                    was_truncated=was_truncated,
                    min_signal_len=min_signal_len,
                    max_signal_len=max_signal_len,
                    samples_per_token=samples_per_token,
                    pad_label_id=pad_label_id,
                    modified_base_positions_1based=modified_base_positions_1based,
                    label_summary=label_summary,
                ),
            }
            fout.write(json.dumps(out_item, ensure_ascii=False, separators=(",", ":")) + "\n")
            stats["written_records"] += 1

            if tqdm is not None:
                iterator.set_postfix(
                    {
                        "written": stats["written_records"],
                        "sig": signal_len,
                        "tok": original_token_len,
                        "C": label_summary["label_1_count"],
                        "modC": label_summary["label_2_count"],
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

    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("=" * 80)
    print(f"Done: {input_jsonl.name}")
    print(f"Written records: {stats['written_records']}")
    print(f"Label 1 tokens (unmodified C): {stats['total_label_1_tokens']}")
    print(f"Label 2 tokens (modified C): {stats['total_label_2_tokens']}")
    print(f"Padded records: {stats['padded_records']}")
    print(f"Truncated records: {stats['truncated_records']}")
    print(f"Stats: {stats_path}")
    print(f"Elapsed: {elapsed / 60:.2f} min")
    print("=" * 80)
    return stats


def parse_splits(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Tokenize already-apple-preprocessed signal records and create token-level labels "
            "for all ref C bases: 0=non-C, 1=unmodified C, 2=modified C."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-jsonl", type=str, help="Single input jsonl/jsonl.gz file.")
    input_group.add_argument("--split-dir", type=str, help="Directory containing train/validation/test jsonl files.")
    parser.add_argument("--output-jsonl-gz", type=str, help="Output path for --input-jsonl mode.")
    parser.add_argument("--output-dir", type=str, help="Output directory for --split-dir mode.")
    parser.add_argument("--splits", type=str, default="train,validation,test", help="Comma-separated split names.")
    parser.add_argument("--model-ckpt", type=str, default=DEFAULT_MODEL_CKPT, help="VQE tokenizer checkpoint path.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for tokenizer inference, for example cuda:0 or cpu.")
    parser.add_argument("--target-token-length", type=int, default=2000, help="Final number of tokens per record.")
    parser.add_argument("--min-signal-len", type=int, default=0, help="Minimum signal length kept before tokenization.")
    parser.add_argument("--max-signal-len", type=int, default=1000000000, help="Maximum signal length kept before tokenization.")
    parser.add_argument("--samples-per-token", type=int, default=5, help="Signal samples represented by one tokenizer token.")
    parser.add_argument("--modified-base-positions", type=str, default=DEFAULT_MODIFIED_BASE_POSITIONS)
    parser.add_argument("--pad-label-id", type=int, default=-100, help="Label id used for padded token positions.")
    parser.add_argument("--pad-token-id", type=int, default=1, help="Vocabulary id used for right padding, usually <|pad|>=1.")
    parser.add_argument("--pad-token-text", type=str, default="<|pad|>", help="Text token used for right padding.")
    parser.add_argument("--codebook-vocab-offset", type=int, default=5, help="Offset that maps codebook ids to BERT vocabulary ids.")
    parser.add_argument("--no-count-total-first", action="store_true", help="Skip a first pass for tqdm total counting.")
    args = parser.parse_args()

    from vqe_tokenizer import VQETokenizer

    modified_base_positions = parse_modified_positions(args.modified_base_positions)
    tokenizer = VQETokenizer(model_ckpt=args.model_ckpt, device=args.device)

    if args.input_jsonl:
        if not args.output_jsonl_gz:
            raise ValueError("--output-jsonl-gz is required when using --input-jsonl.")
        process_one_jsonl(
            input_jsonl=args.input_jsonl,
            output_jsonl_gz=args.output_jsonl_gz,
            tokenizer=tokenizer,
            target_token_length=args.target_token_length,
            min_signal_len=args.min_signal_len,
            max_signal_len=args.max_signal_len,
            samples_per_token=args.samples_per_token,
            modified_base_positions_1based=modified_base_positions,
            pad_label_id=args.pad_label_id,
            pad_token_id=args.pad_token_id,
            pad_token_text=args.pad_token_text,
            codebook_vocab_offset=args.codebook_vocab_offset,
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
            input_jsonl = split_dir / f"{split_name}.jsonl.gz"
        if not input_jsonl.exists():
            print(f"Skip missing split: {split_dir / f'{split_name}.jsonl'}")
            continue

        output_jsonl_gz = output_dir / f"{split_name}_token_c_modlabel.jsonl.gz"
        all_stats[split_name] = process_one_jsonl(
            input_jsonl=input_jsonl,
            output_jsonl_gz=output_jsonl_gz,
            tokenizer=tokenizer,
            target_token_length=args.target_token_length,
            min_signal_len=args.min_signal_len,
            max_signal_len=args.max_signal_len,
            samples_per_token=args.samples_per_token,
            modified_base_positions_1based=modified_base_positions,
            pad_label_id=args.pad_label_id,
            pad_token_id=args.pad_token_id,
            pad_token_text=args.pad_token_text,
            codebook_vocab_offset=args.codebook_vocab_offset,
            count_total_first=not args.no_count_total_first,
        )

    summary_path = output_dir / "process_stats_token_c_modlabel.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(all_stats, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"All split stats saved to: {summary_path}")


if __name__ == "__main__":
    main()
