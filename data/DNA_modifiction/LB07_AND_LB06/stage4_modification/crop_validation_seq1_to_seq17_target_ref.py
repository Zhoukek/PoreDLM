#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from collections import Counter, defaultdict


# ============================================================
# 1. 输入输出路径
# ============================================================

INPUT_JSONL = Path(
    "/mnt/zzbnew/rnamodel/shenhaojie/PoreDLM/data/DNA_modifiction/"
    "LB07_AND_LB06/all-data/validation_signal_cropped.jsonl"
)

OUTPUT_DIR = Path(
    "/mnt/zzbnew/rnamodel/shenhaojie/PoreDLM/data/DNA_modifiction/"
    "LB07_AND_LB06/all-data"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSONL = OUTPUT_DIR / "validation_seq1_to_seq17_ref_target_cropped.jsonl"
OUTPUT_SUMMARY = OUTPUT_DIR / "validation_seq1_to_seq17_ref_target_cropped_summary.txt"


# ============================================================
# 2. seq_1 到 seq_17 对应的标准 ref 序列
#    来源：250F600084012.xlsx 中 va-se-C-1 到 va-se-C-17 的固定序列
# ============================================================

TARGET_REF_BY_LABEL = {
    "seq_1": "AATTTGTGAATAACTAGGTCAGCTAGCAGGCTCGATTGAGAAGTCCCTAATCTTACTAGATCTAGCGTAGCGGTAATATAAGCTATCACCACCTCGCCCATCAGAGCTCTCTCATATGCACACTAAGCTACGCTAGG",
    "seq_2": "AATTAAGCGTGACCCTTATGCTGAGGCTATTCCCCATTAGGCCCTCTACTTCAGAGGAGCACACACTTACCGGAACTGATTTTTTATTACTATGCGGTAGGCGAGCCACTGTGAGATTAGCTCTCCGCATAGTGTTC",
    "seq_3": "AATTCACACCCTCCTGGGGAAGGTTCCAGGTCCCACTTGCCGTACTATACCCACGTGAAGAGTCCGATGCCAACGTGCGAACTGGTGCCCTAACCAGACTGGTAGCCACCGGCTCTACTCGCGACTTCATATACATC",
    "seq_4": "AATTGGCAGTCCTCCTGCACGGGGAGAGATGCCTCAATACTGGCTCCGCGGCATGGAATTTATTTGGAAACCTGTCGGTTACTATAGTGCACACTACAAGTGTTTATACGAGTCGCGAGCAAGCAGACCAATAAATG",
    "seq_5": "AATTACGTAGATGCGGCACCATTAATTATCGACCTCGACACTAAATCAGGCCCAGGCGAGAGTGCCGCCACTAGTTAGCCGTACCGAATCATACTAGCACGGTCTGCCCGTTTTCGAAATTGTTATACACCTATAAG",
    "seq_6": "AATTTCCTGACGGCCTAGATTCGATTAACTGCCATCTCAGCGCCAATACTACATTCATGACGGTGATGGGCAAGCTAGGGCCTGAAGTACTGCCATGCTACTGATCGCCTGACGTTCTGGGGACTGCCCCTGCGGTT",
    "seq_7": "AATTCATATGTTCCTCTCAATCTTGTCTCTTCCGGAGGGTCCGCCCAGTCCCAGACTCTCAACACGATCTCGTAGTGCACCGATCGGGGCGGCCAGAACATTTGCGGTCTCTGATGCTCGTGCTGACCTGGGTTCCT",
    "seq_8": "AATTGCACTGAGTCTTTTTGCTAACGAACTCGCCCGTCAGCAGGCTTTAATCTAGGTGTATACGGCCCGTCAGATTAGTTCTTATAGAACAGGCACCACCCCACACCGCCGCATAATGTAGGGAGGTCCACAGAGTA",
    "seq_9": "AATTAATTCCCATCCCGTCAAATTATTGTTGCCGCCACAATGTAATTGTTTCCTTAGCGAGAGTCTTGTGCTCAATATCGGGGCACTCACCACCAGTATGGGGCCTGACGGGGTTCGACTTCGCGAACGCTTCAATC",
    "seq_10": "AATTAAAAGGGCGCTCTTCACAGATCCAATGACACCACGACTATGGAGCCTCTTCGGGTTGGCCACGCCGCAGATACGTCAGAACCCAACAAGCGTACGTTATAGCGGCGACTATGTACTGCGCGAGCTTGGGAAGG",
    "seq_11": "AATTCATCAGAAACCAGAGATTCGGGTCTACACAAAGCGATCAATCTTAGGCCGCTCAGGCCCATCAAAGCAGCTAAGGCTAGTAGGCTCCCCCGAGGTTGTCGGCTTCTCCTAAATATTCATAAGGCTTTCACACG",
    "seq_12": "AATTACCTTACACCCCGTACCTGAGGTTGCCTCCACTATAACTTCAAGGTCCTATGGGTCGTGGGGTCAGCTCTGACATCTTTCGCGCACGCCCGAGGCTATGAGCAGCCGATTCAGCACAGGGCAACCGTAACGTC",
    "seq_13": "AATTTACTCTATGCTTGGACAGGTAAATCGTTCGAACGATTCACTTCCCTACTCAAACTAGTCCCGGTGCCAGTTGGACGTTGCATCAGCGACGAGCGGATACTAGGACAATTTATGCATGCCCGACCAGGCGCGTT",
    "seq_14": "AATTACCGGACTCCGAGCCTGTCCCCACCGAACCCGCGATTGTTCAGAGTTCGCTAAGCCTACCCAGCGACGTGTAGTCCCAGGAATCTCTGCATGCACGTCATTCCACTCGGGGTGGTCCTTTGCGCGGAGACAAC",
    "seq_15": "AATTTATACCGAGCGCATTGTACCGAGCGTATCGATCAACGAGGCGTGAGACGCACCAAAGACACTTGGCCGGTCGGCGAGCATGTATACCGCACACCCAAGATCTAGCATACGAGATTATACATAACATATTAAAC",
    "seq_16": "AATTGCCCACGTCCTTCCAATCTAAGTCCCGGCTCGGCTCCCTTCCTCCATCGCGTGGTTTGTACTAGCACCTGCGACCTGATAATGTTCGGGTAGAAGTCGATTTATCTCTGTTTCCCTATTACGTCGGGGAGTCC",
    "seq_17": "AATTCTGGTCGCGCTATACGGCGACACCGACTCAACAGTGCAGACATAGTGCCTCGCCTCGGTTACACCACAGATCGCATTGCGACCTCCGTAACCACCACCTCAACGCCTGTAGGGGCATCGATTTCCCACTAGAC",
}


# ============================================================
# 3. 工具函数
# ============================================================

def clean_seq(x):
    if x is None:
        return ""
    return str(x).strip().upper()


def safe_int(x):
    try:
        return int(x)
    except Exception:
        return None


def parse_span_item(item):
    """
    解析 base_sample_span_ref 中一个碱基对应的 signal span。

    兼容：
      [start, end]
      [base_idx, start, end]
      [xxx, xxx, start, end]
      [None, None]
      {"start": x, "end": y}
      {"sample_start": x, "sample_end": y}
      {"signal_start": x, "signal_end": y}
      {"span": [start, end]}
    """
    if item is None:
        return None

    if isinstance(item, dict):
        key_pairs = [
            ("sample_start", "sample_end"),
            ("signal_start", "signal_end"),
            ("start", "end"),
            ("s_start", "s_end"),
            ("left", "right"),
        ]

        for k1, k2 in key_pairs:
            if k1 in item and k2 in item:
                s = safe_int(item[k1])
                e = safe_int(item[k2])
                if s is not None and e is not None:
                    return s, e

        if "span" in item and isinstance(item["span"], (list, tuple)) and len(item["span"]) >= 2:
            s = safe_int(item["span"][-2])
            e = safe_int(item["span"][-1])
            if s is not None and e is not None:
                return s, e

        return None

    if isinstance(item, (list, tuple)):
        if len(item) < 2:
            return None

        s = safe_int(item[-2])
        e = safe_int(item[-1])

        if s is None or e is None:
            return None

        return s, e

    return None


def normalize_span(s, e, signal_len):
    s = int(s)
    e = int(e)

    if e < s:
        s, e = e, s

    s = max(0, min(s, signal_len))
    e = max(0, min(e, signal_len))

    if e <= s:
        return None

    return s, e


def shift_span_item(item, crop_start):
    """
    将 span 从原始 signal 坐标转换为裁剪后 signal 的相对坐标。
    例如原始 [1200, 1215]，crop_start=1000，转换后 [200, 215]。
    """
    if item is None:
        return item

    if isinstance(item, list):
        new_item = list(item)
        if len(new_item) >= 2:
            s = safe_int(new_item[-2])
            e = safe_int(new_item[-1])
            if s is not None and e is not None:
                new_item[-2] = s - crop_start
                new_item[-1] = e - crop_start
        return new_item

    if isinstance(item, tuple):
        new_item = list(item)
        if len(new_item) >= 2:
            s = safe_int(new_item[-2])
            e = safe_int(new_item[-1])
            if s is not None and e is not None:
                new_item[-2] = s - crop_start
                new_item[-1] = e - crop_start
        return new_item

    if isinstance(item, dict):
        new_item = dict(item)

        key_pairs = [
            ("sample_start", "sample_end"),
            ("signal_start", "signal_end"),
            ("start", "end"),
            ("s_start", "s_end"),
            ("left", "right"),
        ]

        for k1, k2 in key_pairs:
            if k1 in new_item and k2 in new_item:
                s = safe_int(new_item[k1])
                e = safe_int(new_item[k2])
                if s is not None and e is not None:
                    new_item[k1] = s - crop_start
                    new_item[k2] = e - crop_start

        if "span" in new_item and isinstance(new_item["span"], list) and len(new_item["span"]) >= 2:
            s = safe_int(new_item["span"][-2])
            e = safe_int(new_item["span"][-1])
            if s is not None and e is not None:
                new_item["span"][-2] = s - crop_start
                new_item["span"][-1] = e - crop_start

        return new_item

    return item


# ============================================================
# 4. 单条记录裁剪逻辑
# ============================================================

def crop_one_record(obj, line_idx):
    label = str(obj.get("label", ""))

    if label not in TARGET_REF_BY_LABEL:
        return None, "label_not_in_seq_1_to_seq_17"

    target_ref = TARGET_REF_BY_LABEL[label]

    ref = clean_seq(obj.get("ref", ""))
    if len(ref) == 0:
        return None, "missing_or_empty_ref"

    ref_start = ref.find(target_ref)
    if ref_start < 0:
        return None, f"target_ref_not_found_for_{label}"

    ref_end = ref_start + len(target_ref)

    base_sample_span_ref = obj.get("base_sample_span_ref", None)
    if not isinstance(base_sample_span_ref, list) or len(base_sample_span_ref) == 0:
        return None, "missing_or_empty_base_sample_span_ref"

    if ref_end > len(base_sample_span_ref):
        return None, (
            f"target_ref_range_out_of_base_sample_span_ref_"
            f"{label}_ref_range_{ref_start}_{ref_end}_span_len_{len(base_sample_span_ref)}"
        )

    signal = obj.get("signal", None)
    if not isinstance(signal, list) or len(signal) == 0:
        return None, "missing_or_empty_signal"

    selected_spans = base_sample_span_ref[ref_start:ref_end]

    valid_spans = []
    invalid_span_count = 0

    for item in selected_spans:
        parsed = parse_span_item(item)
        if parsed is None:
            invalid_span_count += 1
            continue

        norm = normalize_span(parsed[0], parsed[1], len(signal))
        if norm is None:
            invalid_span_count += 1
            continue

        valid_spans.append(norm)

    if len(valid_spans) == 0:
        return None, f"no_valid_span_in_target_ref_for_{label}"

    signal_start = min(s for s, e in valid_spans)
    signal_end = max(e for s, e in valid_spans)

    if signal_end <= signal_start:
        return None, "invalid_signal_crop_range"

    cropped_signal = signal[signal_start:signal_end]
    if len(cropped_signal) == 0:
        return None, "empty_cropped_signal"

    cropped_spans = [
        shift_span_item(item, signal_start)
        for item in selected_spans
    ]

    new_obj = dict(obj)

    # 核心字段更新
    new_obj["ref"] = target_ref
    new_obj["base_sample_span_ref"] = cropped_spans
    new_obj["signal"] = cropped_signal

    # 加入裁剪信息，方便后续检查
    new_obj["crop_info"] = {
        "source_line_idx": line_idx,
        "label": label,
        "target_ref_len": len(target_ref),
        "original_ref_len": len(ref),
        "ref_crop_start_0based": ref_start,
        "ref_crop_end_0based_exclusive": ref_end,
        "original_base_sample_span_ref_len": len(base_sample_span_ref),
        "cropped_base_sample_span_ref_len": len(cropped_spans),
        "original_signal_len": len(signal),
        "signal_crop_start_original": signal_start,
        "signal_crop_end_original_exclusive": signal_end,
        "cropped_signal_len": len(cropped_signal),
        "valid_span_count_in_target_ref": len(valid_spans),
        "invalid_span_count_in_target_ref": invalid_span_count,
    }

    return new_obj, "ok"


# ============================================================
# 5. 主程序
# ============================================================

def main():
    if not INPUT_JSONL.exists():
        raise FileNotFoundError(f"输入文件不存在: {INPUT_JSONL}")

    total_lines = 0
    valid_json = 0
    saved = 0
    skipped = 0

    reason_counter = Counter()
    label_counter_input = Counter()
    label_counter_saved = Counter()
    cropped_signal_lens_by_label = defaultdict(list)
    ref_start_counter_by_label = defaultdict(Counter)

    print("=" * 100)
    print("提取 seq_1 到 seq_17 对应标准 ref，并裁剪 ref / base_sample_span_ref / signal")
    print("=" * 100)
    print(f"输入文件: {INPUT_JSONL}")
    print(f"输出文件: {OUTPUT_JSONL}")
    print(f"summary: {OUTPUT_SUMMARY}")
    print(f"目标 label 数量: {len(TARGET_REF_BY_LABEL)}")
    print("=" * 100)

    with INPUT_JSONL.open("r", encoding="utf-8") as fin, \
            OUTPUT_JSONL.open("w", encoding="utf-8") as fout:

        for line_idx, line in enumerate(fin, start=1):
            line = line.strip()

            if not line:
                reason_counter["empty_line"] += 1
                continue

            total_lines += 1

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                reason_counter["json_decode_error"] += 1
                continue

            valid_json += 1

            label = str(obj.get("label", "missing_label"))
            label_counter_input[label] += 1

            new_obj, reason = crop_one_record(obj, line_idx)

            if reason == "ok":
                fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")
                saved += 1

                saved_label = new_obj["label"]
                label_counter_saved[saved_label] += 1

                cropped_signal_lens_by_label[saved_label].append(
                    new_obj["crop_info"]["cropped_signal_len"]
                )
                ref_start_counter_by_label[saved_label][
                    new_obj["crop_info"]["ref_crop_start_0based"]
                ] += 1

            else:
                skipped += 1
                reason_counter[reason] += 1

    # ========================================================
    # 6. summary
    # ========================================================

    summary_lines = [
        "=" * 100,
        "crop validation seq_1 to seq_17 target ref summary",
        "=" * 100,
        f"INPUT_JSONL: {INPUT_JSONL}",
        f"OUTPUT_JSONL: {OUTPUT_JSONL}",
        f"target labels: seq_1 ~ seq_17",
        "",
        f"total non-empty lines: {total_lines}",
        f"valid json lines: {valid_json}",
        f"saved records: {saved}",
        f"skipped records: {skipped}",
        f"saved ratio over valid json: {saved / valid_json if valid_json else 0:.6f}",
        "",
        "target ref length by label:",
    ]

    for i in range(1, 18):
        lab = f"seq_{i}"
        summary_lines.append(f"  {lab}: {len(TARGET_REF_BY_LABEL[lab])}")

    summary_lines.extend([
        "",
        "input label counts:",
    ])

    for i in range(1, 18):
        lab = f"seq_{i}"
        summary_lines.append(f"  {lab}: {label_counter_input.get(lab, 0)}")

    summary_lines.extend([
        "",
        "saved label counts and cropped signal length:",
    ])

    for i in range(1, 18):
        lab = f"seq_{i}"
        lens = cropped_signal_lens_by_label.get(lab, [])

        if lens:
            mean_len = sum(lens) / len(lens)
            min_len = min(lens)
            max_len = max(lens)
        else:
            mean_len = 0
            min_len = 0
            max_len = 0

        summary_lines.append(
            f"  {lab}: saved={label_counter_saved.get(lab, 0)}, "
            f"signal_len_mean={mean_len:.3f}, min={min_len}, max={max_len}"
        )

    summary_lines.extend([
        "",
        "ref crop start distribution by label:",
    ])

    for i in range(1, 18):
        lab = f"seq_{i}"
        counter = ref_start_counter_by_label.get(lab, Counter())
        if len(counter) == 0:
            summary_lines.append(f"  {lab}: no saved records")
        else:
            starts = ", ".join([f"{k}:{v}" for k, v in counter.most_common()])
            summary_lines.append(f"  {lab}: {starts}")

    summary_lines.extend([
        "",
        "skip reasons:",
    ])

    for reason, count in reason_counter.most_common():
        summary_lines.append(f"  {reason}: {count}")

    summary_lines.append("=" * 100)

    summary_text = "\n".join(summary_lines) + "\n"

    with OUTPUT_SUMMARY.open("w", encoding="utf-8") as f:
        f.write(summary_text)

    print(summary_text)


if __name__ == "__main__":
    main()