#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from pathlib import Path
from collections import Counter, defaultdict


# ============================================================
# 1. 输入文件
# ============================================================

INPUT_FILES = [
    # Path("/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/LB07_AND_LB06/all_data/LB06_250F600084012_0_1_0_0_seq_1_to_seq_17_merge_adjust.jsonl"),
    # Path("/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/LB07_AND_LB06/all_data/LB06_250F600084012_2_1_0_0_seq_1_to_seq_17_merge_adjust.jsonl"),
    Path("/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/LB07_AND_LB06/all_data/LB07-250F600084012_0_1_0_0_seq_1_to_seq_17_merge_adjust.jsonl"),
    Path("/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/LB07_AND_LB06/all_data/LB07-250F600084012_2_1_0_0_seq_1_to_seq_17_merge_adjust.jsonl"),
]


# ============================================================
# 2. 输出路径
# ============================================================

OUTPUT_DIR = Path(
    "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/LB07_AND_LB06/all_data/split_LB07_only"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_JSONL = OUTPUT_DIR / "train.jsonl"
TEST_JSONL = OUTPUT_DIR / "test.jsonl"
VAL_JSONL = OUTPUT_DIR / "val.jsonl"
SUMMARY_TXT = OUTPUT_DIR / "split_train_test_val_summary.txt"


# ============================================================
# 3. 划分配置
# ============================================================

TRAIN_RATIO = 0.8
TEST_RATIO = 0.1
VAL_RATIO = 0.1

RANDOM_SEED = 42

# 如果输出文件已经存在，是否覆盖
OVERWRITE = True


# ============================================================
# 4. 工具函数
# ============================================================

def get_group_from_filename(path: Path):
    name = path.name

    if name.startswith("LB06") or name.startswith("LBO6"):
        return "LB06"

    if name.startswith("LB07") or name.startswith("LBO7"):
        return "LB07"

    raise ValueError(f"无法从文件名判断 LB06/LB07: {path}")


def check_input_files():
    for p in INPUT_FILES:
        if not p.exists():
            raise FileNotFoundError(f"输入文件不存在: {p}")

    if not OVERWRITE:
        for p in [TRAIN_JSONL, TEST_JSONL, VAL_JSONL, SUMMARY_TXT]:
            if p.exists():
                raise FileExistsError(f"输出文件已存在，且 OVERWRITE=False: {p}")


def scan_jsonl_offsets(input_files):
    """
    不把整行内容读入内存，只记录每条非空 jsonl 的：
        source_file_idx
        offset
        byte_len
        group
        source_name
    """

    records = []
    source_line_counts = Counter()
    group_counts = Counter()

    for file_idx, path in enumerate(input_files):
        group = get_group_from_filename(path)
        source_name = path.name

        with path.open("rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()

                if not line:
                    break

                if len(line.strip()) == 0:
                    continue

                byte_len = len(line)

                records.append({
                    "file_idx": file_idx,
                    "offset": offset,
                    "byte_len": byte_len,
                    "group": group,
                    "source_name": source_name,
                })

                source_line_counts[source_name] += 1
                group_counts[group] += 1

    return records, source_line_counts, group_counts


def copy_records_to_jsonl(records, output_path, input_files):
    """
    根据 byte offset 复制原始 jsonl 行。
    """

    file_handles = {}

    try:
        with output_path.open("wb") as fout:
            for rec in records:
                file_idx = rec["file_idx"]

                if file_idx not in file_handles:
                    file_handles[file_idx] = input_files[file_idx].open("rb")

                fin = file_handles[file_idx]
                fin.seek(rec["offset"])
                line = fin.read(rec["byte_len"])

                fout.write(line)

                if not line.endswith(b"\n"):
                    fout.write(b"\n")

    finally:
        for fh in file_handles.values():
            fh.close()


def count_by_source_and_group(records):
    source_counter = Counter()
    group_counter = Counter()

    for r in records:
        source_counter[r["source_name"]] += 1
        group_counter[r["group"]] += 1

    return source_counter, group_counter


# ============================================================
# 5. 主程序
# ============================================================

def main():
    random.seed(RANDOM_SEED)

    check_input_files()

    print("=" * 100)
    print("读取 4 个 jsonl，并划分 train/test/val")
    print("=" * 100)
    for p in INPUT_FILES:
        print(f"输入: {p}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"比例: train:test:val = {TRAIN_RATIO}:{TEST_RATIO}:{VAL_RATIO}")
    print("约束: val 全部来自 LB06")
    print("=" * 100)

    records, source_line_counts, group_counts = scan_jsonl_offsets(INPUT_FILES)

    total_n = len(records)

    if total_n == 0:
        raise RuntimeError("没有扫描到任何有效 jsonl 记录。")

    lb06_records = [r for r in records if r["group"] == "LB06"]
    lb07_records = [r for r in records if r["group"] == "LB07"]

    if len(lb06_records) == 0:
        raise RuntimeError("没有 LB06 记录，无法保证 val 全部来自 LB06。")

    # 目标数量
    val_n_target = int(round(total_n * VAL_RATIO))
    test_n_target = int(round(total_n * TEST_RATIO))

    # val 全部来自 LB06
    val_n = min(val_n_target, len(lb06_records))

    if val_n < val_n_target:
        print(
            f"[Warning] LB06 记录数不足，val_n 从目标 {val_n_target} "
            f"调整为 {val_n}"
        )

    # 从 LB06 中随机抽 val
    lb06_indices = list(range(len(lb06_records)))
    random.shuffle(lb06_indices)

    val_local_indices = set(lb06_indices[:val_n])
    val_records = [lb06_records[i] for i in val_local_indices]

    # 剩余记录进入 train/test 候选
    val_identity = {
        (r["file_idx"], r["offset"], r["byte_len"])
        for r in val_records
    }

    remaining_records = [
        r for r in records
        if (r["file_idx"], r["offset"], r["byte_len"]) not in val_identity
    ]

    random.shuffle(remaining_records)

    # test 按总数 10% 取，剩余给 train
    test_n = min(test_n_target, len(remaining_records))
    test_records = remaining_records[:test_n]
    train_records = remaining_records[test_n:]

    # 打乱各 split 内部顺序
    random.shuffle(train_records)
    random.shuffle(test_records)
    random.shuffle(val_records)

    # 写文件
    copy_records_to_jsonl(train_records, TRAIN_JSONL, INPUT_FILES)
    copy_records_to_jsonl(test_records, TEST_JSONL, INPUT_FILES)
    copy_records_to_jsonl(val_records, VAL_JSONL, INPUT_FILES)

    train_source_counter, train_group_counter = count_by_source_and_group(train_records)
    test_source_counter, test_group_counter = count_by_source_and_group(test_records)
    val_source_counter, val_group_counter = count_by_source_and_group(val_records)

    # summary
    summary_lines = []
    summary_lines.append("=" * 100)
    summary_lines.append("train/test/val split summary")
    summary_lines.append("=" * 100)
    summary_lines.append(f"RANDOM_SEED: {RANDOM_SEED}")
    summary_lines.append(f"TRAIN_RATIO: {TRAIN_RATIO}")
    summary_lines.append(f"TEST_RATIO: {TEST_RATIO}")
    summary_lines.append(f"VAL_RATIO: {VAL_RATIO}")
    summary_lines.append("Constraint: val records are all from LB06")
    summary_lines.append("")
    summary_lines.append("Input files:")
    for p in INPUT_FILES:
        summary_lines.append(f"  - {p}")
    summary_lines.append("")
    summary_lines.append("Input source line counts:")
    for k, v in source_line_counts.items():
        summary_lines.append(f"  {k}: {v}")
    summary_lines.append("")
    summary_lines.append("Input group counts:")
    for k, v in group_counts.items():
        summary_lines.append(f"  {k}: {v}")
    summary_lines.append("")
    summary_lines.append(f"Total records: {total_n}")
    summary_lines.append("")
    summary_lines.append("Split counts:")
    summary_lines.append(f"  train: {len(train_records)} ({len(train_records) / total_n:.6f})")
    summary_lines.append(f"  test : {len(test_records)} ({len(test_records) / total_n:.6f})")
    summary_lines.append(f"  val  : {len(val_records)} ({len(val_records) / total_n:.6f})")
    summary_lines.append("")
    summary_lines.append("Train group counts:")
    for k, v in train_group_counter.items():
        summary_lines.append(f"  {k}: {v}")
    summary_lines.append("")
    summary_lines.append("Test group counts:")
    for k, v in test_group_counter.items():
        summary_lines.append(f"  {k}: {v}")
    summary_lines.append("")
    summary_lines.append("Val group counts:")
    for k, v in val_group_counter.items():
        summary_lines.append(f"  {k}: {v}")
    summary_lines.append("")
    summary_lines.append("Train source counts:")
    for k, v in train_source_counter.items():
        summary_lines.append(f"  {k}: {v}")
    summary_lines.append("")
    summary_lines.append("Test source counts:")
    for k, v in test_source_counter.items():
        summary_lines.append(f"  {k}: {v}")
    summary_lines.append("")
    summary_lines.append("Val source counts:")
    for k, v in val_source_counter.items():
        summary_lines.append(f"  {k}: {v}")
    summary_lines.append("")
    summary_lines.append("Output files:")
    summary_lines.append(f"  train: {TRAIN_JSONL}")
    summary_lines.append(f"  test : {TEST_JSONL}")
    summary_lines.append(f"  val  : {VAL_JSONL}")
    summary_lines.append(f"  summary: {SUMMARY_TXT}")
    summary_lines.append("=" * 100)

    summary_text = "\n".join(summary_lines) + "\n"

    with SUMMARY_TXT.open("w", encoding="utf-8") as f:
        f.write(summary_text)

    print(summary_text)


if __name__ == "__main__":
    main()