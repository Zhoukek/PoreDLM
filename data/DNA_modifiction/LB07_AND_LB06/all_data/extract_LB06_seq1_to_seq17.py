#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from collections import Counter


# ============================================================
# 1. 输入输出路径
# ============================================================

INPUT_JSONL = Path(
    # "/mnt/zzbnew/rnamodel/data/DNA_data/S0_Artificial_synthesis/250F600084012/"
    # "signal2base_result_prefix/LB06/250F600084012_0_1_0_0/merge_adjust.jsonl"
    "/mnt/zzbnew/rnamodel/data/DNA_data/S0_Artificial_synthesis/250F600084012/signal2base_result_prefix/LB07/250F600084012_0_1_0_0/merge_adjust.jsonl"
)


OUTPUT_DIR = Path(
    "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/"
    "LB07_AND_LB06/all_data"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSONL = OUTPUT_DIR / "LB07-250F600084012_0_1_0_0_seq_1_to_seq_17_merge_adjust.jsonl"
OUTPUT_SUMMARY = OUTPUT_DIR / "LB07-250F600084012_0_1_0_0_seq_1_to_seq_17_summary.txt"


# ============================================================
# 2. 筛选 label
# ============================================================

TARGET_LABELS = {f"seq_{i}" for i in range(1, 18)}

PRINT_EVERY = 10000


# ============================================================
# 3. 主程序
# ============================================================

def main():
    if not INPUT_JSONL.exists():
        raise FileNotFoundError(f"输入文件不存在: {INPUT_JSONL}")

    total_lines = 0
    valid_json_lines = 0
    saved_lines = 0
    bad_json_lines = 0

    label_counter_all = Counter()
    label_counter_saved = Counter()
    skip_reason_counter = Counter()

    print("=" * 100)
    print("提取 label 为 seq_1 到 seq_17 的数据")
    print("=" * 100)
    print(f"输入文件: {INPUT_JSONL}")
    print(f"输出文件: {OUTPUT_JSONL}")
    print(f"summary: {OUTPUT_SUMMARY}")
    print(f"目标 labels: {sorted(TARGET_LABELS, key=lambda x: int(x.split('_')[1]))}")
    print("=" * 100)

    with INPUT_JSONL.open("r", encoding="utf-8") as fin, \
            OUTPUT_JSONL.open("w", encoding="utf-8") as fout:

        for line_idx, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                skip_reason_counter["empty_line"] += 1
                continue

            total_lines += 1

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad_json_lines += 1
                skip_reason_counter["json_decode_error"] += 1
                continue

            valid_json_lines += 1

            label = obj.get("label", None)

            if label is None:
                skip_reason_counter["missing_label"] += 1
                continue

            label = str(label)
            label_counter_all[label] += 1

            if label not in TARGET_LABELS:
                skip_reason_counter["label_not_in_seq_1_to_seq_17"] += 1
                continue

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            saved_lines += 1
            label_counter_saved[label] += 1

            if saved_lines % PRINT_EVERY == 0:
                print(f"[Progress] saved={saved_lines}, processed_lines={total_lines}")

    # ========================================================
    # 4. summary
    # ========================================================

    summary_lines = []
    summary_lines.append("=" * 100)
    summary_lines.append("提取 label 为 seq_1 到 seq_17 的数据结果")
    summary_lines.append("=" * 100)
    summary_lines.append(f"输入文件: {INPUT_JSONL}")
    summary_lines.append(f"输出文件: {OUTPUT_JSONL}")
    summary_lines.append("")
    summary_lines.append(f"总读取非空行数: {total_lines}")
    summary_lines.append(f"有效 JSON 行数: {valid_json_lines}")
    summary_lines.append(f"JSON 解析失败行数: {bad_json_lines}")
    summary_lines.append(f"成功保存行数: {saved_lines}")
    summary_lines.append(f"保存比例: {saved_lines / valid_json_lines if valid_json_lines else 0:.6f}")
    summary_lines.append("")
    summary_lines.append("目标 label 保存数量:")
    for i in range(1, 18):
        lab = f"seq_{i}"
        summary_lines.append(f"  {lab}: {label_counter_saved.get(lab, 0)}")
    summary_lines.append("")
    summary_lines.append("跳过原因统计:")
    for reason, count in skip_reason_counter.most_common():
        summary_lines.append(f"  {reason}: {count}")
    summary_lines.append("")
    summary_lines.append("输入文件中所有 label 计数 Top 50:")
    for label, count in label_counter_all.most_common(50):
        summary_lines.append(f"  {label}: {count}")
    summary_lines.append("=" * 100)

    summary_text = "\n".join(summary_lines) + "\n"

    with OUTPUT_SUMMARY.open("w", encoding="utf-8") as f:
        f.write(summary_text)

    print(summary_text)


if __name__ == "__main__":
    main()