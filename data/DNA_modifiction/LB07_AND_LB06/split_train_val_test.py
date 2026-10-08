import json
import random
from pathlib import Path


def split_jsonl(
    input_jsonl,
    output_dir,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
    keep_record_type=True
):
    """
    将合并后的 jsonl 文件划分为 train / validation / test。

    参数：
    input_jsonl: 输入 jsonl 文件路径
    output_dir: 输出文件夹
    train_ratio, val_ratio, test_ratio: 数据划分比例
    seed: 随机种子，保证每次划分一致
    keep_record_type: 是否在输出中保留 record_type/source_folder 等字段
    """

    input_jsonl = Path(input_jsonl)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_jsonl.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_jsonl}")

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio 必须等于 1，当前为 {ratio_sum}"
        )

    records = []
    skipped_stats = 0
    skipped_empty = 0

    print(f"读取文件: {input_jsonl}")

    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                skipped_empty += 1
                continue

            obj = json.loads(line)

            # 跳过最后的统计行
            if obj.get("record_type") == "stats":
                skipped_stats += 1
                continue

            # 如果有 record_type 字段，只保留 sample
            if "record_type" in obj and obj.get("record_type") != "sample":
                continue

            if not keep_record_type:
                obj = {
                    "pattern": obj.get("pattern"),
                    "base_sample_spans_rel": obj.get("base_sample_spans_rel"),
                    "signal": obj.get("signal"),
                }

            records.append(obj)

    print(f"有效样本数: {len(records)}")
    print(f"跳过 stats 行数: {skipped_stats}")
    print(f"跳过空行数: {skipped_empty}")

    random.seed(seed)
    random.shuffle(records)

    total = len(records)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_records = records[:train_end]
    val_records = records[train_end:val_end]
    test_records = records[val_end:]

    output_train = output_dir / "train.jsonl"
    output_val = output_dir / "validation.jsonl"
    output_test = output_dir / "test.jsonl"

    def write_jsonl(path, data):
        with open(path, "w", encoding="utf-8") as fout:
            for obj in data:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    write_jsonl(output_train, train_records)
    write_jsonl(output_val, val_records)
    write_jsonl(output_test, test_records)

    print("=" * 60)
    print("划分完成")
    print(f"Train:      {len(train_records)} -> {output_train}")
    print(f"Validation: {len(val_records)} -> {output_val}")
    print(f"Test:       {len(test_records)} -> {output_test}")
    print("=" * 60)


if __name__ == "__main__":
    split_jsonl(
        input_jsonl="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/LB07_AND_LB06/LB06/lb06_signal_none_selected.jsonl",
        output_dir="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/LB07_AND_LB06/LB06/split",
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
        keep_record_type=True
    )