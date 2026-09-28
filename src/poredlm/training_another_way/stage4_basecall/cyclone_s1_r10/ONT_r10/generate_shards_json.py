#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生成 shards.json。

功能：
1. 读取指定目录下所有 *_chunks.npy 文件。
2. 统计每个 chunks 文件的样本数。
3. 检查所有 chunks 文件的 chunk_size 和 dtype 是否一致。
4. 生成 shards.json。

输出格式：
{
  "total_samples": 29021,
  "chunk_size": 5000,
  "dtype": "float16",
  "shards": [
    {
      "path": "chunks_test.npy",
      "num_samples": 29021
    }
  ]
}
"""

import json
import os
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


# =============================================================================
# CONFIG：只需要修改这里
# =============================================================================

INPUT_DIR = (
    "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/"
    "/S0_HG002_UNMOD-35g/stage1_tokenizer/train"
)

OUTPUT_JSON = "shards.json"

FILE_PATTERN = "*_chunks.npy"

# 是否允许覆盖已有 shards.json
OVERWRITE = True

# 如果你想强制检查 chunk_size 是否为 6000，就设为 True
CHECK_EXPECTED_CHUNK_SIZE = True
EXPECTED_CHUNK_SIZE = 6000


# =============================================================================
# 主逻辑
# =============================================================================

def str2bool(v):
    if isinstance(v, bool):
        return v

    v = str(v).lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate shards.json for a directory containing *_chunks.npy files."
    )
    parser.add_argument("--input_dir", default=INPUT_DIR)
    parser.add_argument("--output_json", default=OUTPUT_JSON)
    parser.add_argument("--file_pattern", default=FILE_PATTERN)
    parser.add_argument("--overwrite", default=OVERWRITE, type=str2bool)
    parser.add_argument("--check_expected_chunk_size", default=CHECK_EXPECTED_CHUNK_SIZE, type=str2bool)
    parser.add_argument("--expected_chunk_size", default=EXPECTED_CHUNK_SIZE, type=int)
    return parser.parse_args()


def find_chunk_files(input_dir: str, pattern: str):
    input_path = Path(input_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    chunk_files = sorted(input_path.glob(pattern))

    if len(chunk_files) == 0:
        raise FileNotFoundError(
            f"在 {input_dir} 下没有找到匹配 {pattern} 的 chunks 文件"
        )

    return chunk_files


def inspect_chunk_file(path: Path):
    """
    读取 npy 文件 shape / dtype。
    使用 mmap_mode='r'，不会一次性加载整个大文件。
    """
    arr = np.load(path, mmap_mode="r", allow_pickle=False)

    if arr.ndim != 2:
        raise ValueError(
            f"chunks 文件应该是二维数组 [N, chunk_size]，"
            f"但当前文件 shape={arr.shape}, file={path}"
        )

    num_samples = int(arr.shape[0])
    chunk_size = int(arr.shape[1])
    dtype = str(arr.dtype)

    return num_samples, chunk_size, dtype


def main():
    global INPUT_DIR, OUTPUT_JSON, FILE_PATTERN
    global OVERWRITE, CHECK_EXPECTED_CHUNK_SIZE, EXPECTED_CHUNK_SIZE

    args = parse_args()
    INPUT_DIR = args.input_dir
    OUTPUT_JSON = args.output_json
    FILE_PATTERN = args.file_pattern
    OVERWRITE = args.overwrite
    CHECK_EXPECTED_CHUNK_SIZE = args.check_expected_chunk_size
    EXPECTED_CHUNK_SIZE = args.expected_chunk_size

    input_dir = Path(INPUT_DIR)
    output_path = input_dir / OUTPUT_JSON

    print("=" * 80)
    print("Generate shards.json")
    print("=" * 80)
    print(f"[Input Dir] {input_dir}")
    print(f"[Output JSON] {output_path}")
    print(f"[Pattern] {FILE_PATTERN}")
    print("=" * 80)

    if output_path.exists() and not OVERWRITE:
        raise FileExistsError(
            f"输出文件已存在: {output_path}\n"
            f"如果确认覆盖，请设置 OVERWRITE = True"
        )

    chunk_files = find_chunk_files(str(input_dir), FILE_PATTERN)

    print(f"[Found] chunks 文件数量: {len(chunk_files)}")

    shards = []
    total_samples = 0
    global_chunk_size = None
    global_dtype = None

    for chunk_file in tqdm(chunk_files, desc="Inspecting chunks"):
        num_samples, chunk_size, dtype = inspect_chunk_file(chunk_file)

        print(
            f"[File] {chunk_file.name} | "
            f"shape=({num_samples}, {chunk_size}) | dtype={dtype}"
        )

        if global_chunk_size is None:
            global_chunk_size = chunk_size
        else:
            if chunk_size != global_chunk_size:
                raise ValueError(
                    f"chunk_size 不一致:\n"
                    f"  expected={global_chunk_size}\n"
                    f"  got={chunk_size}\n"
                    f"  file={chunk_file}"
                )

        if global_dtype is None:
            global_dtype = dtype
        else:
            if dtype != global_dtype:
                raise ValueError(
                    f"dtype 不一致:\n"
                    f"  expected={global_dtype}\n"
                    f"  got={dtype}\n"
                    f"  file={chunk_file}"
                )

        if CHECK_EXPECTED_CHUNK_SIZE and chunk_size != EXPECTED_CHUNK_SIZE:
            raise ValueError(
                f"chunk_size 不是期望的 {EXPECTED_CHUNK_SIZE}:\n"
                f"  got={chunk_size}\n"
                f"  file={chunk_file}"
            )

        total_samples += num_samples

        # path 使用相对路径，和你给的示例一致
        shards.append(
            {
                "path": chunk_file.name,
                "num_samples": num_samples,
            }
        )

    result = {
        "total_samples": int(total_samples),
        "chunk_size": int(global_chunk_size),
        "dtype": str(global_dtype),
        "shards": shards,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("[Done] shards.json 生成完成")
    print(f"[Output] {output_path}")
    print(f"[total_samples] {total_samples}")
    print(f"[chunk_size] {global_chunk_size}")
    print(f"[dtype] {global_dtype}")
    print(f"[num_shards] {len(shards)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
