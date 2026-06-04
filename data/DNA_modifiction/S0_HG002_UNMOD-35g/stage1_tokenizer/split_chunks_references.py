#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 all_data 文件夹下所有 *_chunks.npy 文件按 8:1:1 随机划分为
train / test / validation。

同时按照完全相同的索引划分对应的：
1. *_references.npy
2. *_reference_lengths.npy

输入文件示例：
    250F601844011_0_0_0_0_chunks.npy
    250F601844011_0_0_0_0_references.npy
    250F601844011_0_0_0_0_reference_lengths.npy

输出到：
    train/
    test/
    validation/
"""

import os
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm


# =============================================================================
# CONFIG：一般只需要修改这里
# =============================================================================

INPUT_DIR = (
    "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/"
    "S0_HG002_UNMOD-35g/stage1_tokenizer/all_data"
)

TRAIN_DIR = (
    "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/"
    "S0_HG002_UNMOD-35g/stage1_tokenizer/train"
)

TEST_DIR = (
    "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/"
    "S0_HG002_UNMOD-35g/stage1_tokenizer/test"
)

VALIDATION_DIR = (
    "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/"
    "S0_HG002_UNMOD-35g/stage1_tokenizer/validation"
)

TRAIN_RATIO = 0.8
TEST_RATIO = 0.1
VAL_RATIO = 0.1

SEED = 42

# 每次写入多少条样本，数据大时可以适当调小，比如 512 / 1024
BATCH_SIZE = 4096

# 是否覆盖已有文件
OVERWRITE = False

# 是否检查 chunks 第二维是否为 6000
CHECK_CHUNK_LEN = True
EXPECTED_CHUNK_LEN = 6000


# =============================================================================
# 工具函数
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
        description="Split merged chunks/references into train/test/validation."
    )
    parser.add_argument("--input_dir", default=INPUT_DIR)
    parser.add_argument("--train_dir", default=TRAIN_DIR)
    parser.add_argument("--test_dir", default=TEST_DIR)
    parser.add_argument("--validation_dir", default=VALIDATION_DIR)
    parser.add_argument("--train_ratio", default=TRAIN_RATIO, type=float)
    parser.add_argument("--test_ratio", default=TEST_RATIO, type=float)
    parser.add_argument("--val_ratio", default=VAL_RATIO, type=float)
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--batch_size", default=BATCH_SIZE, type=int)
    parser.add_argument("--overwrite", default=OVERWRITE, type=str2bool)
    parser.add_argument("--check_chunk_len", default=CHECK_CHUNK_LEN, type=str2bool)
    parser.add_argument("--expected_chunk_len", default=EXPECTED_CHUNK_LEN, type=int)
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def check_ratios():
    s = TRAIN_RATIO + TEST_RATIO + VAL_RATIO
    if abs(s - 1.0) > 1e-6:
        raise ValueError(
            f"划分比例之和必须为 1.0, 当前为 {s}: "
            f"train={TRAIN_RATIO}, test={TEST_RATIO}, val={VAL_RATIO}"
        )


def check_output_file(path):
    if os.path.exists(path):
        if OVERWRITE:
            print(f"[Overwrite] 删除已有文件: {path}")
            os.remove(path)
        else:
            raise FileExistsError(
                f"输出文件已存在，为避免误覆盖，程序停止: {path}\n"
                f"如果确认覆盖，请设置 OVERWRITE = True"
            )


def find_all_chunk_files(input_dir):
    input_path = Path(input_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    chunk_files = sorted(input_path.glob("*_chunks.npy"))

    if len(chunk_files) == 0:
        raise FileNotFoundError(f"在 {input_dir} 下没有找到 *_chunks.npy 文件")

    return chunk_files


def get_prefix_from_chunk_path(chunk_path):
    name = chunk_path.name

    if not name.endswith("_chunks.npy"):
        raise ValueError(f"不是合法的 chunks 文件名: {name}")

    return name[:-len("_chunks.npy")]


def get_related_paths(input_dir, prefix):
    input_dir = Path(input_dir)

    chunk_path = input_dir / f"{prefix}_chunks.npy"
    refs_path = input_dir / f"{prefix}_references.npy"
    lengths_path = input_dir / f"{prefix}_reference_lengths.npy"

    if not chunk_path.exists():
        raise FileNotFoundError(f"缺少 chunks 文件: {chunk_path}")
    if not refs_path.exists():
        raise FileNotFoundError(f"缺少 references 文件: {refs_path}")
    if not lengths_path.exists():
        raise FileNotFoundError(f"缺少 reference_lengths 文件: {lengths_path}")

    return chunk_path, refs_path, lengths_path


def split_indices(n, seed):
    """
    对 n 条样本生成 8:1:1 随机划分索引。
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    n_train = int(n * TRAIN_RATIO)
    n_test = int(n * TEST_RATIO)
    n_val = n - n_train - n_test

    train_idx = indices[:n_train]
    test_idx = indices[n_train:n_train + n_test]
    val_idx = indices[n_train + n_test:]

    assert len(train_idx) + len(test_idx) + len(val_idx) == n

    return {
        "train": train_idx,
        "test": test_idx,
        "validation": val_idx,
    }


def save_subset_memmap(src_array, indices, output_path, batch_size=4096):
    """
    按 indices 从 src_array 中取子集，并保存到 output_path。
    使用 open_memmap 分批写入，避免一次性占用过大内存。
    """
    output_shape = (len(indices), *src_array.shape[1:])
    output_dtype = src_array.dtype

    check_output_file(output_path)

    dst_array = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=output_dtype,
        shape=output_shape,
    )

    offset = 0

    for start in tqdm(
        range(0, len(indices), batch_size),
        desc=f"Writing {Path(output_path).name}",
        leave=False,
    ):
        end = min(start + batch_size, len(indices))
        batch_indices = indices[start:end]

        # 保持随机划分后的顺序
        batch_data = src_array[batch_indices]

        dst_array[offset:offset + len(batch_indices)] = batch_data
        offset += len(batch_indices)

    dst_array.flush()

    print(f"[Save] {output_path} shape={output_shape}, dtype={output_dtype}")


def process_one_file(prefix, input_dir, output_dirs, seed):
    """
    处理一个 prefix 对应的三件套：
        prefix_chunks.npy
        prefix_references.npy
        prefix_reference_lengths.npy
    """
    chunk_path, refs_path, lengths_path = get_related_paths(input_dir, prefix)

    print("=" * 100)
    print(f"[Process] {prefix}")
    print(f"[Chunks] {chunk_path}")
    print(f"[References] {refs_path}")
    print(f"[Reference Lengths] {lengths_path}")
    print("=" * 100)

    chunks = np.load(chunk_path, mmap_mode="r", allow_pickle=False)
    refs = np.load(refs_path, mmap_mode="r", allow_pickle=False)
    lengths = np.load(lengths_path, mmap_mode="r", allow_pickle=False)

    print(f"[Shape] chunks={chunks.shape}, dtype={chunks.dtype}")
    print(f"[Shape] references={refs.shape}, dtype={refs.dtype}")
    print(f"[Shape] reference_lengths={lengths.shape}, dtype={lengths.dtype}")

    if chunks.ndim != 2:
        print(f"[Warning] chunks 不是二维数组: shape={chunks.shape}")

    if CHECK_CHUNK_LEN and chunks.ndim >= 2:
        if chunks.shape[1] != EXPECTED_CHUNK_LEN:
            print(
                f"[Warning] chunks 第二维不是 {EXPECTED_CHUNK_LEN}: "
                f"当前 shape={chunks.shape}"
            )

    n = chunks.shape[0]

    if refs.shape[0] != n:
        raise ValueError(
            f"references 样本数和 chunks 不一致: "
            f"chunks={chunks.shape}, references={refs.shape}"
        )

    if lengths.shape[0] != n:
        raise ValueError(
            f"reference_lengths 样本数和 chunks 不一致: "
            f"chunks={chunks.shape}, reference_lengths={lengths.shape}"
        )

    # 为了保证不同文件的随机划分不同，但又可复现，给 seed 加一个 prefix hash
    file_seed = seed + (abs(hash(prefix)) % 1000000)
    split_dict = split_indices(n, file_seed)

    print(
        f"[Split] total={n}, "
        f"train={len(split_dict['train'])}, "
        f"test={len(split_dict['test'])}, "
        f"validation={len(split_dict['validation'])}"
    )

    for split_name, indices in split_dict.items():
        out_dir = output_dirs[split_name]

        out_chunks = os.path.join(out_dir, f"{prefix}_chunks.npy")
        out_refs = os.path.join(out_dir, f"{prefix}_references.npy")
        out_lengths = os.path.join(out_dir, f"{prefix}_reference_lengths.npy")

        print("-" * 100)
        print(f"[Split Save] {prefix} -> {split_name}")
        print(f"[Num Samples] {len(indices)}")
        print("-" * 100)

        save_subset_memmap(
            src_array=chunks,
            indices=indices,
            output_path=out_chunks,
            batch_size=BATCH_SIZE,
        )

        save_subset_memmap(
            src_array=refs,
            indices=indices,
            output_path=out_refs,
            batch_size=BATCH_SIZE,
        )

        save_subset_memmap(
            src_array=lengths,
            indices=indices,
            output_path=out_lengths,
            batch_size=BATCH_SIZE,
        )


def main():
    global INPUT_DIR, TRAIN_DIR, TEST_DIR, VALIDATION_DIR
    global TRAIN_RATIO, TEST_RATIO, VAL_RATIO, SEED
    global BATCH_SIZE, OVERWRITE, CHECK_CHUNK_LEN, EXPECTED_CHUNK_LEN

    args = parse_args()
    INPUT_DIR = args.input_dir
    TRAIN_DIR = args.train_dir
    TEST_DIR = args.test_dir
    VALIDATION_DIR = args.validation_dir
    TRAIN_RATIO = args.train_ratio
    TEST_RATIO = args.test_ratio
    VAL_RATIO = args.val_ratio
    SEED = args.seed
    BATCH_SIZE = args.batch_size
    OVERWRITE = args.overwrite
    CHECK_CHUNK_LEN = args.check_chunk_len
    EXPECTED_CHUNK_LEN = args.expected_chunk_len

    check_ratios()

    ensure_dir(TRAIN_DIR)
    ensure_dir(TEST_DIR)
    ensure_dir(VALIDATION_DIR)

    output_dirs = {
        "train": TRAIN_DIR,
        "test": TEST_DIR,
        "validation": VALIDATION_DIR,
    }

    print("=" * 100)
    print("Split all *_chunks.npy with corresponding references.npy and reference_lengths.npy")
    print("=" * 100)
    print(f"[Input Dir] {INPUT_DIR}")
    print(f"[Train Dir] {TRAIN_DIR}")
    print(f"[Test Dir] {TEST_DIR}")
    print(f"[Validation Dir] {VALIDATION_DIR}")
    print(f"[Ratio] train:test:validation = {TRAIN_RATIO}:{TEST_RATIO}:{VAL_RATIO}")
    print(f"[Seed] {SEED}")
    print(f"[Overwrite] {OVERWRITE}")
    print("=" * 100)

    chunk_files = find_all_chunk_files(INPUT_DIR)

    print(f"[Found] chunks 文件数量: {len(chunk_files)}")

    prefixes = [get_prefix_from_chunk_path(p) for p in chunk_files]

    for prefix in prefixes:
        process_one_file(
            prefix=prefix,
            input_dir=INPUT_DIR,
            output_dirs=output_dirs,
            seed=SEED,
        )

    print("=" * 100)
    print("[All Done]")
    print("=" * 100)


if __name__ == "__main__":
    main()
