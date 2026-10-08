#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
递归读取输入目录下所有子文件夹中的 chunks.npy 和 references.npy，
并合并保存到指定输出目录。

特点：
1. 支持命令行参数传入 input_root_dir / output_dir / output_prefix / overwrite。
2. chunks.npy 直接按第 0 维合并。
3. references.npy 如果长度不一致，会自动 padding 到最大长度。
4. 额外保存 reference_lengths.npy，记录每条 reference 的真实长度。
5. 使用 open_memmap 分批写入，避免一次性占用过大内存。
"""

import os
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


# =============================================================================
# 默认配置：如果命令行不传参，就使用这里的默认值
# =============================================================================

DEFAULT_INPUT_ROOT_DIR = (
    "/mnt/zzbnew/rnamodel/wangxue/data/DNA_data/S0_HG002_UNMOD/"
    "250F601844011/basecall_chunk/250F601844011_0_0_0_0"
)

DEFAULT_OUTPUT_DIR = (
    "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/"
    "S0_HG002_UNMOD-35g/stage1_tokenizer/all_data"
)

DEFAULT_OUTPUT_PREFIX = "250F601844011_0_0_0_0"

CHUNKS_FILENAME = "chunks.npy"
REFERENCES_FILENAME = "references.npy"

REFERENCE_PAD_VALUE = -1


# =============================================================================
# 参数解析
# =============================================================================

def str2bool(v):
    """
    将 bash 传入的 true/false 字符串转为 bool。
    """
    if isinstance(v, bool):
        return v

    v = str(v).lower()

    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError(
        f"Boolean value expected, got: {v}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge chunks.npy and references.npy recursively."
    )

    parser.add_argument(
        "--input_root_dir",
        type=str,
        default=DEFAULT_INPUT_ROOT_DIR,
        help="输入大目录，脚本会递归查找其子目录中的 chunks.npy 和 references.npy",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="输出目录",
    )

    parser.add_argument(
        "--output_prefix",
        type=str,
        default=DEFAULT_OUTPUT_PREFIX,
        help="输出文件名前缀，例如 250F601844011_0_0_0_1",
    )

    parser.add_argument(
        "--overwrite",
        type=str2bool,
        default=False,
        help="是否覆盖已有输出文件，true/false",
    )

    parser.add_argument(
        "--chunks_filename",
        type=str,
        default=CHUNKS_FILENAME,
        help="输入 chunks 文件名，默认 chunks.npy",
    )

    parser.add_argument(
        "--references_filename",
        type=str,
        default=REFERENCES_FILENAME,
        help="输入 references 文件名，默认 references.npy",
    )

    parser.add_argument(
        "--reference_pad_value",
        type=int,
        default=REFERENCE_PAD_VALUE,
        help="references padding 值，默认 -1",
    )

    return parser.parse_args()


# =============================================================================
# 工具函数
# =============================================================================

def check_output_files(output_paths, overwrite=False):
    existing = [p for p in output_paths if os.path.exists(p)]

    if existing and not overwrite:
        msg = "\n".join(existing)
        raise FileExistsError(
            f"以下输出文件已存在，为避免误覆盖，程序停止：\n{msg}\n\n"
            f"如果确认要覆盖，请使用参数：--overwrite true"
        )

    if existing and overwrite:
        for p in existing:
            print(f"[Overwrite] 删除已有文件: {p}")
            os.remove(p)


def collect_sample_pairs(input_root_dir, chunks_filename, references_filename):
    """
    收集所有同时包含 chunks.npy 和 references.npy 的子文件夹。

    返回:
        pairs: [(chunk_path, reference_path), ...]
    """
    root = Path(input_root_dir)

    if not root.exists():
        raise FileNotFoundError(f"输入目录不存在: {root}")

    chunk_files = sorted(root.rglob(chunks_filename))

    pairs = []

    for chunk_path in chunk_files:
        folder = chunk_path.parent
        reference_path = folder / references_filename

        if reference_path.exists():
            pairs.append((chunk_path, reference_path))
        else:
            print(
                f"[Warning] 找到 {chunks_filename} "
                f"但缺少 {references_filename}: {folder}"
            )

    if len(pairs) == 0:
        raise FileNotFoundError(
            f"在 {input_root_dir} 下没有找到成对的 "
            f"{chunks_filename} 和 {references_filename}"
        )

    return pairs


def scan_shapes(pairs):
    """
    第一遍扫描所有文件 shape：
    1. 统计总样本数
    2. 找到 reference 最大长度
    3. 检查每个文件夹内 chunks 和 references 的样本数是否一致
    4. 检查 chunks 除第 0 维外的 shape 是否一致
    """
    total_samples = 0
    max_reference_len = 0

    expected_chunk_tail_shape = None
    chunk_dtype = None
    reference_dtype = None

    print("=" * 80)
    print("[Step 1] 扫描文件 shape")
    print("=" * 80)

    for chunk_path, reference_path in tqdm(pairs, desc="Scanning"):
        chunks = np.load(chunk_path, mmap_mode="r", allow_pickle=False)
        refs = np.load(reference_path, mmap_mode="r", allow_pickle=False)

        if chunks.ndim < 2:
            raise ValueError(
                f"chunks 维度异常: {chunk_path}, shape={chunks.shape}"
            )

        current_chunk_tail_shape = chunks.shape[1:]

        if expected_chunk_tail_shape is None:
            expected_chunk_tail_shape = current_chunk_tail_shape
            chunk_dtype = chunks.dtype
        else:
            if current_chunk_tail_shape != expected_chunk_tail_shape:
                raise ValueError(
                    f"chunks 后续维度不一致:\n"
                    f"  file={chunk_path}\n"
                    f"  expected tail shape={expected_chunk_tail_shape}\n"
                    f"  got shape={chunks.shape}"
                )

        if reference_dtype is None:
            reference_dtype = refs.dtype

        if refs.ndim == 1:
            ref_num = 1
            ref_len = refs.shape[0]
        elif refs.ndim == 2:
            ref_num = refs.shape[0]
            ref_len = refs.shape[1]
        else:
            raise ValueError(
                f"references 维度异常: {reference_path}, shape={refs.shape}"
            )

        chunk_num = chunks.shape[0]

        if chunk_num != ref_num:
            raise ValueError(
                f"样本数不一致:\n"
                f"  folder={chunk_path.parent}\n"
                f"  chunks shape={chunks.shape}\n"
                f"  references shape={refs.shape}"
            )

        total_samples += chunk_num
        max_reference_len = max(max_reference_len, ref_len)

    print(f"[Info] 成对文件数量: {len(pairs)}")
    print(f"[Info] 总样本数: {total_samples}")
    print(f"[Info] chunks tail shape: {expected_chunk_tail_shape}")
    print(f"[Info] chunks dtype: {chunk_dtype}")
    print(f"[Info] references 最大长度: {max_reference_len}")
    print(f"[Info] references dtype: {reference_dtype}")

    return {
        "total_samples": total_samples,
        "max_reference_len": max_reference_len,
        "chunk_tail_shape": expected_chunk_tail_shape,
        "chunk_dtype": chunk_dtype,
        "reference_dtype": reference_dtype,
    }


def merge_chunks(pairs, output_path, total_samples, chunk_tail_shape, chunk_dtype):
    """
    合并 chunks.npy。
    """
    output_shape = (total_samples, *chunk_tail_shape)

    print("=" * 80)
    print("[Step 2] 合并 chunks")
    print("=" * 80)
    print(f"[Output] {output_path}")
    print(f"[Shape] {output_shape}")
    print(f"[Dtype] {chunk_dtype}")

    merged_chunks = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=chunk_dtype,
        shape=output_shape,
    )

    offset = 0

    for chunk_path, _ in tqdm(pairs, desc="Merging chunks"):
        arr = np.load(chunk_path, mmap_mode="r", allow_pickle=False)
        n = arr.shape[0]

        merged_chunks[offset:offset + n] = arr
        offset += n

    merged_chunks.flush()

    print(f"[Done] chunks 合并完成: {output_path}")


def merge_references_and_lengths(
    pairs,
    references_output_path,
    lengths_output_path,
    total_samples,
    max_reference_len,
    reference_dtype,
    pad_value,
):
    """
    合并 references.npy。
    如果 reference 长度不一致，则右侧 padding 到 max_reference_len。
    同时保存每条样本的真实 reference 长度。
    """
    output_shape = (total_samples, max_reference_len)

    print("=" * 80)
    print("[Step 3] 合并 references 并保存 reference_lengths")
    print("=" * 80)
    print(f"[References Output] {references_output_path}")
    print(f"[Lengths Output] {lengths_output_path}")
    print(f"[Shape] {output_shape}")
    print(f"[Dtype] {reference_dtype}")
    print(f"[Pad Value] {pad_value}")

    merged_refs = np.lib.format.open_memmap(
        references_output_path,
        mode="w+",
        dtype=reference_dtype,
        shape=output_shape,
    )

    reference_lengths = np.lib.format.open_memmap(
        lengths_output_path,
        mode="w+",
        dtype=np.int32,
        shape=(total_samples,),
    )

    # 先整体填充 padding
    merged_refs[:] = pad_value

    offset = 0

    for _, reference_path in tqdm(pairs, desc="Merging references"):
        refs = np.load(reference_path, mmap_mode="r", allow_pickle=False)

        if refs.ndim == 1:
            refs = refs[None, :]

        n, cur_len = refs.shape

        merged_refs[offset:offset + n, :cur_len] = refs
        reference_lengths[offset:offset + n] = cur_len

        offset += n

    merged_refs.flush()
    reference_lengths.flush()

    print(f"[Done] references 合并完成: {references_output_path}")
    print(f"[Done] reference_lengths 保存完成: {lengths_output_path}")


def main():
    args = parse_args()

    input_root_dir = args.input_root_dir
    output_dir = args.output_dir
    output_prefix = args.output_prefix
    overwrite = args.overwrite
    chunks_filename = args.chunks_filename
    references_filename = args.references_filename
    reference_pad_value = args.reference_pad_value

    os.makedirs(output_dir, exist_ok=True)

    output_chunks_path = os.path.join(
        output_dir,
        f"{output_prefix}_chunks.npy",
    )

    output_references_path = os.path.join(
        output_dir,
        f"{output_prefix}_references.npy",
    )

    output_reference_lengths_path = os.path.join(
        output_dir,
        f"{output_prefix}_reference_lengths.npy",
    )

    output_paths = [
        output_chunks_path,
        output_references_path,
        output_reference_lengths_path,
    ]

    print("=" * 80)
    print("Merge chunks.npy and references.npy")
    print("=" * 80)
    print(f"[Input Root] {input_root_dir}")
    print(f"[Output Dir] {output_dir}")
    print(f"[Output Prefix] {output_prefix}")
    print(f"[Chunks Filename] {chunks_filename}")
    print(f"[References Filename] {references_filename}")
    print(f"[Reference Pad Value] {reference_pad_value}")
    print(f"[Overwrite] {overwrite}")
    print("=" * 80)

    check_output_files(output_paths, overwrite=overwrite)

    pairs = collect_sample_pairs(
        input_root_dir=input_root_dir,
        chunks_filename=chunks_filename,
        references_filename=references_filename,
    )

    print(f"[Found] 成对文件数量: {len(pairs)}")

    info = scan_shapes(pairs)

    merge_chunks(
        pairs=pairs,
        output_path=output_chunks_path,
        total_samples=info["total_samples"],
        chunk_tail_shape=info["chunk_tail_shape"],
        chunk_dtype=info["chunk_dtype"],
    )

    merge_references_and_lengths(
        pairs=pairs,
        references_output_path=output_references_path,
        lengths_output_path=output_reference_lengths_path,
        total_samples=info["total_samples"],
        max_reference_len=info["max_reference_len"],
        reference_dtype=info["reference_dtype"],
        pad_value=reference_pad_value,
    )

    print("=" * 80)
    print("[All Done]")
    print(f"chunks: {output_chunks_path}")
    print(f"references: {output_references_path}")
    print(f"reference_lengths: {output_reference_lengths_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()