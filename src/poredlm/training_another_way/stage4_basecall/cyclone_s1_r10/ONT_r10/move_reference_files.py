#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 test 目录下所有 *_references.npy 和 *_reference_lengths.npy 文件
移动到 test/reference 目录中。

移动前:
    test/
      xxx_chunks.npy
      xxx_references.npy
      xxx_reference_lengths.npy

移动后:
    test/
      xxx_chunks.npy
      reference/
        xxx_references.npy
        xxx_reference_lengths.npy
"""

import os
import argparse
import shutil
from pathlib import Path


# =============================================================================
# CONFIG：只需要修改这里
# =============================================================================

SOURCE_DIR = (
    "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/"
    "S0_HG002_UNMOD-35g/stage1_tokenizer/validation"
)

TARGET_DIR = (
    "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/"
    "S0_HG002_UNMOD-35g/stage1_tokenizer/validation/reference"
)

# 是否覆盖 reference 文件夹中已有的同名文件
OVERWRITE = False

# 是否只是预览，不真正移动
DRY_RUN = False


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
        description="Move *_references.npy and *_reference_lengths.npy into a reference folder."
    )
    parser.add_argument("--source_dir", default=SOURCE_DIR)
    parser.add_argument("--target_dir", default=TARGET_DIR)
    parser.add_argument("--overwrite", default=OVERWRITE, type=str2bool)
    parser.add_argument("--dry_run", default=DRY_RUN, type=str2bool)
    return parser.parse_args()


def collect_reference_files(source_dir: Path):
    """
    收集 source_dir 下的 *_references.npy 和 *_reference_lengths.npy 文件。
    注意：只搜索 source_dir 当前层，不递归搜索子目录。
    """
    reference_files = []

    patterns = [
        "*_references.npy",
        "*_reference_lengths.npy",
    ]

    for pattern in patterns:
        reference_files.extend(sorted(source_dir.glob(pattern)))

    reference_files = sorted(reference_files)

    return reference_files


def move_file(src_path: Path, target_dir: Path, overwrite: bool, dry_run: bool):
    dst_path = target_dir / src_path.name

    if dst_path.exists():
        if overwrite:
            print(f"[Overwrite] 删除已有文件: {dst_path}")
            if not dry_run:
                dst_path.unlink()
        else:
            raise FileExistsError(
                f"目标文件已存在，为避免误覆盖，程序停止:\n"
                f"{dst_path}\n\n"
                f"如果确认覆盖，请设置 OVERWRITE = True"
            )

    print(f"[Move] {src_path} -> {dst_path}")

    if not dry_run:
        shutil.move(str(src_path), str(dst_path))


def main():
    global SOURCE_DIR, TARGET_DIR, OVERWRITE, DRY_RUN

    args = parse_args()
    SOURCE_DIR = args.source_dir
    TARGET_DIR = args.target_dir
    OVERWRITE = args.overwrite
    DRY_RUN = args.dry_run

    source_dir = Path(SOURCE_DIR)
    target_dir = Path(TARGET_DIR)

    print("=" * 100)
    print("Move reference files")
    print("=" * 100)
    print(f"[Source Dir] {source_dir}")
    print(f"[Target Dir] {target_dir}")
    print(f"[Overwrite] {OVERWRITE}")
    print(f"[Dry Run] {DRY_RUN}")
    print("=" * 100)

    if not source_dir.exists():
        raise FileNotFoundError(f"源目录不存在: {source_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)

    files = collect_reference_files(source_dir)

    if len(files) == 0:
        print("[Info] 没有找到 *_references.npy 或 *_reference_lengths.npy 文件")
        return

    print(f"[Found] 待移动文件数量: {len(files)}")
    for f in files:
        print(f"  {f.name}")

    print("=" * 100)

    for src_path in files:
        move_file(
            src_path=src_path,
            target_dir=target_dir,
            overwrite=OVERWRITE,
            dry_run=DRY_RUN,
        )

    print("=" * 100)
    print("[Done] 所有 reference 文件移动完成")
    print("=" * 100)


if __name__ == "__main__":
    main()
