import json
import ast
import time
import numpy as np
from pathlib import Path
from scipy.ndimage import median_filter
from scipy.signal import medfilt

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def parse_maybe_list(x):
    """
    兼容：
    1. x 本身就是 list
    2. x 是字符串形式的 list，例如 "[1, 2, 3]"
    """
    if isinstance(x, list):
        return x

    if isinstance(x, str):
        s = x.strip()

        if s.startswith("[") and s.endswith("]"):
            try:
                return json.loads(s)
            except Exception:
                try:
                    return ast.literal_eval(s)
                except Exception:
                    return x

        return x

    return x


def _nanopore_normalize_huada(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)

    if signal.size == 0:
        return np.array([], dtype=np.float32)

    med = np.median(signal)
    mad = 1.4826 * np.median(np.abs(signal - med))
    mad = max(mad, 1.0)

    normalized = (signal - med) / mad
    return normalized.astype(np.float32)


def _nanopore_normalize_novel(signal: np.ndarray):
    signal = np.asarray(signal, dtype=np.float32)

    if signal.size == 0:
        return np.array([], dtype=np.float32), 1.0

    signal_MED = np.median(signal)
    residual = signal - signal_MED

    q01, q99 = np.quantile(residual, [0.01, 0.99])
    masked_residual = residual[(residual >= q01) & (residual <= q99)]

    if masked_residual.size == 0:
        return np.array([], dtype=np.float32), 1.0

    global_MAD = 1.4826 * np.median(np.abs(masked_residual))
    global_MAD = max(global_MAD, 1.0)

    normalized = residual / global_MAD
    return normalized.astype(np.float32), global_MAD


def _nanopore_repair_errors(signal, min_value, max_value):
    signal = np.asarray(signal, dtype=np.float32)

    if signal.size == 0:
        return signal

    if not (np.any(signal < min_value) or np.any(signal > max_value)):
        return signal

    cleaned = signal.copy()

    valid_mask = (cleaned >= min_value) & (cleaned <= max_value)
    outlier_indices = np.where(~valid_mask)[0]

    for i in outlier_indices:
        if i < 1:
            if cleaned[0] > max_value:
                cleaned[0] = max_value
            else:
                cleaned[0] = min_value
        else:
            cleaned[i] = cleaned[i - 1]

    return cleaned


def _make_valid_median_window(window_size, signal_len):
    """
    median_filter 的窗口不能超过当前 signal 长度太多。
    这里保留 apple 原始 window_size=6000 的思想，
    但如果当前 signal 长度小于 6000，就自动缩小到不超过 signal_len 的奇数。
    """
    if signal_len <= 1:
        return 1

    actual_window_size = min(window_size, signal_len)

    if actual_window_size % 2 == 0:
        actual_window_size -= 1

    return max(actual_window_size, 1)


def _nanopore_remove_spikes(
    signal,
    window_size=6000,
    spike_threshold=5.0
):
    """
    Detect and remove spikes using global MAD on baseline-removed residual.

    这里恢复为原始 apple 的 window_size=6000。
    如果整条 signal 比 6000 短，会自动调整成合适的奇数窗口。
    """
    mad_factor = 1.4826
    min_mad = 1.0

    signal = np.asarray(signal, dtype=np.float32)

    if signal.size == 0:
        return signal

    actual_window_size = _make_valid_median_window(
        window_size=window_size,
        signal_len=signal.size
    )

    local_med = median_filter(
        signal,
        size=actual_window_size,
        mode="reflect"
    )

    residual = signal - local_med

    global_mad = mad_factor * np.median(np.abs(residual))
    global_mad = max(global_mad, min_mad)

    is_spike = np.abs(residual) > (spike_threshold * global_mad)

    if not np.any(is_spike):
        return signal.copy()

    cleaned = signal.copy()
    outlier_indices = np.where(is_spike)[0]

    for i in outlier_indices:
        if i == 0:
            cleaned[0] = local_med[0]
        else:
            cleaned[i] = cleaned[i - 1]

    return cleaned


def _nanopore_truncate_signal(signal: np.ndarray, truncate_threshold=3.0) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)

    if signal.size == 0:
        return signal

    truncated_signal = signal.copy()
    mask = np.abs(truncated_signal) > truncate_threshold

    truncated_signal[mask] = np.clip(
        truncated_signal[mask],
        -truncate_threshold,
        truncate_threshold
    )

    return truncated_signal


def _nanopore_soft_clip_tanh(signal: np.ndarray, limit=3.0) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)

    if signal.size == 0:
        return signal

    return (limit * np.tanh(signal / limit)).astype(np.float32)


def nanopore_process_signal(
    signal_raw,
    strategy="apple",
    spike_window_size=6000
):
    """
    对整条 signal 做预处理。

    重点：
    apple 策略里使用 spike_window_size=6000。
    之后再切 500 点 chunk。
    """
    signal_raw = np.asarray(signal_raw, dtype=np.float32)

    if strategy == "stone":
        signal_return = _nanopore_normalize_huada(signal_raw)

    elif strategy == "apple":
        signal_clear = _nanopore_repair_errors(signal_raw, 1, 220)

        signal_elite = _nanopore_remove_spikes(
            signal_clear,
            window_size=spike_window_size,
            spike_threshold=5.0
        )

        signal_nomal, _ = _nanopore_normalize_novel(signal_elite)

        if signal_nomal.size == 0:
            return signal_nomal

        signal_return = medfilt(
            signal_nomal,
            kernel_size=5
        ).astype(np.float32)

    elif strategy == "lemon":
        signal_clear = _nanopore_repair_errors(signal_raw, 1, 220)

        signal_elite = _nanopore_remove_spikes(
            signal_clear,
            window_size=spike_window_size,
            spike_threshold=5.0
        )

        signal_nomal, _ = _nanopore_normalize_novel(signal_elite)

        if signal_nomal.size == 0:
            return signal_nomal

        signal_medfilt = medfilt(
            signal_nomal,
            kernel_size=5
        ).astype(np.float32)

        signal_return = _nanopore_truncate_signal(
            signal_medfilt,
            truncate_threshold=5.0
        )

    elif strategy == "tango":
        signal_clear = _nanopore_repair_errors(signal_raw, 1, 220)

        signal_elite = _nanopore_remove_spikes(
            signal_clear,
            window_size=spike_window_size,
            spike_threshold=5.0
        )

        signal_nomal, _ = _nanopore_normalize_novel(signal_elite)

        signal_return = _nanopore_soft_clip_tanh(
            signal_nomal,
            limit=5.0
        )

    elif strategy == "mongo":
        signal_clear = _nanopore_repair_errors(signal_raw, 1, 220)

        signal_elite = _nanopore_remove_spikes(
            signal_clear,
            window_size=spike_window_size,
            spike_threshold=5.0
        )

        signal_nomal, _ = _nanopore_normalize_novel(signal_elite)

        signal_return = signal_nomal

    else:
        raise ValueError(f"未知 strategy: {strategy}")

    return signal_return.astype(np.float32)


def iter_sample_records(jsonl_path):
    """
    读取 jsonl，只保留 sample 行。
    自动跳过 stats 行。
    """
    jsonl_path = Path(jsonl_path)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            obj = json.loads(line)

            if obj.get("record_type") == "stats":
                continue

            if "record_type" in obj and obj.get("record_type") != "sample":
                continue

            yield obj


def count_sample_records(jsonl_path):
    """
    统计 jsonl 中 sample 行数量，用于 tqdm 显示总进度。
    """
    count = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            obj = json.loads(line)

            if obj.get("record_type") == "stats":
                continue

            if "record_type" in obj and obj.get("record_type") != "sample":
                continue

            count += 1

    return count


def iter_signal_chunks(signal, chunk_size=500, overlap=450):
    """
    将一条已经预处理后的 signal 切成固定长度 chunk。

    chunk_size = 500
    overlap = 450
    stride = 50
    """
    signal = np.asarray(signal, dtype=np.float32)

    stride = chunk_size - overlap

    if stride <= 0:
        raise ValueError(
            f"overlap 必须小于 chunk_size，当前 chunk_size={chunk_size}, overlap={overlap}"
        )

    if signal.size < chunk_size:
        return

    for start in range(0, signal.size - chunk_size + 1, stride):
        end = start + chunk_size
        yield signal[start:end]


def process_one_split_jsonl_to_npy(
    input_jsonl,
    output_npy,
    chunk_size=500,
    overlap=450,
    strategy="apple",
    spike_window_size=6000,
    progress_interval=1000,
    count_total_first=True
):
    """
    读取一个 split 的 jsonl，例如 train.jsonl。

    新流程：
    1. 读取完整 signal
    2. 先对整条 signal 做 apple 预处理
    3. 再切成 500 点 chunk，overlap=450
    4. 保存为 npy
    """
    input_jsonl = Path(input_jsonl)
    output_npy = Path(output_npy)
    output_npy.parent.mkdir(parents=True, exist_ok=True)

    stride = chunk_size - overlap

    print("=" * 80)
    print(f"开始处理文件: {input_jsonl}")
    print(f"输出文件: {output_npy}")
    print(f"处理流程: 整条 signal 先预处理，然后再切 chunk")
    print(f"chunk_size: {chunk_size}")
    print(f"overlap: {overlap}")
    print(f"stride: {stride}")
    print(f"strategy: {strategy}")
    print(f"spike_window_size: {spike_window_size}")
    print("=" * 80)

    if count_total_first:
        print("正在统计 sample 总数，用于显示进度条...")
        total_sample_records = count_sample_records(input_jsonl)
        print(f"sample 总数: {total_sample_records}")
    else:
        total_sample_records = None

    all_chunks = []

    total_records = 0
    valid_signal_records = 0
    short_signal_records = 0
    invalid_signal_records = 0
    preprocess_empty_records = 0
    processed_short_records = 0
    total_chunks = 0

    start_time = time.time()

    if tqdm is not None:
        pbar = tqdm(
            total=total_sample_records,
            desc=f"Processing {input_jsonl.name}",
            ncols=120
        )
    else:
        pbar = None

    for obj in iter_sample_records(input_jsonl):
        total_records += 1

        signal = parse_maybe_list(obj.get("signal"))

        if not isinstance(signal, list):
            invalid_signal_records += 1

            if pbar is not None:
                pbar.update(1)

            continue

        signal = np.asarray(signal, dtype=np.float32)

        if signal.ndim != 1:
            invalid_signal_records += 1

            if pbar is not None:
                pbar.update(1)

            continue

        raw_signal_len = signal.size

        if raw_signal_len < chunk_size:
            short_signal_records += 1

            if pbar is not None:
                pbar.update(1)

            continue

        valid_signal_records += 1

        # 核心修改：先对整条 signal 进行 apple 预处理
        processed_signal = nanopore_process_signal(
            signal,
            strategy=strategy,
            spike_window_size=spike_window_size
        )

        if processed_signal.size == 0:
            preprocess_empty_records += 1

            if pbar is not None:
                pbar.update(1)

            continue

        if processed_signal.size < chunk_size:
            processed_short_records += 1

            if pbar is not None:
                pbar.update(1)

            continue

        record_chunk_count = 0

        # 再对预处理后的整条 signal 切 500 点 chunk
        for chunk in iter_signal_chunks(
            processed_signal,
            chunk_size=chunk_size,
            overlap=overlap
        ):
            if chunk.size != chunk_size:
                continue

            all_chunks.append(chunk.astype(np.float32))
            total_chunks += 1
            record_chunk_count += 1

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix({
                "chunks": total_chunks,
                "raw_len": raw_signal_len,
                "proc_len": processed_signal.size,
                "last_chunks": record_chunk_count
            })

        if pbar is None and total_records % progress_interval == 0:
            elapsed = time.time() - start_time
            record_speed = total_records / max(elapsed, 1e-6)
            chunk_speed = total_chunks / max(elapsed, 1e-6)

            print("-" * 80)
            print(f"当前文件: {input_jsonl.name}")
            print(f"已处理 jsonl 样本数: {total_records}")
            print(f"有效 signal 样本数: {valid_signal_records}")
            print(f"过短信号样本数: {short_signal_records}")
            print(f"无效 signal 样本数: {invalid_signal_records}")
            print(f"预处理后为空样本数: {preprocess_empty_records}")
            print(f"当前原始 signal 长度: {raw_signal_len}")
            print(f"当前预处理后 signal 长度: {processed_signal.size}")
            print(f"当前 signal 切出 chunk 数: {record_chunk_count}")
            print(f"累计生成 chunk 数: {total_chunks}")
            print(f"已用时间: {elapsed / 60:.2f} min")
            print(f"处理速度: {record_speed:.2f} records/s")
            print(f"chunk 速度: {chunk_speed:.2f} chunks/s")
            print("-" * 80)

    if pbar is not None:
        pbar.close()

    print("正在 stack 所有 chunk 为 numpy array...")

    if len(all_chunks) == 0:
        chunk_array = np.empty((0, chunk_size), dtype=np.float32)
    else:
        chunk_array = np.stack(all_chunks, axis=0).astype(np.float32)

    print("正在保存 npy 文件...")

    np.save(output_npy, chunk_array)

    elapsed = time.time() - start_time

    print("=" * 80)
    print(f"完成: {input_jsonl.name}")
    print(f"总 jsonl 样本数: {total_records}")
    print(f"有效 signal 样本数: {valid_signal_records}")
    print(f"原始过短信号样本数: {short_signal_records}")
    print(f"无效 signal 样本数: {invalid_signal_records}")
    print(f"预处理后为空样本数: {preprocess_empty_records}")
    print(f"预处理后过短样本数: {processed_short_records}")
    print(f"生成 chunk 数: {total_chunks}")
    print(f"最终数组 shape: {chunk_array.shape}")
    print(f"最终数组 dtype: {chunk_array.dtype}")
    print(f"总耗时: {elapsed / 60:.2f} min")
    print(f"平均 record 速度: {total_records / max(elapsed, 1e-6):.2f} records/s")
    print(f"平均 chunk 速度: {total_chunks / max(elapsed, 1e-6):.2f} chunks/s")
    print(f"保存到: {output_npy}")
    print("=" * 80)

    return {
        "input_jsonl": str(input_jsonl),
        "output_npy": str(output_npy),
        "process_order": "full_signal_first_then_chunk",
        "total_records": total_records,
        "valid_signal_records": valid_signal_records,
        "short_signal_records": short_signal_records,
        "invalid_signal_records": invalid_signal_records,
        "preprocess_empty_records": preprocess_empty_records,
        "processed_short_records": processed_short_records,
        "total_chunks": total_chunks,
        "shape": chunk_array.shape,
        "dtype": str(chunk_array.dtype),
        "elapsed_seconds": elapsed,
        "records_per_second": total_records / max(elapsed, 1e-6),
        "chunks_per_second": total_chunks / max(elapsed, 1e-6),
    }


def process_all_splits(
    split_dir,
    output_dir,
    chunk_size=500,
    overlap=450,
    strategy="apple",
    spike_window_size=6000
):
    """
    依次处理 train.jsonl / validation.jsonl / test.jsonl。
    """
    split_dir = Path(split_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_files = {
        "train": split_dir / "train.jsonl",
        "validation": split_dir / "validation.jsonl",
        "test": split_dir / "test.jsonl",
    }

    all_stats = {}

    for split_name, jsonl_path in split_files.items():
        if not jsonl_path.exists():
            print(f"跳过，不存在: {jsonl_path}")
            continue

        output_npy = output_dir / f"{split_name}_signal_fullapple_chunk{chunk_size}_overlap{overlap}.npy"

        stats = process_one_split_jsonl_to_npy(
            input_jsonl=jsonl_path,
            output_npy=output_npy,
            chunk_size=chunk_size,
            overlap=overlap,
            strategy=strategy,
            spike_window_size=spike_window_size,
            progress_interval=1000,
            count_total_first=True
        )

        all_stats[split_name] = stats

    stats_path = output_dir / f"process_stats_fullapple_chunk{chunk_size}_overlap{overlap}.json"

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)

    print(f"统计信息已保存到: {stats_path}")


if __name__ == "__main__":
    split_dir = "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/LB07_AND_LB06/LB07/split"

    output_dir = "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/LB07_AND_LB06/LB07/signal_fullapple_chunks_500_overlap450"

    process_all_splits(
        split_dir=split_dir,
        output_dir=output_dir,
        chunk_size=500,
        overlap=450,
        strategy="apple",
        spike_window_size=6000
    )