import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import medfilt

def _nanopore_normalize_huada(signal: np.ndarray) -> np.ndarray:
    """
    使用 Median-MAD 方法对 Nanopore 信号进行标准化归一化（工业级实现）。
    归一化公式：
        normalized = (signal - median) / MAD
    其中：
        - median = np.median(signal)
        - MAD = 1.4826 * np.median(|signal - median|)
        - 系数 1.4826 是正态分布下 MAD 与标准差的一致性缩放因子，
          确保在高斯噪声下 MAD ≈ std。
    ⚠️ 特殊处理：
        - 若输入信号为空，返回空 float32 数组；
        - 若 MAD == 0（即所有采样点值相同），视为无效信号，返回空数组。
    📌 输出始终为 float32（'f4'），以兼顾精度与内存效率，符合下游深度学习训练惯例。
    Args:
        signal (np.ndarray): 一维原始电流信号（单位：pA），形状为 (N,)。

    Returns:
        np.ndarray: 归一化后的信号，dtype=np.float32。
                    若信号无效（MAD=0 或空输入），返回 shape=(0,) 的空数组。
    """
    # 快速路径：空输入直接返回空 float32 数组
    if signal.size == 0:
        return np.array([], dtype=np.float32)
    # Step 1: 计算全局中位数（robust center）
    med = np.median(signal)
    # Step 2: 计算中位数绝对偏差（MAD），使用标准一致性因子 1.4826
    mad = 1.4826 * np.median(np.abs(signal - med))
    # Step 3: 安全检查 —— 零 MAD 表示无信号变化（如全零、常量），无法归一化
    mad = max(mad, 1.0)  # avoid division by near-zero
    # Step 4: 执行归一化并强制转换为 float32（节省内存，兼容 GPU 训练）
    normalized = (signal - med) / mad
    return normalized.astype(np.float32)

def _nanopore_normalize_novel(signal: np.ndarray) -> np.ndarray:
    """
    Normalize by subtracting global median and scaling with robust MAD
    estimated from central 98% of residuals (1st to 99th percentile).
    """
    signal_MED = np.median(signal)
    residual = signal - signal_MED

    # Use 1st and 99th percentiles to exclude extreme outliers
    q01, q99 = np.quantile(residual, [0.01, 0.99])
    masked_residual = residual[(residual >= q01) & (residual <= q99)]

    # Robust scale estimate (MAD)
    global_MAD = 1.4826 * np.median(np.abs(masked_residual))
    global_MAD = max(global_MAD, 1.0)  # avoid division by near-zero

    normalized = residual / global_MAD
    return normalized.astype(np.float32), global_MAD  # ✅ fixed variable name

def _nanopore_repair_errors(signal, min_value, max_value):
    """
    Fast version: only process outlier indices in increasing order.
    Uses the fact that cleaned[i] depends only on cleaned[i-1],
    and i-1 is always processed before i if we go left-to-right.
    """
    signal = np.asarray(signal, dtype=np.float32)
    if np.any(signal < min_value) or np.any(signal > max_value):
        do_repair = True
    else:
        do_repair = False
    if not do_repair:
        return signal

    cleaned = signal.copy()
    n = cleaned.size

    if n == 0:
        return cleaned

    # Find all outlier indices
    valid_mask = (cleaned >= min_value) & (cleaned <= max_value)
    outlier_indices = np.where(~valid_mask)[0]

    if outlier_indices.size == 0:
        return cleaned
    # Process outliers from left to right (they are already sorted)
    for i in outlier_indices:
        if i < 1:
            # First point: clamp
            if cleaned[0] > max_value:
                cleaned[0] = max_value
            else:
                cleaned[0] = min_value
        else:
            # Use immediate left neighbor (which is already final)
            cleaned[i] = cleaned[i - 1]
    return cleaned


def _nanopore_remove_spikes(
    signal,
    window_size=5001,
    spike_threshold=5.0
):
    """
    Detect and remove spikes using global MAD on baseline-removed residual.
    Spikes are repaired using forward-fill (left-to-right).
    
    Returns:
        cleaned: np.ndarray, repaired signal (same shape as input)
    """
    mad_factor=1.4826
    min_mad=1.0
    spike_threshold=5.0
    signal = np.asarray(signal, dtype=np.float32)
    
    # 1. Estimate baseline with median filter
    local_med = median_filter(signal, size=window_size, mode='reflect')
    
    # 2. Compute residual
    residual = signal - local_med
    
    # 3. Global MAD on residual
    global_mad = mad_factor * np.median(np.abs(residual))
    global_mad = max(global_mad, min_mad)
    
    # 4. Detect spikes
    is_spike = np.abs(residual) > (spike_threshold * global_mad)
    
    if not np.any(is_spike):
        return signal.copy()

    # 5. Repair spikes using forward-fill
    cleaned = signal.copy()
    outlier_indices = np.where(is_spike)[0]
    for i in outlier_indices:
        if i == 0:
            cleaned[0] = local_med[0]
        else:
            cleaned[i] = cleaned[i - 1]
    return cleaned

import numpy as np
from scipy.ndimage import median_filter

def _nanopore_repair_final(signal: np.ndarray, repair_threshold=3.0, repair_window_size=100) -> np.ndarray:
    """
    修复信号中的异常值，将超过阈值的点用附近窗口内的中值替代
    Args:
        signal (np.ndarray): 输入信号
        repair_threshold (float): 修复阈值，默认3.0
        repair_window_size (int): 修复窗口大小，默认100
        
    Returns:
        np.ndarray: 修复后的信号
    """
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size == 0:
        return signal
    
    # 计算信号的绝对值，找出超过阈值的位置
    abs_signal = np.abs(signal)
    outlier_mask = abs_signal > repair_threshold
    
    if not np.any(outlier_mask):
        return signal
    # 创建修复后的信号副本
    repaired_signal = signal.copy()
    # 使用median_filter计算每个点附近的中值
    # 注意：scipy的median_filter需要奇数窗口大小
    if repair_window_size % 2 == 0:
        actual_window_size = repair_window_size + 1
    else:
        actual_window_size = repair_window_size
    # 计算局部中值
    local_median = median_filter(signal, size=actual_window_size, mode='reflect')
    # 将异常值替换为其对应位置的局部中值
    repaired_signal[outlier_mask] = local_median[outlier_mask]
    return repaired_signal

import numpy as np

def _nanopore_truncate_signal(signal: np.ndarray, truncate_threshold=3.0) -> np.ndarray:
    """
    截断信号中绝对值大于阈值的部分
    
    Args:
        signal (np.ndarray): 输入信号
        truncate_threshold (float): 截断阈值，默认3.0
        
    Returns:
        np.ndarray: 截断后的信号
    """
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size == 0:
        return signal
    
    # 创建信号副本以避免修改原数组
    truncated_signal = signal.copy()
    
    # 找出绝对值大于阈值的位置
    mask = np.abs(truncated_signal) > truncate_threshold
    
    # 将这些位置的值截断到阈值范围内
    truncated_signal[mask] = np.clip(truncated_signal[mask], -truncate_threshold, truncate_threshold)
    
    return truncated_signal


def _nanopore_soft_clip_tanh(signal: np.ndarray, limit=3.0) -> np.ndarray:
    """
    使用 Tanh 函数对信号进行平滑压缩（软截断）。
    公式：f(x) = limit * tanh(x / limit)
    """
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size == 0:
        return signal
    
    # 执行软截断
    return (limit * np.tanh(signal / limit)).astype(np.float32)

def nanopore_process_signal(signal_raw,strategy="apple"):
    signal_return = None
    if strategy == "stone":
        signal_return = _nanopore_normalize_huada(signal_raw)
    elif strategy == "apple":
        signal_clear = _nanopore_repair_errors(signal_raw, 1, 220)
        signal_elite = _nanopore_remove_spikes(signal_clear, window_size=6000, spike_threshold=5.0)
        signal_nomal,_ = _nanopore_normalize_novel(signal_elite)
        signal_return = medfilt(signal_nomal, kernel_size=5).astype(np.float32)
    elif strategy == "lemon":
        signal_clear = _nanopore_repair_errors(signal_raw, 1, 220)
        signal_elite = _nanopore_remove_spikes(signal_clear, window_size=6000, spike_threshold=5.0)
        signal_nomal,_ = _nanopore_normalize_novel(signal_elite)
        signal_medfilt = medfilt(signal_nomal, kernel_size=5).astype(np.float32)
        signal_return = _nanopore_truncate_signal(signal_medfilt, truncate_threshold=5.0) 
    elif strategy == "tango":
        # 1. 基础修复与去刺（沿用 apple/lemon 的前处理）
        signal_clear = _nanopore_repair_errors(signal_raw, 1, 220)
        signal_elite = _nanopore_remove_spikes(signal_clear, window_size=6000, spike_threshold=5.0)
        # 2. 标准化（Median-MAD）
        signal_nomal,_ = _nanopore_normalize_novel(signal_elite)
        # 3. 降噪（中值滤波）
        # 如果你担心保真度，这里可以考虑将 kernel_size 改为 3
        # signal_medfilt = medfilt(signal_nomal, kernel_size=5).astype(np.float32)
        # 4. 核心替换：用 Tanh 软截断代替 Truncate
        # 此时信号超过 3 的部分会平滑靠拢，而不是被直接切平
        signal_return = _nanopore_soft_clip_tanh(signal_nomal,5.0)
    elif strategy == "mongo":
        # 1. 基础修复与去刺（沿用 apple/lemon 的前处理）
        signal_clear = _nanopore_repair_errors(signal_raw, 1, 220)
        signal_elite = _nanopore_remove_spikes(signal_clear, window_size=6000, spike_threshold=5.0)
        # 2. 标准化（Median-MAD）
        signal_nomal,_ = _nanopore_normalize_novel(signal_elite)
        # 3. 降噪（中值滤波）
        # 如果你担心保真度，这里可以考虑将 kernel_size 改为 3
        # signal_medfilt = medfilt(signal_nomal, kernel_size=5).astype(np.float32)
        # 4. 核心替换：用 Tanh 软截断代替 Truncate
        # 此时信号超过 3 的部分会平滑靠拢，而不是被直接切平
        signal_return = signal_nomal
    return signal_return