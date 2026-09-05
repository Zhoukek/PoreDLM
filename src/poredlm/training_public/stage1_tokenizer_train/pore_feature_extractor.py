import numpy as np
import torch
from typing import Union, List, Dict, Any, Optional
from scipy.ndimage import median_filter
from scipy.signal import medfilt
from transformers import SequenceFeatureExtractor

class PoreFeatureExtractor(SequenceFeatureExtractor):
    """
    PoreFeatureExtractor: A unified Hugging Face-compatible feature extractor tailored for nanopore electrical signals.
    Supports multiple industrial-grade preprocessing pipelines via the `strategy` parameter:
    
    1. 'mongo' pipeline:
        - Physical boundaries validation & patch correction (_repair_errors)
        - Dynamic despiking based on baseline residuals and local median filtering (_remove_spikes)
        - Robust Median-MAD normalization excluding 1% and 99% extreme outliers (_normalize_novel)
        
    2. 'apple' pipeline (Default):
        - Includes all components of the 'mongo' pipeline.
        - Appends a high-frequency localized denoising layer via a continuous 1D median filter (medfilt).

    3. 'stone' pipeline:
        - Applies only global Median-MAD normalization.
        - Matches `nanopore_process_signal(..., strategy="stone")` in `poredlm.utils.signal`.
    """
    model_input_names = ["signal"]

    def __init__(
        self, 
        feature_size: int = 1, 
        sampling_rate: int = 4000, 
        padding_value: float = 0.0, 
        strategy: str = "apple",
        sclamp_linear_bound: float = 5.0,
        sclamp_target_bound: float = 6.0,
        **kwargs
    ):
        strategy = strategy.lower()
        self.strategy = strategy
        self.sclamp_linear_bound = sclamp_linear_bound
        self.sclamp_target_bound = sclamp_target_bound

        super().__init__(
            feature_size=feature_size, 
            sampling_rate=sampling_rate, 
            padding_value=padding_value, 
            strategy=strategy,
            sclamp_linear_bound=sclamp_linear_bound,
            sclamp_target_bound=sclamp_target_bound,
            **kwargs
        )

        # 统一转化为小写，防止大小写输入错误，并进行策略校验
        supported_strategies = ["apple", "mongo", "stone"]
        if strategy.lower() not in supported_strategies:
            raise ValueError(
                f"Unsupported strategy: '{strategy}'. Choose from {supported_strategies}."
            )
        self.strategy = strategy


    def _sclamp_bounds(self, x: np.ndarray) -> np.ndarray:
        """
        In-place modification of the input NumPy array:
        - Maintains absolute linearity within [-self.sclamp_linear_bound, self.sclamp_linear_bound].
        - Smoothly compresses values outside this range using a tanh function.

        ALPHA is set to 1.0 as a constant to ensure C2 continuity
        (perfect first and second-order derivative smoothness) at the transition boundary.
        """
        ALPHA = 1.0
        linear_bound = self.sclamp_linear_bound
        target_bound = self.sclamp_target_bound
        delta = target_bound - linear_bound

        # Skip operation if boundaries are invalid or clamping is disabled
        if delta <= 0 or linear_bound <= 0:
            return x

        mask_upper = x > linear_bound
        mask_lower = x < -linear_bound

        if np.any(mask_upper):
            # Apply tanh smoothing for values exceeding the upper bound
            x[mask_upper] = linear_bound + delta * np.tanh(ALPHA * (x[mask_upper] - linear_bound) / delta)

        if np.any(mask_lower):
            # Apply tanh smoothing for values exceeding the lower bound
            x[mask_lower] = -linear_bound + delta * np.tanh(ALPHA * (x[mask_lower] + linear_bound) / delta)

        return x



    def _repair_errors(self, signal: np.ndarray, min_value: float = 1.0, max_value: float = 220.0) -> np.ndarray:
        """Fast physical boundary validation: out-of-bounds points are safely forward-filled."""
        signal = np.asarray(signal, dtype=np.float32)
        if not (np.any(signal < min_value) or np.any(signal > max_value)):
            return signal

        cleaned = signal.copy()
        n = cleaned.size
        if n == 0:
            return cleaned

        valid_mask = (cleaned >= min_value) & (cleaned <= max_value)
        outlier_indices = np.where(~valid_mask)[0]

        if outlier_indices.size == 0:
            return cleaned

        for i in outlier_indices:
            if i < 1:
                cleaned[0] = max_value if cleaned[0] > max_value else min_value
            else:
                cleaned[i] = cleaned[i - 1]
        return cleaned

    def _remove_spikes(self, signal: np.ndarray, window_size: int = 6000, spike_threshold: float = 5.0) -> np.ndarray:
        """Detects and repairs transient current spike noises dynamically using baseline residuals and global MAD."""
        mad_factor = 1.4826
        min_mad = 1.0
        signal = np.asarray(signal, dtype=np.float32)

        n = signal.size

        # 🚀 核心修复：防止底层 C 语言因为 mode='reflect' 发生内存越界崩溃
        # 强制 window_size 最大不超过信号长度，且最好保持为奇数（减 1 或 2）
        actual_window = min(window_size, n)
        if actual_window % 2 == 0:
            actual_window -= 1
        
        # 如果信号短到无法滤波（比如小于3个点），直接跳过
        if actual_window < 3:
            return signal.copy()

        local_med = median_filter(signal, size=actual_window, mode='reflect')
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

    def _normalize_novel(self, signal: np.ndarray) -> np.ndarray:
        """Estimates high-fidelity global Median-MAD parameters by truncating outer 2% extreme tails."""
        signal_MED = np.median(signal)
        residual = signal - signal_MED

        q01, q99 = np.quantile(residual, [0.01, 0.99])
        masked_residual = residual[(residual >= q01) & (residual <= q99)]

        global_MAD = 1.4826 * np.median(np.abs(masked_residual))
        global_MAD = max(global_MAD, 1.0)

        normalized = residual / global_MAD
        return normalized.astype(np.float32)

    def _normalize_stone(self, signal: np.ndarray) -> np.ndarray:
        """Apply the global Median-MAD normalization used by the stone pipeline."""
        signal_median = np.median(signal)
        global_mad = 1.4826 * np.median(np.abs(signal - signal_median))
        global_mad = max(global_mad, 1.0)
        normalized = (signal - signal_median) / global_mad
        return normalized.astype(np.float32)

    def __call__(
        self,
        raw_signals: Union[torch.Tensor, np.ndarray, List[np.ndarray]],
        padding: bool = True,
        return_tensors: Optional[str] = "pt",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main execution entry point for feature processing pipelines.
        """
        # 1. 统一输入格式为 List[np.ndarray]
        if isinstance(raw_signals, torch.Tensor):
            raw_signals = raw_signals.cpu().numpy()

        if isinstance(raw_signals, np.ndarray):
            if raw_signals.ndim == 1:
                raw_signals = [raw_signals]
            elif raw_signals.ndim == 2:
                raw_signals = [x for x in raw_signals]
        elif not isinstance(raw_signals, list):
            raw_signals = [np.asarray(raw_signals)]

        # 2. 序列循环处理
        processed_signals = []
        for signal in raw_signals:
            sig_arr = np.asarray(signal, dtype=np.float32)
            if sig_arr.size == 0:
                processed_signals.append(np.array([], dtype=np.float32))
                continue

            if self.strategy == "stone":
                # Keep this branch identical to poredlm.utils.signal's stone
                # preprocessing: no repair, despiking, filtering, or clipping.
                sig_final = self._normalize_stone(sig_arr)
            else:
                # Common apple/mongo pipeline: repair, despike, and robustly normalize.
                sig_clear = self._repair_errors(sig_arr, min_value=1.0, max_value=220.0)
                sig_elite = self._remove_spikes(sig_clear, window_size=6000, spike_threshold=5.0)
                sig_normal = self._normalize_novel(sig_elite)

                # Apple additionally applies a localized median filter.
                if self.strategy == "apple":
                    sig_final = medfilt(sig_normal, kernel_size=5).astype(np.float32)
                else:
                    sig_final = sig_normal

                # Smooth clipping belongs only to the apple/mongo pipelines.
                if self.sclamp_linear_bound > 1e-6:
                    sig_final = self._sclamp_bounds(sig_final)

            processed_signals.append(sig_final)

        # 3. 动态长度对齐与零填充 (Padding)
        max_len = max(len(s) for s in processed_signals) if processed_signals else 0

        batch_signals = []
        for s in processed_signals:
            if padding and len(s) < max_len:
                padded = np.pad(s, (0, max_len - len(s)), mode='constant', constant_values=self.padding_value)
                batch_signals.append(padded)
            else:
                batch_signals.append(s)

        # 4. 组装成下游 CNN 要求的 3D 矩阵形状: [Batch, Channel=1, Sequence_Length]
        batch_array = np.stack(batch_signals, axis=0) if batch_signals else np.empty((0, 0), dtype=np.float32)
        batch_array = np.expand_dims(batch_array, axis=1)

        if return_tensors == "pt":
            return {"signal": torch.from_numpy(batch_array).float()}
        return {"signal": batch_array}
