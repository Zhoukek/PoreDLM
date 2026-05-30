# nanopore_signal_tokenizer/vq_tokenizer.py
# Suppress known deprecation warnings

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*pkg_resources is deprecated.*",
    category=UserWarning,
    module="ont_fast5_api"
)

import os
import json
import gzip
import numpy as np
import torch
from math import ceil
from ont_fast5_api.fast5_interface import get_fast5_file
from utils.signal import nanopore_process_signal
from tqdm import tqdm
from scipy.signal import medfilt
import numpy as np
from typing import List

import numpy as np
from typing import List
import pyccf5 as slow5


from accelerate import Accelerator
# Import your model definition (must define NanoporeVQModel)
from training.stage1_tokenizer.tokenizer_model_v0 import Nanopore_Tokenizer_Model_V0
from training.stage1_tokenizer.tokenizer_model_v1 import Nanopore_Tokenizer_Model_V1

import torch.nn.functional as F


def load_accelerate_checkpoint(model_ckpt_dir: str):
    """
    从accelerate保存的目录中加载模型检查点
    """
    from accelerate import Accelerator
    from safetensors.torch import load_file
    import json

    # 直接从model.safetensors加载模型权重
    model_weights_path = os.path.join(model_ckpt_dir, "model.safetensors")
    if os.path.exists(model_weights_path):
        state_dict = load_file(model_weights_path, device="cpu")
    else:
        # 如果没有model.safetensors，尝试pytorch_model.bin
        model_bin_path = os.path.join(model_ckpt_dir, "pytorch_model.bin")
        if os.path.exists(model_bin_path):
            state_dict = torch.load(model_bin_path, map_location="cpu", weights_only=False)
        else:
            # 查找其他可能的模型权重文件
            model_files = [f for f in os.listdir(model_ckpt_dir)
                         if f.endswith(('.bin', '.safetensors')) and 'model' in f]
            if model_files:
                model_weights_path = os.path.join(model_ckpt_dir, model_files[0])
                if model_weights_path.endswith('.safetensors'):
                    state_dict = load_file(model_weights_path, device="cpu")
                else:
                    state_dict = torch.load(model_weights_path, map_location="cpu", weights_only=False)
            else:
                raise FileNotFoundError(f"No model weights file found in {model_ckpt_dir}")

    cnn_type = 0
    model_type = 0
    codebook_size = 0
    codebook_dim = 0
    metadata_path = os.path.join(model_ckpt_dir, "metadata.json")
    # 检查文件是否存在
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.json not found in {model_ckpt_dir}")

    # 尝试加载 JSON
    try:
        with open(metadata_path, 'r') as meta_f:
            metadata = json.load(meta_f)
    except (IOError, OSError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to read or parse metadata.json in {model_ckpt_dir}: {e}")

    # 定义必需字段
    required_keys = ['cnn_type', 'model_type', 'codebook_size', 'codebook_dim']
    missing_keys = [key for key in required_keys if key not in metadata]

    if missing_keys:
        raise KeyError(f"Missing required keys in metadata.json: {missing_keys}. Full path: {metadata_path}")

    # 安全提取必需字段（其实已确保存在，但保留 .get 可加类型检查）
    cnn_type = metadata['cnn_type']
    model_type = metadata['model_type']
    codebook_size = metadata['codebook_size']
    codebook_dim = metadata['codebook_dim']


    return {'model_state_dict': state_dict, 'cnn_type': cnn_type,'model_type':model_type,"codebook_size":codebook_size,"codebook_dim":codebook_dim}



class VQETokenizer:
    """
    Nanopore Single-Layer VQ Tokenizer.
    - Uses VectorQuantize (not RVQ)
    - No reconstruction loss needed
    - Designed for diversity + waveform backbone retention
    """

    def __init__(
        self,
        model_ckpt: str = "nanopore_vq_tokenizer.pth",
        device: str = "cuda",
        token_batch_size: int = 8000
    ):
        # --- Device setup ---
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = device.strip()
            if device.startswith("cuda"):
                if not torch.cuda.is_available():
                    print("⚠️ CUDA not available, falling back to CPU.")
                    self.device = "cpu"
                else:
                    self.device = device
            elif device == "cpu":
                self.device = "cpu"
            else:
                raise ValueError(f"Unsupported device: {device}")
        
        print(f"✅ Using device: {self.device}")

        # --- Load checkpoint ---
        print(f"📂 Loading checkpoint: {model_ckpt}")
        ckpt_data = load_accelerate_checkpoint(model_ckpt)
        # ✅ 1. 从 checkpoint 中读取 cnn_type（关键！）
        if 'cnn_type' not in ckpt_data:
            print("Checkpoint does not contain 'cnn_type'. forced to 0")
            cnn_type = 0
        else:
            cnn_type = ckpt_data['cnn_type']

        # ✅ 1. 从 checkpoint 中读取 cnn_type（关键！）
        if 'model_type' not in ckpt_data:
            print("Checkpoint does not contain 'cnn_type'. forced to 0")
            model_type = 0
        else:
            model_type = ckpt_data['model_type']


        # ✅ 1. 从 checkpoint 中读取 codebook_size（关键！）
        if 'codebook_size' not in ckpt_data:
            print("Checkpoint does not contain 'codebook_size'. forced to 0")
            codebook_size = 0
            raise RuntimeError(f"Unexpected codebook size: {codebook_size}")
        else:
            codebook_size = ckpt_data['codebook_size']

        if 'codebook_dim' not in ckpt_data:
            print("Checkpoint does not contain 'codebook_size'. forced to 0")
            codebook_dim = 0
            raise RuntimeError(f"Unexpected codebook dim: {codebook_dim}")
        else:
            codebook_dim = ckpt_data['codebook_dim']


        if codebook_size == 0:
            raise RuntimeError(f"Unexpected codebook size: {codebook_size}")

        ## ✅ 正确：从 model_state_dict 中找 codebook
        state_dict = ckpt_data['model_state_dict']

        self.codebook_size = codebook_size

        # --- Instantiate model ---
        if model_type == 0:
            self.model = Nanopore_Tokenizer_Model_V0(codebook_size=codebook_size, cnn_type=cnn_type)
        elif model_type == 1:
            self.model = Nanopore_Tokenizer_Model_V1(codebook_size=codebook_size, cnn_type=cnn_type, )
        else:
            raise RuntimeError(f"Unexpected model type: {model_type}")
        
        # 由CNN决定
        self.dim = self.model.codebook_dim
        print(f"🎯 Inferred: codebook_size={self.codebook_size}, dim={self.dim}, cnn_type={cnn_type}")

        if not hasattr(self.model, 'cnn_stride'):
            raise AttributeError("Model must define 'cnn_stride' (total downsampling rate).")
        if not hasattr(self.model, 'margin_stride_count'):
            self.model.margin_stride_count = 2  # default fallback

        self.downsample_rate = self.model.cnn_stride
        self.margin_stride_count = self.model.margin_stride_count
        self.margin = self.margin_stride_count * self.downsample_rate
        self.model_RF = self.model.RF
        if token_batch_size < 1:
            token_batch_size = 1
        self.chunk_size = token_batch_size * self.downsample_rate

        # --- Load state dict ---
        self.model.load_state_dict(state_dict, strict=False)

        print(self.model)
        self.model.eval()
        self.model.to(self.device)

        print("\n✅ VQTokenizer initialized:")
        print(f"   Checkpoint       : {os.path.abspath(model_ckpt)}")
        print(f"   Device           : {self.device}")
        print(f"   Model type       : {model_type}")
        print(f"   Codebook size    : {self.codebook_size}")
        print(f"   Latent dim       : {self.dim}")
        print(f"   Downsample rate  : {self.downsample_rate}")
        print(f"   Chunk size       : {self.chunk_size}")
        print(f"   Margin           : {self.margin} samples")
        print("-" * 60)
    

    def _tokenize_chunked_signal(self, signal: np.ndarray, layer: int = 0) -> np.ndarray:
        """
        Tokenize 1D signal using sliding window with margin.
        
        Args:
            signal (np.ndarray): 1D raw signal array.
            layer (int): Number of quantizer layers to use for tokenization.
                         - 0: use all layers (default)
                         - 1: use first layer only
                         - 2: use first two layers, etc.

        Returns:
            np.ndarray: Token ID sequence of shape [T_expected], dtype=np.int64
        """
        if signal.ndim != 1:
            raise ValueError("Signal must be 1D.")
        L = len(signal)
        if L < self.model_RF or L == 0:
            return np.array([], dtype=np.int64)

        T_expected = (L + self.downsample_rate - 1) // self.downsample_rate

        # Safely access the underlying model (handle DDP)
        raw_model = self.model.module if hasattr(self.model, 'module') else self.model

        if L <= self.chunk_size:
            padded = np.pad(signal, (0, self.chunk_size - L), mode='constant')
            x = torch.from_numpy(padded).float().unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(x)
                # Assume: (recon, level_indices, loss, ...)
                level_indices = outputs[1]  # [B, N, K]

            tokens_tensor = raw_model.tokenize_indices(level_indices, layer=layer)
            tokens = tokens_tensor[0].cpu().numpy().astype(np.int64)  # [T]
            return tokens[:T_expected].astype(np.int64)

        # Long signal: sliding window
        margin_samples = self.margin
        step_samples = self.chunk_size - 2 * margin_samples
        if step_samples <= 0:
            raise ValueError("chunk_size too small for margin.")

        all_tokens = []
        start = 0
        chunk_index = 0

        while start < L:
            real_len = min(self.chunk_size, L - start)
            chunk = signal[start:start + real_len]
            if len(chunk) < self.chunk_size:
                chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)), mode='constant')
            x = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(x)
                level_indices = outputs[1]  # [B, N, K]

            tokens_tensor = raw_model.tokenize_indices(level_indices, layer=layer)
            tokens = tokens_tensor[0].cpu().numpy().astype(np.int64)  # [T]
            T_valid = (real_len + self.downsample_rate - 1) // self.downsample_rate

            kept_tokens = np.array([], dtype=np.int64)
            if chunk_index == 0:
                end_idx = T_valid - self.margin_stride_count if self.margin_stride_count > 0 else T_valid
                kept_tokens = tokens[:max(0, end_idx)]
            elif start + step_samples >= L:
                start_idx = self.margin_stride_count
                max_len = T_valid - self.margin_stride_count
                if max_len > 0:
                    kept_tokens = tokens[start_idx : start_idx + max_len]
            else:
                if self.margin_stride_count > 0 and len(tokens) > 2 * self.margin_stride_count:
                    max_len = T_valid - 2 * self.margin_stride_count
                    if max_len > 0:
                        kept_tokens = tokens[
                            self.margin_stride_count : self.margin_stride_count + max_len
                        ]
                else:
                    kept_tokens = tokens[:T_valid]

            if kept_tokens.size > 0:
                all_tokens.append(kept_tokens)

            start += step_samples
            chunk_index += 1

        if not all_tokens:
            return np.zeros(T_expected, dtype=np.int64)

        final_tokens = np.concatenate(all_tokens, axis=0)
        if len(final_tokens) > T_expected:
            final_tokens = final_tokens[:T_expected]
        elif len(final_tokens) < T_expected:
            final_tokens = np.pad(final_tokens, (0, T_expected - len(final_tokens)), constant_values=0)

        return final_tokens.astype(np.int64)


    def tokenize_signal_batched(self, 
                           signal: np.ndarray, 
                           signal_chunk_size: int, 
                           signal_chunk_overlap_size: int, 
                           max_batch_size: int) -> List[np.ndarray]:
        """
        将信号严格切分为等长块（不足长度的直接丢弃），并按批次进行推理。
        使用 extend，返回的是一维列表（所有块的 tokens 连在一起）。
        
        Args:
            signal: 输入的一维信号数组。
            signal_chunk_size: 每个块的严格大小。
            signal_chunk_overlap_size: 块之间的重叠大小。
            max_batch_size: 每个推理批次的最大块数量。
            
        Returns:
            List[np.ndarray]: 一维列表。每个元素是单个块的推理结果 (tokens)。
                             (注意：不再是二维列表，所有批次的数据都被合并到了同一个列表中)
        """
        raw_model = self.model.module if hasattr(self.model, 'module') else self.model
    
        if signal.ndim != 1:
            raise ValueError("Signal must be 1D.")
        
        L = len(signal)
        batched_results = [] # 初始化结果列表
        
        # --- 情况 1: 信号总长度小于一个块的大小 ---
        if L < signal_chunk_size:
            return batched_results

        # --- 情况 2: 信号足够长，进行严格切分 ---
        step_size = signal_chunk_size - signal_chunk_overlap_size
        if step_size <= 0:
            raise ValueError("signal_chunk_size must be greater than signal_chunk_overlap_size.")
        
        # 1. 第一阶段：严格切分 (不填充)
        chunks = []
        start = 0    
        while start + signal_chunk_size <= L:
            chunk = signal[start : start + signal_chunk_size]
            chunks.append(chunk)
            start += step_size

        if not chunks:
            return batched_results

        # 2. 第二阶段：批量推理
        for i in range(0, len(chunks), max_batch_size):
            batch_chunks = chunks[i : i + max_batch_size]
            
            # --- 核心推理代码 ---
            batch_np = np.array(batch_chunks)
            x = torch.from_numpy(batch_np).float().unsqueeze(1).to(self.device)
            with torch.no_grad():
                # 模型推理，得到重构信号、层级索引等
                # level_indices 的形状是 [批次大小(Batch_Size), 序列长度(N), 层数(K)]
                recon, level_indices, _, _ = self.model(x) 
            

            # 使用模型的 tokenize_indices 方法将层级索引转换为单一的token ID序列
            # 返回的 tokens_tensor 形状是 [批次大小(Batch_Size), 序列长度(N)]
            tokens_tensor = raw_model.tokenize_indices(level_indices, layer=layer)
            
            # 将整个批次的 tokens 张量转换为 numpy 数组
            # tokens_np_batch 的形状是 [批次大小(Batch_Size), 序列长度(N)]
            tokens_np_batch = tokens_tensor.cpu().numpy().astype(np.int64)  
           
            # --- 修改点：使用 extend 将批次结果添加到总列表 ---
            # tokens_np_batch 是一个二维数组，形状为 [Batch_Size, N]
            # extend() 函数会遍历这个二维数组的第一个维度（也就是批次维度）
            # 将 tokens_np_batch[0] (第一个块的tokens, 形状[N]),
            #   tokens_np_batch[1] (第二个块的tokens, 形状[N]),
            #   ... 依次添加到 batched_results 列表中
            # 这样，一个批次内的 Batch_Size 个块的 tokens 都会被添加进列表，
            # 而不是只添加第一个块的 tokens。
            batched_results.extend(tokens_np_batch) 
            
        return batched_results # 返回 List[np.ndarray] (一维)


    # tokenize_data不支持任何归一化, medf, lpf等操作
    def tokenize_data(self, signal: np.ndarray,layer: int = 0) -> list:
        flat_tokens = self._tokenize_chunked_signal(signal,layer)
        if flat_tokens.size == 0:
            return []
        parts = []
        for token_id in flat_tokens:
            parts.append(f"<|bwav:{int(token_id)}|>")
        return parts

    # tokenize_data不支持任何归一化, medf, lpf等操作
    def tokenize_chunk(self, signal: np.ndarray) -> list:
        flat_tokens = self._tokenize_chunked_signal(signal)
        if flat_tokens.size == 0:
            return []
        parts = []
        for token_id in flat_tokens:
            parts.append(f"<|bwav:{int(token_id)}|>")
        return parts



    def tokenize_read(self, read, nanopore_signal_process_strategy="apple") -> list:
        try:
            channel_info = read.handle[read.global_key + 'channel_id'].attrs
            offset = int(channel_info['offset'])
            scaling = channel_info['range'] / channel_info['digitisation']
            raw = read.handle[read.raw_dataset_name][:]
            signal_raw = np.array(scaling * (raw + offset), dtype=np.float32)
            signal_processed = nanopore_process_signal(signal_raw,nanopore_signal_process_strategy)
            return self.tokenize_data(signal_processed)
        except Exception as e:
            fast5_path = getattr(read.handle, 'filename', 'unknown.fast5')
            print(f"❌ Error on read {read.read_id} in {fast5_path}: {e}")
            return []
    

    def tokenize_fast5(self, fast5_path: str, output_path:str, nanopore_signal_process_strategy="apple"):
        print(f"✅ Processing {fast5_path} with strategy{nanopore_signal_process_strategy}")
        results = []
        with get_fast5_file(fast5_path, mode="r") as f5:
            for read in tqdm(f5.get_reads(), desc=os.path.basename(fast5_path)):
                try:
                    token_list = self.tokenize_read(read,nanopore_signal_process_strategy)
                    token_str = "".join(token_list)
                    results.append({"id": read.read_id, "text": token_str})
                except Exception as e:
                    print(f"❌ Failed on read {read.read_id}: {e}")
                    continue

        with gzip.open(output_path, 'wt', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item) + '\n')
        print(f"✅ Wrote {len(results)} reads to {output_path}")

    def tokenize_data_batched(self, 
                 signal: np.ndarray, 
                 signal_chunk_size: int, 
                 signal_chunk_overlap_size: int, 
                 max_batch_size: int,
                 chunk_token_count: int) -> list:
        """
        使用批量推理函数获取 tokens，并进行严格校验与拼接。
        
        Args:
            signal: 输入信号
            signal_chunk_size: 信号块大小
            signal_chunk_overlap_size: 重叠大小
            max_batch_size: 最大批大小
            chunk_token_count: 期望每个 chunk 输出的 token 数量 (用于校验)
            
        Returns:
            list: 每个元素是一个字符串，由符合长度要求的 chunk tokens 拼接而成
        """
        
        # 调用批量处理函数
        chunks_tokens_list = self.tokenize_signal_batched(
            signal=signal,
            signal_chunk_size=signal_chunk_size,       
            signal_chunk_overlap_size=signal_chunk_overlap_size, 
            max_batch_size=max_batch_size
        )
        
        # 如果没有结果，返回空列表
        if not chunks_tokens_list:
            return []

        string_parts = []
        for chunk_tokens in chunks_tokens_list:
            # --- 新增校验逻辑 ---
            # 检查当前 chunk 的 token 数量是否符合预期
            if len(chunk_tokens) != chunk_token_count:
                # 如果长度不符，可以选择跳过(this)、填充或报错
                # 这里选择跳过，不加入结果列表
                continue 
            
            # 1. 将每个 token ID 转换为字符串格式
            token_strings = [f"<|bwav:{int(token_id)}|>" for token_id in chunk_tokens]
            # 2. 使用 "".join() 拼接
            joined_string = "".join(token_strings)
            string_parts.append(joined_string)
        return string_parts

    def tokenize_read_batched(self, read, 
        nanopore_signal_process_strategy:str="apple",
        # 新增参数：用于传递给 tokenize_data
        signal_chunk_size: int = 40000,
        signal_chunk_overlap_size: int = 10000,
        max_batch_size: int = 32,
        chunk_token_count: int = 8000 # 用于内部校验
    ) -> list:
        try:
            channel_info = read.handle[read.global_key + 'channel_id'].attrs
            offset = int(channel_info['offset'])
            scaling = channel_info['range'] / channel_info['digitisation']
            raw = read.handle[read.raw_dataset_name][:]
            signal_raw = np.array(scaling * (raw + offset), dtype=np.float32)
            signal_processed = nanopore_process_signal(signal_raw,nanopore_signal_process_strategy)
            return self.tokenize_data_batched(signal_processed,signal_chunk_size,signal_chunk_overlap_size,max_batch_size,chunk_token_count)
        except Exception as e:
            fast5_path = getattr(read.handle, 'filename', 'unknown.fast5')
            print(f"❌ Error on read {read.read_id} in {fast5_path}: {e}")
            return []



    def tokenize_fast5_batched(self,
        fast5_path: str,
        output_path: str,
        nanopore_signal_process_strategy="apple",
        # 新增参数：用于传递给 tokenize_data
        signal_chunk_size: int = 40000,
        signal_chunk_overlap_size: int = 10000,
        max_batch_size: int = 32,
        chunk_token_count: int = 8000 # 用于内部校验
    ):
        print(f"✅ Processing {fast5_path} with strategy {nanopore_signal_process_strategy}")
        results = []

        with get_fast5_file(fast5_path, mode="r") as f5:
            for read in tqdm(f5.get_reads(), desc=os.path.basename(fast5_path)):
                try:
                    # 调用 tokenize_data，传入所有必要的参数
                    # 返回的是 List[str]，例如 ["<|bwav:1|><|bwav:2|>", "<|bwav:3|><|bwav:4|>"]
                    chunked_token_strings = self.tokenize_read_batched(
                        read=read,
                        signal_chunk_size=signal_chunk_size,
                        signal_chunk_overlap_size=signal_chunk_overlap_size,
                        max_batch_size=max_batch_size,
                        chunk_token_count=chunk_token_count
                    )
                    # --- 修改开始 ---
                    # 循环每个分块字符串，作为独立行追加
                    for chunk_token_str in chunked_token_strings:
                        results.append({
                            "id": read.read_id, 
                            "text": chunk_token_str
                        })
                    # --- 修改结束 ---
                    # 将多个 chunk 的字符串用空格（或其他分隔符）连接成一个完整的字符串
                    # 如果不需要分隔符，使用 ""；如果需要区分 chunk 边界，建议使用 " " 或 "<|chunk_end|>"
                except Exception as e:
                    print(f"❌ Failed on read {read.read_id}: {e}")
                    continue

        # 写入文件
        with gzip.open(output_path, 'wt', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item) + '\n')
        print(f"✅ Wrote {len(results)} reads to {output_path}")


    def _get_codebook_embed(self) -> torch.Tensor:
        """
        获取 codebook embedding，统一成形状 [K, D]
        K = codebook_size, D = latent_dim
        """
        embed = self.model.vq._codebook.embed

        # 可能是 [1, K, D]，也可能是 [K, D]
        if embed.ndim == 3:
            embed = embed[0]
        elif embed.ndim != 2:
            raise RuntimeError(f"Unexpected codebook embed shape: {embed.shape}")

        return embed

    def decode_token_ids(
        self,
        token_ids,
        target_signal_len: int | None = None,
        return_numpy: bool = True,
    ):
        """
        将 token ids 重建为近似信号。

        Args:
            token_ids:
                - list[int]                -> 单条 token 序列
                - np.ndarray [T]           -> 单条 token 序列
                - np.ndarray [B, T]        -> batch token 序列
                - torch.Tensor [T] / [B,T]
            target_signal_len:
                - 如果提供，则输出裁剪/补零到这个长度
                - 如果不提供，则默认输出长度约为 T * downsample_rate
            return_numpy:
                - True: 返回 numpy
                - False: 返回 torch.Tensor

        Returns:
            shape:
                - 单条输入: [L]
                - batch输入: [B, L]
        """
        # 1) 规范化输入 shape -> [B, T]
        if isinstance(token_ids, list):
            token_ids = np.asarray(token_ids, dtype=np.int64)
        elif isinstance(token_ids, np.ndarray):
            token_ids = token_ids.astype(np.int64)
        elif isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().long().numpy()
        else:
            raise TypeError(f"Unsupported token_ids type: {type(token_ids)}")

        single_input = False
        if token_ids.ndim == 1:
            token_ids = token_ids[None, :]
            single_input = True
        elif token_ids.ndim != 2:
            raise ValueError(f"token_ids must be 1D or 2D, got shape {token_ids.shape}")

        # 2) 转 tensor
        token_ids_t = torch.from_numpy(token_ids).long().to(self.device)   # [B, T]

        # 3) 边界检查
        if token_ids_t.numel() == 0:
            raise ValueError("Empty token_ids.")
        if token_ids_t.min() < 0 or token_ids_t.max() >= self.codebook_size:
            raise ValueError(
                f"token id out of range. valid=[0, {self.codebook_size - 1}], "
                f"got min={int(token_ids_t.min())}, max={int(token_ids_t.max())}"
            )

        # 4) codebook lookup: [B, T] -> [B, T, D]
        codebook = self._get_codebook_embed().to(self.device)  # [K, D]
        z_q = codebook[token_ids_t]                            # [B, T, D]

        # 5) decoder 需要 [B, C, T]
        z_q = z_q.permute(0, 2, 1).contiguous()               # [B, D, T]

        # 6) decode
        with torch.no_grad():
            recon = self.model.decode(z_q)                   # [B, 1, L]

        recon = recon.squeeze(1)                              # [B, L]

        # 7) 长度对齐
        if target_signal_len is None:
            # 默认按 stride 反推近似长度
            target_signal_len = token_ids.shape[1] * self.downsample_rate

        current_len = recon.shape[-1]
        if current_len > target_signal_len:
            recon = recon[..., :target_signal_len]
        elif current_len < target_signal_len:
            recon = F.pad(recon, (0, target_signal_len - current_len))

        if return_numpy:
            recon = recon.detach().cpu().numpy()
            if single_input:
                return recon[0]
            return recon

        if single_input:
            return recon[0]
        return recon

    def parse_token_string(self, text: str) -> np.ndarray:
        """
        把 <|bwav:12|><|bwav:31|>... 解析成 int 数组
        """
        ids = re.findall(r"<\|bwav:(\d+)\|>", text)
        return np.asarray([int(x) for x in ids], dtype=np.int64)


    def tokenize_ccf5(self, ccf5_path: str, output_path:str,nanopore_signal_process_strategy="apple"):
        print(f"✅ Process {ccf5_path}")
        """内部方法：处理单个 FAST5 → JSONL.GZ"""
        results = []
        s5 = slow5.Open(ccf5_path, 'r', DEBUG=0)
        all_read_ids = s5.get_read_ids()[0]
        reads = s5.get_read_list_multi(all_read_ids, threads=1, batchsize=1, aux='all')
        for read in reads:
            query_name = read['read_id']
            if "lvdsmid" in read:
                signal = (read['signal'] - read['lvdsmid']) * read['unit']
            else:
                signal = read['K'] * read['scale'] * (read["signal"].astype(np.uint16) + read['offset']) + read['B']

            signal_processed = nanopore_process_signal(signal,nanopore_signal_process_strategy)
            token_str = self.tokenize_data(signal_processed)
            results.append({
                "id": query_name,
                "text": token_str
            })
        s5.close()
        # Save
        with gzip.open(output_path, 'wt', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item) + '\n')
        print(f"✅ Wrote {len(results)} reads to {output_path}")