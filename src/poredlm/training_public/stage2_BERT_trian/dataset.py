import os
import glob
import gzip
import logging
import random
import torch
import numpy as np
from typing import List, Union
from torch.utils.data import IterableDataset, get_worker_info
import tqdm
logger = logging.getLogger(__name__)

class PoreSignalDataset(IterableDataset):
    """
    纳米孔电信号专用分布式流式加载器 (去表头·极致吞吐定型版)
    
    规范底座约束 (无表头纯数据行)：
    - 第 0 列: start (int)
    - 第 1 列: end (int)
    - 第 2 列: size (int)
    - 第 3 列: type (str)
    - 第 4 列: id (uint64)
    - 第 5 列: from (str)
    """
    # 🌟 静态常量定义：将列索引集中在头部，兼顾硬编码的极致性能与后期的修改便利
    IDX_START = 0
    IDX_END = 1
    IDX_ID = 4
    IDX_FROM = 5
    REQUIRED_MIN_COLUMNS = 6  # 严格防御：每行至少要有 6 列数据

    def __init__(
        self,
        shard_paths: Union[str, List[str]],
        logic_chunk_size: int = 6000,
        buffer_size: int = 20000,
        memmap_dtype: str = "float32",
        shuffle_buffer: bool = True,
        rank: int = 0,
        world_size: int = 1,
        is_repeat: bool = True,
        seed: int = 6198
    ):
        super().__init__()
        input_paths = [shard_paths] if isinstance(shard_paths, str) else shard_paths

        self.logic_chunk_size = logic_chunk_size
        self.buffer_size = buffer_size
        self.data_dtype = np.dtype(memmap_dtype) # 确保它是 numpy dtype 对象
        self.shuffle_buffer = shuffle_buffer
        self.rank = rank
        self.world_size = world_size
        self.is_repeat = is_repeat
        self.seed = seed

        self._mmap_cache = {}

        # 1. 智能化解析混合输入
        found_npy_files = []
        for path in input_paths:
            logger.info(f"looking from: {path}")
            print(f"looking from: {path}")
            if os.path.isdir(path):
                # 递归扫描所有 .npy
                found_npy_files.extend(glob.glob(os.path.join(path, "**/*.npy"), recursive=True))
            elif os.path.isfile(path) and path.endswith(".npy"):
                found_npy_files.append(path)
        # 核心逻辑：基于 .npy 查找对应的 .csv.gz
        self.all_csv_files = []
        for npy_path in sorted(list(set(found_npy_files))):
            csv_path = npy_path.replace(".npy", ".csv.gz")
            if os.path.exists(csv_path):
                self.all_csv_files.append(csv_path)
            else:
                logger.warning(f"⚠️ 发现 .npy 但找不到对应索引: {npy_path}，已跳过。")

        if len(self.all_csv_files) == 0:
            raise FileNotFoundError("❌ 未找到任何有效的 [.npy + .csv.gz] 匹配对！")

        # 2. 初始化阶段：秒级建立反查字典并统计行数（直接顺序扫描）
        if self.rank == 0:
            logger.info(f"💾 [Dataset Init] 正在极速扫描全局总行数并构建调试反查图谱...")

        self.total_rows_cached = 0
        for csv_path in tqdm.tqdm(self.all_csv_files, desc="Scanning Files"):
            try:
                with gzip.open(csv_path, 'rt', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        # 使用类常量进行严格边界防御
                        if len(parts) < self.REQUIRED_MIN_COLUMNS:
                            continue

                        chunk_id = int(parts[self.IDX_ID])
                        from_path = parts[self.IDX_FROM]

                        self.total_rows_cached += 1
            except Exception as e:
                logger.error(f"⚠️ 初始化解析文件失败: {csv_path}。错误: {e}")
                continue

        if self.rank == 0:
            logger.info(f"🛰️ [流式图谱构建成功] 总行数: {self.total_rows_cached}")

    def _get_mmap_stream_old(self, npy_path: str) -> np.memmap:
        if npy_path not in self._mmap_cache:
            file_size_bytes = os.path.getsize(npy_path)
            mmap_obj = np.memmap(npy_path, dtype=self.data_dtype, mode='r')
            expected_bytes = len(mmap_obj) * np.dtype(self.data_dtype).itemsize
            if file_size_bytes != expected_bytes:
                raise ValueError(f"❌ 物理文件大小校验失败！文件: {npy_path}")
            self._mmap_cache[npy_path] = mmap_obj
        return self._mmap_cache[npy_path]

    def _get_mmap_stream(self, npy_path: str) -> np.memmap:
        # 只缓存当前正在活跃使用的那一个 npy，换文件时自动释放老的
        if not hasattr(self, "_active_mmap_path") or self._active_mmap_path != npy_path:
            self._active_mmap_path = npy_path
            # 覆盖写入，老对象的引用计数归零，Python 会自动 close 释放句柄
            self._active_mmap_obj = np.memmap(npy_path, dtype=self.data_dtype, mode='r') 
        return self._active_mmap_obj

    def _get_pipeline_shards(self):
        if self.world_size > 1:
            rank_files = [f for i, f in enumerate(self.all_csv_files) if i % self.world_size == self.rank]
        else:
            rank_files = self.all_csv_files

        worker_info = get_worker_info()
        if worker_info is None:
            final_files = rank_files
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            final_files = [f for i, f in enumerate(rank_files) if i % num_workers == worker_id]

        if self.shuffle_buffer:
            worker_seed = self.seed + self.rank * 100 + (worker_info.id if worker_info else 0)
            random.seed(worker_seed)
            random.shuffle(final_files)

        # ---------------- 追踪代码开始 ----------------
        worker_id = worker_info.id if worker_info else 0
        file_names = [os.path.basename(f) for f in final_files]
        #logger.info(
        #    f"[Trace] Rank {self.rank} | Worker {worker_id} | "
        #    f"最终分片列表长度: {len(final_files)} | 顺序: {file_names}"
        #)
        # ---------------- 追踪代码结束 ----------------

        return final_files




    def _parse_and_yield(self, my_shards, rng):
        # worker_info = get_worker_info()
        # 当 num_workers=0 时，torch.utils.data.get_worker_info() 返回的是 None。
        # 你代码中直接访问 worker_info.id，在没有开启多进程加载（worker）的情况下，
        # 程序会直接因 AttributeError: 'NoneType' object has no attribute 'id' 而崩溃。
        # worker_id = worker_info.id if worker_info is not None else 0
        
        # 日志：记录当前任务开始加载分片
        # logger.info(f"[Trace] (Worker {worker_id}) 开始处理分片列表: {len(my_shards)} 个文件")
 
        # 建立当前 Epoch 内部的本地缓冲区
        buffer = []
        # 每一轮循环前，可以把当前分片顺序也随机打乱一下（推荐）
        shuffled_shards = list(my_shards)
        rng.shuffle(shuffled_shards)        
        for csv_path in shuffled_shards:
            # 日志：记录具体正在处理的 CSV 文件
            # logger.info(f"[Trace] (Worker {worker_id}) 正在打开: {os.path.basename(csv_path)}")
            
            npy_path = csv_path.replace(".csv.gz", ".npy")
            if not os.path.exists(npy_path):
                continue

            try:
                mmap_matrix = self._get_mmap_stream(npy_path)
            except Exception as e:
                print(e)
                continue

            try:
                with gzip.open(csv_path, 'rt', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        # 💡 极致简练的过滤：直奔主题，拒绝一切动态判断
                        if len(parts) < self.REQUIRED_MIN_COLUMNS:
                            print("Error f{csv_path}")
                            continue

                        start_idx = int(parts[self.IDX_START])
                        end_idx = int(parts[self.IDX_END])
                        chunk_id = int(parts[self.IDX_ID]) 

                        if end_idx <= start_idx or end_idx > mmap_matrix.shape[0]:
                            print("Error")
                            continue

                        # 极速内存映射切片
                        # 极速内存映射切片
                        # 移除 np.array(...) 转换，直接从 mmap 切片创建 tensor
                        # signal_slice = torch.from_numpy(np.array(mmap_matrix[start_idx:end_idx], dtype=np.float32))
                        # 这样可以保持原始数据的 dtype (uint16 或 float32)
                        raw_slice = mmap_matrix[start_idx:end_idx]
                        signal_slice = torch.from_numpy(raw_slice.copy()) # copy() 是必须的，因为 mmap 是只读的

                        if signal_slice.shape[0] < self.logic_chunk_size:
                            pad_len = self.logic_chunk_size - signal_slice.shape[0]
                            # 获取当前 slice 的类型,如果是float，填充0.0，如果是整数 现充0
                            pad_value = 0.0 if signal_slice.is_floating_point() else 0
                            signal_slice = torch.nn.functional.pad(signal_slice, (0, pad_len), mode='constant', value=pad_value)
                        elif signal_slice.shape[0] > self.logic_chunk_size:
                            signal_slice = signal_slice[:self.logic_chunk_size]

                        sample = {
                            #"labels": signal_slice,
                            "signal": signal_slice,
                            "id": torch.as_tensor(chunk_id, dtype=torch.int64)
                        }
                        
                        if self.shuffle_buffer and self.buffer_size > 0:
                            if len(buffer) < self.buffer_size:
                                buffer.append(sample)
                            else:
                                idx = rng.randint(0, len(buffer) - 1)
                                yield_sample = buffer[idx]
                                buffer[idx] = sample
                                yield yield_sample
                        else:
                            yield sample

            except Exception as e:
                print("Exception:",e)
                continue

        if self.shuffle_buffer and len(buffer) > 0:
            rng.shuffle(buffer)
            for remain_sample in buffer:
                yield remain_sample
        elif len(buffer) > 0:
            for remain_sample in buffer:
                yield remain_sample

    def __iter__(self):
        """策略控制层：在这里拦截无任务的 Worker，并全局控制随机种子"""
        # 1. 🌟 核心修复：在最外层分发分片
        my_shards = self._get_pipeline_shards()
        if not my_shards:
            # 如果这个 Worker 没分到文件，直接优雅退出，绝不进入 while True 空转！
            return 

        # 2. 🌟 核心修复：建立基础种子
        worker_info = get_worker_info()
        base_seed = self.seed + self.rank * 1000 + (worker_info.id if worker_info else 0)
        rng = random.Random(base_seed)

        if self.is_repeat:
            epoch = 0
            while True:
                # 3. 🌟 核心修复：每进入一轮新 Epoch，通过重设种子保证打乱顺序不同
                epoch_seed = base_seed + epoch * 555
                rng.seed(epoch_seed)
                yield from self._parse_and_yield(my_shards, rng)
                epoch += 1
        else:
            # 验证模式：带入 rng 跑一次，优雅结束
            yield from self._parse_and_yield(my_shards, rng)
    def __len__(self):
        """
        🚨 核心修复：不能返回全局总行数，否则多卡 DDP 训练时 Epoch 显示会变慢 world_size 倍。
        💡 修正：返回当前 Rank (单卡) 实际分摊到的预估样本数。
        """
        if self.world_size > 1:
            return self.total_rows_cached // self.world_size
        return self.total_rows_cached

    def get_source_path_by_id(self, chunk_id: int) -> str:
        return "Unknown_Source_Path"
