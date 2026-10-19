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
from collections import OrderedDict
logger = logging.getLogger(__name__)


"""
================================================================================
FLOWMAP DATA PROCESSING SPECIFICATION (Best Practices)
================================================================================

FlowMap 采用了高性能的“裸二进制(Raw Binary) + CSV索引”模式。为了确保 
极高的加载吞吐量，请务必遵循以下格式协议，严禁在 token 数据中使用 
numpy.save() 格式（因为它包含 Header 会导致偏移对齐错误）。



--- 1. 数据存储规范 (Data Production) ---
使用 .tofile() 存储 token 序列。这会直接写入原始二进制流，无 Header。

    # 实例代码：
    subset_tokens = np.zeros(total_len, dtype=np.uint16)
    # ... (数据填充逻辑) ...
    subset_tokens.tofile("data_uint16.bin") # 必须明确后缀为 .bin

--- 2. 数据读取规范 (Data Consumption) ---
读取时，必须通过显式指定 dtype 映射二进制流。

    # 实例代码：
    # 裸读取，无 Header，速度极快
    verified_tokens = np.fromfile("data_uint16.bin", dtype=np.uint16)
    
    # 结合 memmap 实现按需读取（推荐用于超大规模数据集）
    # mode='r' 映射磁盘文件，不占用物理内存
    mmap_matrix = np.memmap("data_uint16.bin", dtype=np.uint16, mode='r')

--- 3. 校验防错指南 ---
由于是裸二进制格式，请务必在生成数据时执行边界检查：
    if seq[0] != 2 or seq[-1] != 3:
        raise ValueError("边界异常：Token 格式必须以 2 开头，3 结尾")

================================================================================
"""


def resolve_input_paths(input_paths: list) -> list:
    """
    Resolves input paths to a list of .npy files based on the following logic:
    - If the path contains glob patterns, resolve them directly.
    - If the path is a directory, recursively scan for all .npy files.
    - If the path is a file, validate its extension and include it.
    """
    resolved_files = []
    # Define glob meta-characters to identify pattern strings
    glob_chars = {'*', '?', '[', ']'}

    for path in input_paths:
        # Check if the input is a pattern
        is_pattern = any(char in path for char in glob_chars)
        
        if is_pattern:
            matches = glob.glob(path, recursive=True)
            resolved_files.extend(matches)
            logger.info(f"Resolved pattern '{path}' to {len(matches)} files.")
            
        elif os.path.isdir(path):
            # Treat directories as a recursive scan for .npy files
            pattern = os.path.join(path, "**/*.npy")
            matches = glob.glob(pattern, recursive=True)
            resolved_files.extend(matches)
            logger.info(f"Scanned directory '{path}' and found {len(matches)} files.")
            
        elif os.path.isfile(path):
            # Validate individual file extensions
            if path.endswith(".npy"):
                resolved_files.append(path)
            else:
                logger.warning(f"Skipping non-npy file: {path}")
        else:
            logger.warning(f"Invalid path skipped: {path}")

    # Deduplicate and sort to ensure deterministic behavior across distributed workers
    return list(set(resolved_files))

class LRUMMapCache:
    """
    A Least Recently Used (LRU) cache for memory-mapped files.
    Ensures memory efficiency and avoids exceeding file descriptor limits.
    """
    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, path: str, dtype: np.dtype):
        """
        Retrieves an np.memmap object for the given path.
        If the file is already cached, it is moved to the end as most recently used.
        If the cache is full, the least recently used entry is evicted.
        """
        # If cache hit, move to end to mark as most recently used
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        
        # If cache is full, remove the least recently used item (first in OrderedDict)
        if len(self.cache) >= self.capacity:
            old_path, old_mmap = self.cache.popitem(last=False)
            logger.debug(f"Cache capacity reached, releasing mmap for: {old_path}")
            # Explicitly delete reference to trigger file handle closure
            del old_mmap
            
        # Create a new memory map and store it in the cache
        mmap_obj = np.memmap(path, dtype=dtype, mode='r')
        self.cache[path] = mmap_obj
        
        return mmap_obj

class FlowMapDataset(IterableDataset):
    """
    High-performance distributed streaming loader (headerless, optimized for extreme throughput).

    Standard constraints (headerless pure data rows):
    - Column 0: start (int)
    - Column 1: end (int)
    - Column 2: size (int)
    - Column 3: type (str)
    - Column 4: id (uint64)
    - Column 5: from (str)
    """
    # 🌟 Static constants: Centralize column indices at the top for maximum hardcoded performance and future modifiability
    IDX_START = 0
    IDX_END = 1
    IDX_ID = 4
    IDX_FROM = 5
    REQUIRED_MIN_COLUMNS = 6  # Strict defense: Each row must have at least 6 columns of data

    def __init__(
        self,
        shard_paths: Union[str, List[str]],
        data_name: str="signal",
        buffer_size: int = 20000,
        memmap_dtype: str = "float32",
        shuffle_buffer: bool = True,
        rank: int = 0,
        world_size: int = 1,
        is_repeat: bool = True,
        seed: int = 6198,
        memmap_cache_capacity: int = 64,
        max_samples: int = None,
        verbose: bool = False
    ):
        super().__init__()
        input_paths = [shard_paths] if isinstance(shard_paths, str) else shard_paths
        self.buffer_size = buffer_size
        self.data_dtype = np.dtype(memmap_dtype) # Ensure it is a numpy dtype object
        self.shuffle_buffer = shuffle_buffer
        self.rank = rank
        self.world_size = world_size
        self.is_repeat = is_repeat
        self.seed = seed
        self.data_name = data_name
        self.max_samples = max_samples
        self.verbose = verbose  # Control verbosity of runtime details
        self.mmap_cache = LRUMMapCache(capacity=memmap_cache_capacity) # 设置合适的缓存容量
        found_npy_files = resolve_input_paths(input_paths)

        if self.max_samples is not None:
            self.max_samples_per_rank = (self.max_samples + self.world_size - 1) // self.world_size
        else:
            self.max_samples_per_rank = None

        # Find corresponding .csv.gz based on .npy
        self.all_csv_files = []
        for npy_path in found_npy_files:
            base_path, _ = os.path.splitext(npy_path)
            csv_path = f"{base_path}.csv.gz"
            if os.path.exists(csv_path):
                self.all_csv_files.append(csv_path)
            else:
                logger.warning(f"Missing corresponding index: {csv_path}, skipping .npy: {npy_path}")
        if not self.all_csv_files:
            raise FileNotFoundError("No valid [.npy + .csv.gz] matching pairs found.")

        # 2. Initialization phase: Rapidly count total rows and build debug dictionary (direct sequential scan)
        if self.rank == 0 and self.verbose:
            logger.info(f"💾 [Dataset Init] Scanning global total rows and building debug map...")


        self.total_rows_cached = 0
        for csv_path in tqdm.tqdm(self.all_csv_files, desc="Scanning Files", disable=not self.verbose):
            try:
                with gzip.open(csv_path, 'rt', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        # Strict boundary defense using class constants
                        if len(parts) < self.REQUIRED_MIN_COLUMNS:
                            logger.warning(f"Error csv format: {csv_path}")
                            continue
                        chunk_id = int(parts[self.IDX_ID])
                        from_path = parts[self.IDX_FROM]
                        self.total_rows_cached += 1
            except Exception as e:
                logger.error(f"⚠️ Failed to parse file during init: {csv_path}. Error: {e}")
                continue
        if self.rank == 0 and self.verbose:
            logger.info(f"🛰️ [Streaming graph built successfully] Total rows: {self.total_rows_cached}")



    def _get_mmap_stream(self, npy_path: str) -> np.memmap:
        """
        LRU cache
        """
        try:
            return self.mmap_cache.get(npy_path, self.data_dtype)
        except Exception as e:
            logger.error(f"Failed to create mmap for {npy_path}: {e}")
            raise
    

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

        # ---------------- Trace Code Start ----------------
        worker_id = worker_info.id if worker_info else 0
        file_names = [os.path.basename(f) for f in final_files]
        if self.verbose:
            logger.info(
               f"[Trace] Rank {self.rank} | Worker {worker_id} | "
               f"Final shard list length: {len(final_files)} | Order: {file_names}"
            )
        # ---------------- Trace Code End ----------------

        return final_files

    def _parse_and_yield(self, my_shards, rng):
        # worker_info = get_worker_info()
        # When num_workers=0, torch.utils.data.get_worker_info() returns None.
        # Direct access to worker_info.id will cause an AttributeError if multiprocessing (worker) is not enabled.
        # The program will crash.
        # worker_id = worker_info.id if worker_info is not None else 0

        # Log: Record the start of loading shards for the current task
        # if self.verbose: 
        #     logger.info(f"[Trace] (Worker {worker_id}) Start processing shard list: {len(my_shards)} files")

        # Establish local buffer within the current Epoch
        buffer = []
        # 当前iteration已经输出数量
        yielded_samples = 0
        # Shuffle the current shard order before each loop (recommended)
        shuffled_shards = list(my_shards)
        rng.shuffle(shuffled_shards)
       

        
        # buffer pbar
        desc = f"Rank {self.rank} | Buffering" 
        
        pbar = None
        if self.shuffle_buffer and self.buffer_size > 0:
            pbar = tqdm.tqdm(
                total=self.buffer_size, 
                desc=desc,            # 动态描述
                position=self.rank,   # 🚨 关键点：每个 rank 占用不同的行，防止进度条重叠
                leave=True,           # 设置为 True 以便训练开始后能看到最终完成状态
                disable=not self.verbose
            )

        for csv_path in shuffled_shards:
            # Log: Record the specific CSV file being processed
            if self.verbose:
                logger.info(f"Opening: {csv_path}") # Focus: Check if it gets stuck on opening a specific file
                
            npy_path = csv_path.replace(".csv.gz", ".npy")
            if not os.path.exists(npy_path):
                continue

            try:
                mmap_matrix = self._get_mmap_stream(npy_path)
            except Exception as e:
                logger.error(e)
                continue

            try:
                with gzip.open(csv_path, 'rt', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        # 💡 Extremely concise filtering: Straight to the point, reject all dynamic checks
                        if len(parts) < self.REQUIRED_MIN_COLUMNS:
                            logger.warning(f"Error {csv_path}")
                            continue

                        start_idx = int(parts[self.IDX_START])
                        end_idx = int(parts[self.IDX_END])
                        chunk_id = int(parts[self.IDX_ID])

                        if end_idx <= start_idx or end_idx > mmap_matrix.shape[0]:
                            logger.warning("Error")
                            continue

                        # Ultra-fast memory-mapped slicing
                        # Remove np.array(...) conversion, create tensor directly from mmap slice
                        # data_slice = torch.from_numpy(np.array(mmap_matrix[start_idx:end_idx], dtype=np.float32))
                        # This preserves the original data dtype (uint16 or float32)
                        raw_slice = mmap_matrix[start_idx:end_idx]

                        # --- 开始检查逻辑 ---
                        # 检查第一个 token 是否为 2，最后一个是否为 3
                        first_token = raw_slice[0]
                        last_token = raw_slice[-1]
                        # 这里的逻辑根据你的数据格式调整（如果 raw_slice 是二维或多维，请注意索引）
                        if first_token != 2 or last_token != 3:
                            print(f"DEBUG: 异常数据监测 - chunk_id: {chunk_id}, "
                                    f"First: {start_idx}:{first_token}, Last: {end_idx}:{last_token}")
                        # --- 结束检查逻辑 ---

                        data_slice = torch.from_numpy(raw_slice.copy()) # copy() is necessary because mmap is read-only

                        sample = {
                            self.data_name: data_slice,
                            "id": torch.as_tensor(chunk_id, dtype=torch.int64)
                        }
                        
                        if self.shuffle_buffer and self.buffer_size > 0:
                            if len(buffer) < self.buffer_size:
                                buffer.append(sample)
                                if pbar is not None:
                                    pbar.update(1)
                            else:
                                if pbar is not None:
                                    pbar.close() # 一旦 buffer 充满，进度条就会关闭，避免它在训练过程中继续显示无意义的进度（或者你可以改为在 buffer 充满后打印一行 logger.info("Buffer filled, starting yield...")）。
                                    pbar = None
                                idx = rng.randint(0, len(buffer) - 1)
                                yield_sample = buffer[idx]
                                buffer[idx] = sample
                                yield yield_sample
                                # Generator resumes here after the caller requests the next sample.
                                yielded_samples += 1
                                if self.max_samples_per_rank is not None and yielded_samples >= self.max_samples_per_rank:
                                    return
                        else:
                            yield sample
                            yielded_samples += 1
                            if self.max_samples_per_rank is not None and yielded_samples >= self.max_samples_per_rank:
                                return

            except Exception as e:
                logger.error(f"Error in yield: {e}")
                continue

        if self.shuffle_buffer and len(buffer) > 0:
            rng.shuffle(buffer)
            for remain_sample in buffer:
                yield remain_sample
                yielded_samples += 1
                if self.max_samples_per_rank is not None and yielded_samples >= self.max_samples_per_rank:
                    return
        elif len(buffer) > 0:
            for remain_sample in buffer:
                yield remain_sample
                yielded_samples += 1
                if self.max_samples_per_rank is not None and yielded_samples >= self.max_samples_per_rank:
                    return
    
    def __iter__(self):
        """Strategy Control Layer: Intercept workers without tasks here, and globally control random seeds"""
        # 1. 🌟 Core Fix: Distribute shards at the outermost layer
        my_shards = self._get_pipeline_shards()
        if not my_shards:
            # If this Worker is not assigned any files, exit gracefully, never enter an empty while True loop!
            return

        # 2. 🌟 Core Fix: Establish base seed
        worker_info = get_worker_info()
        base_seed = self.seed + self.rank * 1000 + (worker_info.id if worker_info else 0)
        rng = random.Random(base_seed)

        if self.is_repeat:
            epoch = 0
            while True:
                # Upon entering a new Epoch, reset the seed to ensure different shuffle orders
                epoch_seed = base_seed + epoch * 555
                rng.seed(epoch_seed)
                yield from self._parse_and_yield(my_shards, rng)
                epoch += 1
        else:
            # Validation mode: Run once with rng, exit gracefully
            yield from self._parse_and_yield(my_shards, rng)

    def __len__old(self):
        """
        🚨 Core Fix: Do not return global total rows, otherwise DDP training Epoch display will slow down by world_size times.
        💡 Fix: Return the estimated number of samples allocated to the current Rank (single GPU).
        """
        if self.world_size > 1:
            return self.total_rows_cached // self.world_size
        return self.total_rows_cached

    def __len__(self):

        if self.world_size > 1:
            estimated_len = self.total_rows_cached // self.world_size
        else:
            estimated_len = self.total_rows_cached
        if self.max_samples is not None:
            return min(
                estimated_len,
                self.max_samples_per_rank
            )
        return estimated_len

    def get_source_path_by_id(self, chunk_id: int) -> str:
        return "Unknown_Source_Path"