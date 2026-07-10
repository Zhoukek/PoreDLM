import os
import sys
import yaml
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, AutoModel
import traceback
from tqdm import tqdm  # 🚀 新增：引入进度条

import modeling_pore_vq_codec  # noqa: F401


# 保持你原有的导入路径
from dataset import PoreSignalDataset

def load_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def flush_dolma_chunk(save_folder, output_prefix, chunk_idx, token_buffer, meta_buffer, dtype_np):
    """
    将内存中的 Token 序列和元数据序列化为 standard Dolma 格式落盘
    """
    if not token_buffer:
        return

    # 获取当前 rank，加入文件名中，确保多卡并发写入同一个目录时文件名绝对唯一
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    file_base = f"{output_prefix}_rank{local_rank}_{chunk_idx:04d}"

    npy_path = os.path.join(save_folder, f"{file_base}.npy")
    csv_path = os.path.join(save_folder, f"{file_base}.csv.gz")

    # 1. 保存裸二进制数据 (无 numpy header，完美适配 OLMo 的 memmap 读取)
    flat_tokens = np.array(token_buffer, dtype=dtype_np)
    flat_tokens.tofile(npy_path)

    # 2. 保存元数据 (第 0 列为 start_pos, 第 1 列为 end_pos, 后续为 signal_id)
    df_meta = pd.DataFrame(meta_buffer)
    df_meta.to_csv(csv_path, index=False, header=False, compression="gzip")

    # 💡 既然有了全卡进度条，这里改为单行打印或者通过 tqdm.write 打印，防止冲毁进度条布局
    tqdm.write(f"💾 [Rank {local_rank} Flush] Chunk {chunk_idx:04d}落盘 -> 长度: {flat_tokens.size}")

def main():
    parser = argparse.ArgumentParser(description="Pore RSQ Codec to Dolma Tokenizer pipeline")
    parser.add_argument("--config", type=str, default="configs/infer_codec.yaml", help="Path to config yaml")
    args = parser.parse_args()

    # ---------------------------------------------------------
    # 1. 解析全局分布式配置 (全面对接 OLMo 命名风格)
    # ---------------------------------------------------------
    cfg = load_config(args.config)

    # 获取当前进程的 Rank 和总卡数（torchrun 会自动注入这些环境变量）
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    input_data_paths = cfg['dataset']['input_data_paths']
    if not isinstance(input_data_paths, list):
        raise ValueError(f"配置文件中的 input_data_paths 必须是列表格式")

    model_dir = cfg['model']['model_dir']
    target_layer = cfg['model'].get('target_layer', 0)

    feature_extraction = cfg['dataset']['feature_extraction']
    batch_size = cfg['dataset'].get('batch_size', 1)
    save_folder = cfg['dataset']['save_folder']
    output_prefix = cfg['dataset']['output_prefix']

    items_per_file = cfg['dolma_spec']['items_per_file']
    memmap_dtype_str = cfg['dolma_spec']['memmap_dtype']
    bos_token_id = cfg['dolma_spec']['bos_token_id']
    eos_token_id = cfg['dolma_spec']['eos_token_id']
    token_offset = int(cfg['dolma_spec'].get('token_offset', 128))

    # 类型映射
    dtype_map = {"uint8": np.uint8, "uint16": np.uint16, "uint32": np.uint32}
    if memmap_dtype_str not in dtype_map:
        raise ValueError(f"不支持的 memmap_dtype: {memmap_dtype_str}")
    dtype_np = dtype_map[memmap_dtype_str]
    dtype_max = np.iinfo(dtype_np).max

    os.makedirs(save_folder, exist_ok=True)

    # 新的多卡指定设备代码
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # 只有主进程打印一次欢迎屏
    if local_rank == 0:
        print("=" * 60)
        print(f" 🚀 Starting Codec Tokenization Pipeline (Batch Size: {batch_size})")
        print(f" 🚀 Total GPUs detected by torchrun: {world_size}")
        print("=" * 60)

    # ---------------------------------------------------------
    # 2. 初始化 Codec 模型与特征提取器
    # ---------------------------------------------------------
    try:
        model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(device)
        model.eval()

        feature_extractor = None
        if feature_extraction:
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir, trust_remote_code=True)
    except Exception as e:
        print(f"❌ [Rank {local_rank}] Codec component loading failed!")
        traceback.print_exc()
        sys.exit(1)
    
    # ---------------------------------------------------------
    # 3. 构建分布式数据集并进行单次遍历 (融合 YAML 新参数)
    # ---------------------------------------------------------
    config_buffer_size = cfg['dataset'].get('buffer_size', 1073741824)
    logic_chunk_size = cfg['dataset'].get('logic_chunk_size', 6000)
    memmap_dtype = cfg['dataset'].get('memmap_dtype', 'float32')
    shuffle_buffer = cfg['dataset'].get('shuffle_buffer', False)

    dataset = PoreSignalDataset(
        shard_paths=input_data_paths,
        logic_chunk_size=logic_chunk_size,
        memmap_dtype=memmap_dtype,
        shuffle_buffer=shuffle_buffer,
        rank=local_rank,
        world_size=world_size,
        buffer_size=config_buffer_size,
        is_repeat=False
    )
    
    num_workers = cfg['dataset'].get('num_workers', 4)
    pin_memory = cfg['dataset'].get('pin_memory', True)
    prefetch_factor = cfg['dataset'].get('prefetch_factor', 4) if num_workers > 0 else None
    persistent_workers = cfg['dataset'].get('persistent_workers', True) if num_workers > 0 else False

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )

    # 🚀 【完美修复】直接向 DataLoader 索要当前 Rank 真实的物理总批次数
    try:
        estimated_batches_per_rank = len(dataloader)
    except (TypeError, NotImplementedError):
        estimated_batches_per_rank = None

    # 🚀 【优雅防御】如果当前 Rank 根本没分到数据（total=0），直接隐藏该卡的进度条，免得占屏幕
    disable_tqdm = (estimated_batches_per_rank == 0 or estimated_batches_per_rank is None)

    dataloader = tqdm(
        dataloader, 
        desc=f"⚙️ GPU {local_rank}", 
        total=estimated_batches_per_rank,
        position=local_rank,         
        dynamic_ncols=True,
        leave=True,
        disable=disable_tqdm  # 🌟 没活干的卡直接闭嘴，终端只留真正在干活的卡
    )
    # ---------------------------------------------------------
    # 4. 核心流式批量量化与指针记录逻辑
    # ---------------------------------------------------------
    chunk_idx = 0
    global_item_count = 0
    current_pointer = 0  # 核心变量：记录当前二进制块内的绝对物理偏移指针

    token_buffer = []    # 存放一维化 Token 流
    meta_buffer = []     # 存放 [start_pos, end_pos, signal_id]

    with torch.no_grad():
        for batch in dataloader:
            try:
                # batch['signal'] 形状此时为 [B, Seq_Len]
                signals = batch['signal']
                batch_ids = batch.get('id')

                if not feature_extraction:
                    signal_tensor = signals.to(torch.float32).to(device)
                else:
                    signals_list = [s.numpy() for s in signals]
                    processed_inputs = feature_extractor(signals_list, return_tensors="pt")
                    signal_tensor = processed_inputs["signal"].to(device)

                # 防御性升维：确保张量满足 3D [B, C, T] 的要求
                if signal_tensor.dim() == 2:
                    signal_tensor = signal_tensor.unsqueeze(1)

                # GPU 侧批量前向压缩：一次性获取整个 Batch 的离散 Token IDs
                token_ids_tensor = model.encode_signal(signal_tensor, layer=target_layer)
                token_npy = token_ids_tensor.cpu().numpy()

                # CPU 侧高效流式解包与多路指针合并
                actual_b_size = signal_tensor.size(0)
                for i in range(actual_b_size):
                    if batch_ids is not None and i < len(batch_ids):
                        raw_id_str = str(batch_ids[i].item())
                    else:
                        raw_id_str = f"unknown_{global_item_count}"
                    
                    shifted_tokens = token_npy[i].flatten() + token_offset
                    if shifted_tokens.size > 0 and shifted_tokens.max() > dtype_max:
                        raise ValueError(
                            f"Token id overflow: max shifted token={int(shifted_tokens.max())} "
                            f"exceeds {memmap_dtype_str} max={dtype_max}. "
                            "Use dolma_spec.memmap_dtype: uint32 or lower dolma_spec.token_offset."
                        )
                    token_list = shifted_tokens.tolist()
                    full_sequence = [bos_token_id] + token_list + [eos_token_id]
                    
                    seq_length = len(full_sequence)

                    start_pos = current_pointer
                    end_pos = start_pos + seq_length

                    token_buffer.extend(full_sequence)
                    meta_buffer.append([start_pos, end_pos, raw_id_str])

                    current_pointer = end_pos
                    global_item_count += 1

                    if global_item_count % items_per_file == 0:
                        flush_dolma_chunk(save_folder, output_prefix, chunk_idx, token_buffer, meta_buffer, dtype_np)
                        chunk_idx += 1
                        current_pointer = 0  # 新文件块重新从 0 计数指针
                        token_buffer = []
                        meta_buffer = []

            except Exception as e:
                # 使用 tqdm.write 替代 print，可以防止报错信息撕裂进度条
                tqdm.write(f"⚠️ [Rank {local_rank} Warning] 批次异常. Error: {e}")
                continue

    # ---------------------------------------------------------
    # 5. 扫尾工作 (处理最后不足一整块的剩余数据)
    # ---------------------------------------------------------
    if token_buffer:
        flush_dolma_chunk(save_folder, output_prefix, chunk_idx, token_buffer, meta_buffer, dtype_np)

    tqdm.write(f"🎉 [Rank {local_rank}] 完成！共处理 {global_item_count} 条信号。")

if __name__ == "__main__":
    main()
