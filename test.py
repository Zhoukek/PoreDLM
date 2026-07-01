# # 读取jsonl.gz文件
import gzip
import json

# 文件路径
file_path = "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/LB07_AND_LB06/LB06/stage2_fullapple_token1600/validation/validation_fullapple_token1600_modlabel.jsonl.gz"

# 读取文件并收集所有的keys
all_keys = set()

with gzip.open(file_path, 'rt', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        try:
            data = json.loads(line.strip())
            all_keys.update(data.keys())
        except json.JSONDecodeError as e:
            print(f"Line {line_num}: JSON decode error: {e}")
        except Exception as e:
            print(f"Line {line_num}: Error: {e}")

def read_specific_line(file_path, line_number):
    """读取指定行（从0开始计数）"""
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx == line_number:
                return json.loads(line.strip())
    return None

def truncate_value(value, max_length=100):
    """截断过长的值"""
    if isinstance(value, str):
        if len(value) > max_length:
            return value[:max_length] + "...(truncated)"
        return value
    elif isinstance(value, list):
        if len(value) > 10:  # 列表只显示前10个元素
            return [truncate_value(v, max_length) for v in value[:10]] + [f"...({len(value)-10} more)"]
        return [truncate_value(v, max_length) for v in value]
    elif isinstance(value, dict):
        # 对于字典，限制显示的键值对数量
        new_dict = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= 20:  # 只显示前10个键值对
                new_dict[f"...({len(value)-10} more keys)"] = "..."
                break
            new_dict[k] = truncate_value(v, max_length)
        return new_dict
    else:
        return value

def print_truncated_json(data, max_length=100, indent=2):
    """打印截断后的JSON"""
    truncated = truncate_value(data, max_length)
    print(json.dumps(truncated, indent=indent, ensure_ascii=False))

# 输出结果
print(f"文件: {file_path}")
print(f"总行数: {line_num}")
print(f"\n所有的keys:")
for key in sorted(all_keys):
    print(f"  - {key}")

# 显示第一行数据的示例（截断版）
print("\n" + "="*50)
print("第一行数据（截断版）:")

data = read_specific_line(file_path, 0)
print_truncated_json(data, max_length=300)  # 可以调整这个长度

# data = read_specific_line(file_path, 1)  # 索引1表示第二行
# print("第二行数据:", json.dumps(data, indent=2, ensure_ascii=False))

# # ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

# # 读取npy文件
# import numpy as np

# # 读取 npy 文件
# file_path = "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/without_modifiction/chunks.npy"
# file_path = "/mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/S0_HG002_UNMOD-35g/stage1_tokenizer_apple/validation/250F601844011_0_0_0_0_chunks.npy"
# file_path = "/mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/LB07_AND_LB06/LB07/signal_chunks_500_overlap450_apple/train_signal_chunk500_overlap450_apple.npy"

# data = np.load(file_path, allow_pickle=True)

# # 输出基本信息
# print("=" * 50)
# print("文件信息:")
# print(f"文件路径: {file_path}")
# print(f"数据类型: {data.dtype}")
# print(f"数组形状: {data.shape}")
# print(f"数组维度: {data.ndim}")
# print(f"总元素数: {data.size}")
# print(f"内存大小: {data.nbytes / 1024 / 1024:.2f} MB")
# print("=" * 50)

# # 输出内容
# # print("\n数组内容:")
# # print(data)

# # 如果是高维数组，输出部分内容
# if data.ndim >= 2:
#     print(f"\n前5行（如果有）:")
#     print(data[1] if data.shape[0] > 5 else data)


# ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# 读取
# import h5py

# filename = '/mnt/zzbnew/rnamodel/wangxue/data/DNA_data/S0_HG002_UNMOD/250F601844011/fast5_split_one/250F601844011_0_0_0_0/250F601844011_0_0_0_0_part0001/250F601844011_0_0_0_0_part0001.fast5'  # 替换为你的文件名

# with h5py.File(filename, 'r') as f:
#     # 获取前3个read的名称
#     read_names = list(f.keys())[:3]
    
#     for read_name in read_names:
#         print(f"\n{'='*60}")
#         print(f"Read: {read_name}")
#         print(f"{'='*60}")
        
#         read_group = f[read_name]
        
#         # 显示这个read下面有什么
#         print("包含的内容:")
#         for key in read_group.keys():
#             item = read_group[key]
#             if isinstance(item, h5py.Dataset):
#                 print(f"  📄 {key}: shape={item.shape}, dtype={item.dtype}")
#                 # 如果数据量小，显示部分值
#                 if item.size < 100:
#                     print(f"     值: {item[()]}")
#                 else:
#                     print(f"     前5个值: {item[:5]}")
#             else:
#                 print(f"  📁 {key}/")
        
#         # 显示属性
#         if read_group.attrs:
#             print("\n属性:")
#             for attr, val in read_group.attrs.items():
#                 if isinstance(val, bytes):
#                     val = val.decode('utf-8', errors='ignore')
#                 print(f"  {attr}: {val}")

# ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

# import pysam

# # 打开BAM文件
# bam_file = pysam.AlignmentFile("/mnt/zzbnew/rnamodel/wangxue/data/DNA_data/S0_HG002_UNMOD/250F601844011/basecall_chunk/250F601844011_0_0_0_0/250F601844011_0_0_0_0_part0001/acc95.bam", "rb")

# # 方法1：统计所有reads
# total_reads = 0
# for read in bam_file:
#     total_reads += 1

# print(f"Total reads: {total_reads}")


##################
# 读取jsonl文件

# import json

# def truncate_value(value, max_len=500):
#     """截断过长的值"""
#     s = json.dumps(value, ensure_ascii=False)
#     if len(s) > max_len:
#         return s[:max_len] + "...(截断)"
#     return s

# def analyze_jsonl(file_path, max_value_len=500, show_signal_full=True):
#     all_keys = set()
#     data_samples = []
#     valid_count = 0
#     total_lines = 0
#     error_lines = []
    
#     # 用于统计 signal 的长度
#     signal_lengths = []
    
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for i, line in enumerate(f, 1):
#             total_lines += 1
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 data = json.loads(line)
#                 valid_count += 1
#                 all_keys.update(data.keys())
                
#                 # 统计 signal 长度
#                 if 'signal' in data:
#                     signal = data['signal']
#                     if isinstance(signal, list):
#                         signal_lengths.append(len(signal))
#                     else:
#                         signal_lengths.append(len(str(signal)))
                
#                 # 保存示例数据（前5条）
#                 if len(data_samples) < 5:
#                     truncated = {}
#                     for k, v in data.items():
#                         # 对 signal 特殊处理：完整保留或显示真实长度
#                         if k == 'signal' and show_signal_full:
#                             # 完整保留 signal
#                             truncated[k] = v
#                         elif k == 'signal' and not show_signal_full:
#                             # 显示 signal 的摘要信息
#                             if isinstance(v, list):
#                                 truncated[k] = f"[signal列表，长度: {len(v)}，前10个值: {v[:10]}...]"
#                             else:
#                                 truncated[k] = f"[signal，长度: {len(str(v))}]"
#                         elif isinstance(v, str) and len(v) > max_value_len:
#                             truncated[k] = v[:max_value_len] + "...(截断)"
#                         elif isinstance(v, (list, dict)) and len(json.dumps(v)) > max_value_len:
#                             truncated[k] = f"[数据过长，长度: {len(json.dumps(v))}]"
#                         else:
#                             truncated[k] = v
#                     data_samples.append(truncated)
                    
#             except json.JSONDecodeError as e:
#                 error_lines.append(i)
    
#     print("=" * 60)
#     print(f"📊 文件总行数: {total_lines}")
#     print(f"✅ 有效 JSON 数据条数: {valid_count}")
#     if error_lines:
#         print(f"❌ 解析失败的行号: {error_lines}")
#     print("=" * 60)
    
#     print(f"\n🔑 所有 Key（去重后，共 {len(all_keys)} 个）:")
#     print(sorted(all_keys))
    
#     # 打印 signal 长度统计
#     if signal_lengths:
#         print(f"\n📊 Signal 字段长度统计（共 {len(signal_lengths)} 条）:")
#         print(f"  最小长度: {min(signal_lengths)}")
#         print(f"  最大长度: {max(signal_lengths)}")
#         print(f"  平均长度: {sum(signal_lengths)/len(signal_lengths):.2f}")
#         print(f"  第1条长度: {signal_lengths[0] if signal_lengths else 'N/A'}")
#         print(f"  第3条长度: {signal_lengths[2] if len(signal_lengths) > 2 else 'N/A'}")
    
#     if data_samples:
#         print("\n📝 示例数据（signal字段完整显示）:")
#         # 显示第1条数据（索引0）而不是第3条（索引2）
#         sample_to_show = data_samples[0]
        
#         # 如果 signal 太长，只显示部分
#         if 'signal' in sample_to_show and isinstance(sample_to_show['signal'], list):
#             signal = sample_to_show['signal']
#             print(f"signal 长度: {len(signal)}")
#             print(f"signal 前20个值: {signal[:20]}")
#             print(f"signal 后20个值: {signal[-20:]}")
#             # 创建显示用的副本，将signal替换为摘要
#             display_sample = sample_to_show.copy()
#             display_sample['signal'] = f"[signal列表，长度: {len(signal)}，显示前20个值]"
#             print(json.dumps(display_sample, ensure_ascii=False, indent=2))
#         else:
#             print(json.dumps(sample_to_show, ensure_ascii=False, indent=2))

# # 使用示例
# analyze_jsonl("/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/LB07_AND_LB06/LB06/lb06_signal_none_selected.jsonl")


# import json

# def print_signal(file_path):
#     """专门打印signal字段"""
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for i, line in enumerate(f, 5):
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 data = json.loads(line)
#                 if 'signal' in data:
#                     print(f"第 {i} 行的 base_sample_spans_rel 字段:")
#                     signal = data['signal']
#                     print(f"数据类型: {type(signal)}")
#                     print(f"数据长度: {len(signal) if isinstance(signal, list) else len(str(signal))}")
#                     print(f"内容: {signal}")
#                     print("-" * 80)
                    
#                     # 如果只需要第一条，可以break
#                     break  # 只打印第一条
#             except json.JSONDecodeError as e:
#                 print(f"第 {i} 行解析失败: {e}")

# # 使用
# print_signal("/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/LB07_AND_LB06/LB06/lb06_signal_none_selected.jsonl")