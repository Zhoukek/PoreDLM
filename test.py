# # 读取jsonl.gz文件
import gzip
import json

# 文件路径
file_path = "/mnt/zzbnew/rnamodel/shenhaojie/data/ONT-R9-basecall/ONT-R9_chunks.jsonl.gz"

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
        if len(value) > 30:  # 列表只显示前10个元素
            return [truncate_value(v, max_length) for v in value[:40]] + [f"...({len(value)-40} more)"]
        return [truncate_value(v, max_length) for v in value]
    elif isinstance(value, dict):
        # 对于字典，限制显示的键值对数量
        new_dict = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= 50:  # 只显示前10个键值对
                new_dict[f"...({len(value)-30} more keys)"] = "..."
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
print_truncated_json(data, max_length=100000)  # 可以调整这个长度

data = read_specific_line(file_path, 1)  # 索引1表示第二行
print("第二行数据:", json.dumps(data, indent=2, ensure_ascii=False))

# # ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

# # 读取npy文件
import numpy as np

# 读取 npy 文件
file_path = "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/without_modifiction/chunks.npy"
file_path = "/mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/S0_HG002_UNMOD-35g/stage1_tokenizer_apple/validation/250F601844011_0_0_0_0_chunks.npy"
file_path = "/mnt/si002562jbsc/poregpt/models/HF_VQE768C08A001_DNADLLM_V003/basecall/DNA_S1_HG00200_MIX_250F701901011_validation_1_to_50_stone/reconstructed_signal/validation_00001_references.npy"

data = np.load(file_path, allow_pickle=True)

# 输出基本信息
print("=" * 50)
print("文件信息:")
print(f"文件路径: {file_path}")
print(f"数据类型: {data.dtype}")
print(f"数组形状: {data.shape}")
print(f"数组维度: {data.ndim}")
print(f"总元素数: {data.size}")
print(f"内存大小: {data.nbytes / 1024 / 1024:.2f} MB")
print("=" * 50)

# 输出内容
# print("\n数组内容:")
# print(data)

# 如果是高维数组，输出部分内容
if data.ndim >= 2:
    print(f"\n前5行（如果有）:")
    print(data[0] if data.shape[0] > 5 else data)


# ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# 读取
# import h5py

# filename = '/mnt/zzbnew/rnamodel/zhoukexuan/rockfish/example/fast5/DEAMERNANOPORE_20161117_FNFAB43577_MN16450_sequencing_run_MA_821_R9_4_NA12878_11_17_16_88738_ch301_read41_strand.fast5'  # 替换为你的文件名

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
#         sample_to_show = data_samples[1]
        
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
# analyze_jsonl("/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/LB07_AND_LB06/all_data/split/validation.jsonl")


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
# print_signal("/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/LB07_AND_LB06/all_data/split/val.jsonl")



# import json
# import ast
# from pathlib import Path


# jsonl_path = Path("/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/LB07_AND_LB06/all_data/split/val.jsonl")  # 修改成你的 jsonl 文件路径


# def parse_span(span_field):
#     if isinstance(span_field, list):
#         return span_field

#     if isinstance(span_field, str):
#         try:
#             return json.loads(span_field)
#         except Exception:
#             return ast.literal_eval(span_field)

#     raise TypeError(f"Unsupported span type: {type(span_field)}")


# def is_valid_span(x):
#     """
#     判断一个 span 是否是有效的 [start, end]
#     """
#     if not isinstance(x, (list, tuple)):
#         return False
#     if len(x) != 2:
#         return False
#     if x[0] is None or x[1] is None:
#         return False
#     return True


# def get_signal_length_from_span(span):
#     """
#     从 base_sample_span_ref 中计算有效电信号长度。

#     会跳过 None span，例如：
#     [[22, 35], [None, None], [35, 54]]
#     会按有效部分计算：
#     54 - 22 = 32
#     """
#     valid_spans = [x for x in span if is_valid_span(x)]

#     if len(valid_spans) == 0:
#         return None, None, None, 0

#     start = valid_spans[0][0]
#     end = valid_spans[-1][1]
#     signal_len = end - start

#     return signal_len, valid_spans[0], valid_spans[-1], len(valid_spans)


# max_info = None
# min_info = None
# total_len = 0
# valid_count = 0
# invalid_count = 0
# none_span_count = 0

# with open(jsonl_path, "r", encoding="utf-8") as f:
#     for line_idx, line in enumerate(f, start=1):
#         line = line.strip()
#         if not line:
#             continue

#         try:
#             item = json.loads(line)

#             span = parse_span(item["base_sample_span_ref"])

#             raw_span_num = len(span)
#             valid_span_num = sum(1 for x in span if is_valid_span(x))

#             if valid_span_num < raw_span_num:
#                 none_span_count += 1

#             signal_len, first_span, last_span, valid_ref_bases = get_signal_length_from_span(span)

#             if signal_len is None:
#                 invalid_count += 1
#                 read_id = item.get("read_id", f"line_{line_idx}")
#                 label = item.get("label", "")
#                 print(f"[Warning] line {line_idx} read_id={read_id} label={label} 没有任何有效 span，跳过")
#                 continue

#             read_id = item.get("read_id", f"line_{line_idx}")
#             label = item.get("label", "")

#             info = {
#                 "line_idx": line_idx,
#                 "read_id": read_id,
#                 "label": label,
#                 "signal_length": signal_len,
#                 "first_span": first_span,
#                 "last_span": last_span,
#                 "raw_ref_bases": raw_span_num,
#                 "valid_ref_bases": valid_ref_bases,
#                 "none_or_invalid_spans": raw_span_num - valid_ref_bases,
#             }

#             if max_info is None or signal_len > max_info["signal_length"]:
#                 max_info = info

#             if min_info is None or signal_len < min_info["signal_length"]:
#                 min_info = info

#             total_len += signal_len
#             valid_count += 1

#         except Exception as e:
#             invalid_count += 1
#             print(f"[Warning] line {line_idx} 解析失败: {e}")


# print("=" * 80)
# print(f"有效记录数: {valid_count}")
# print(f"无效/跳过记录数: {invalid_count}")
# print(f"存在 None 或非法 span 的记录数: {none_span_count}")

# if valid_count > 0:
#     avg_len = total_len / valid_count

#     print("\n最长电信号:")
#     print(f"  read_id: {max_info['read_id']}")
#     print(f"  label: {max_info['label']}")
#     print(f"  line_idx: {max_info['line_idx']}")
#     print(f"  signal_length: {max_info['signal_length']}")
#     print(f"  first_span: {max_info['first_span']}")
#     print(f"  last_span: {max_info['last_span']}")
#     print(f"  原始 ref span 数量: {max_info['raw_ref_bases']}")
#     print(f"  有效 ref span 数量: {max_info['valid_ref_bases']}")
#     print(f"  None/非法 span 数量: {max_info['none_or_invalid_spans']}")

#     print("\n最短电信号:")
#     print(f"  read_id: {min_info['read_id']}")
#     print(f"  label: {min_info['label']}")
#     print(f"  line_idx: {min_info['line_idx']}")
#     print(f"  signal_length: {min_info['signal_length']}")
#     print(f"  first_span: {min_info['first_span']}")
#     print(f"  last_span: {min_info['last_span']}")
#     print(f"  原始 ref span 数量: {min_info['raw_ref_bases']}")
#     print(f"  有效 ref span 数量: {min_info['valid_ref_bases']}")
#     print(f"  None/非法 span 数量: {min_info['none_or_invalid_spans']}")

#     print("\n平均电信号长度:")
#     print(f"  avg_signal_length: {avg_len:.2f}")
# else:
#     print("没有成功解析到任何有效记录。")