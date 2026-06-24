# 读取jsonl.gz文件
import gzip
import json

# 文件路径
# file_path = "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/data/stage2_BERT_Encoder/test/train_00005.jsonl.gz"
# file_path = "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/data/stage2_BERT_Encoder/test/train_00005.split.jsonl.gz"
file_path = "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/without_modifiction/stage2_BERT_Encoder/validation/references_validation.jsonl.gz"

# 标准
file_path = "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/S0_HG002_UNMOD-35g/stage2_BERT/03_S0_HG002_UNMOD_35g_model_type_1_cnn_type_0_distill_0.1_8k_vq_apple_split_600/validation/250F601844011_0_0_0_0_chunks.split.jsonl.gz"
# file_path = "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/S0_HG002_UNMOD-35g/stage2_BERT/00_S0_HG002_UNMOD_35g_model_type_0_cnn_type_0_8k_vq/validation/250F601844011_0_0_0_0_chunks.jsonl.gz"

# file_path = "/mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/S0_HG002_UNMOD-35g/stage4_finetune/temp_10/temp_10.jsonl.gz"

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

# 输出结果
print(f"文件: {file_path}")
print(f"总行数: {line_num}")
print(f"\n所有的keys:")
for key in sorted(all_keys):
    print(f"  - {key}")

# 可选：显示第一行数据的示例
print("\n" + "="*50)

data = read_specific_line(file_path, 0)  # 索引1表示第二行
print("第一行数据:", json.dumps(data, indent=2, ensure_ascii=False))

# data = read_specific_line(file_path, 1)  # 索引1表示第二行
# print("第二行数据:", json.dumps(data, indent=2, ensure_ascii=False))

# ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

# 读取npy文件
import numpy as np

# 读取 npy 文件
file_path = "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/without_modifiction/chunks.npy"
file_path = "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/S0_HG002_UNMOD-35g/stage1_tokenizer_mongo/validation/250F601844011_0_0_0_0_chunks.npy"
file_path = "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/S0_HG002_UNMOD-35g/stage1_tokenizer_apple/validation/250F601844011_0_0_0_3_chunks.npy"

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
print("\n数组内容:")
print(data)

# 如果是高维数组，输出部分内容
if data.ndim >= 2:
    print(f"\n前5行（如果有）:")
    print(data[1] if data.shape[0] > 5 else data)


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