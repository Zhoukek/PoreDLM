# 读取token文件
import gzip
import json

# 文件路径
# file_path = "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/data/stage2_BERT_Encoder/test/train_00005.jsonl.gz"
file_path = "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/data/stage2_BERT_Encoder/test/train_00005.split.jsonl.gz"
# file_path = "/mnt/zzbnew/rnamodel/shenhaojie/signalDNAmodel/test-haojieshen-model-type26-cnn_type13_teacher_model_distill0.1_VQ_64k_lemon/basecall/validation_00001.jsonl.gz"

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

# 输出结果
print(f"文件: {file_path}")
print(f"总行数: {line_num}")
print(f"\n所有的keys:")
for key in sorted(all_keys):
    print(f"  - {key}")

# 可选：显示第一行数据的示例
print("\n" + "="*50)
print("第一行数据示例:")
with gzip.open(file_path, 'rt', encoding='utf-8') as f:
    first_line = f.readline()
    first_data = json.loads(first_line.strip())
    print(json.dumps(first_data, indent=2, ensure_ascii=False))

# # 读取npy文件
# import numpy as np

# # 读取 npy 文件
# file_path = "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/without_modifiction/chunks.npy"
# file_path = "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/without_modifiction/references.npy"

# data = np.load(file_path)

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
# print("\n数组内容:")
# print(data)

# # 如果是高维数组，输出部分内容
# if data.ndim >= 2:
#     print(f"\n前5行（如果有）:")
#     print(data[:5] if data.shape[0] > 5 else data)