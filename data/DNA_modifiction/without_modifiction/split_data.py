import numpy as np
from sklearn.model_selection import train_test_split
import os

# 文件路径
base_path = "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/without_modifiction"
chunks_path = os.path.join(base_path, "chunks.npy")
references_path = os.path.join(base_path, "references.npy")

# 加载数据
print("加载数据...")
chunks = np.load(chunks_path)
references = np.load(references_path)

print(f"Chunks shape: {chunks.shape}")
print(f"References shape: {references.shape}")

# 设置分割比例
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# 首先分割出训练集和临时集（验证+测试）
chunks_train, chunks_temp, references_train, references_temp = train_test_split(
    chunks, references, 
    test_size=(1 - train_ratio), 
    random_state=42,  # 设置随机种子，确保结果可重复
    stratify=None  # 如果需要分层采样，可以基于references的某些特征
)

# 然后分割临时集为验证集和测试集
val_test_ratio = test_ratio / (val_ratio + test_ratio)  # 0.5 因为验证和测试各占15%
chunks_val, chunks_test, references_val, references_test = train_test_split(
    chunks_temp, references_temp,
    test_size=val_test_ratio,
    random_state=42,
    stratify=None
)

# 打印分割后的形状
print("\n分割结果:")
print(f"训练集 - Chunks: {chunks_train.shape}, References: {references_train.shape}")
print(f"验证集 - Chunks: {chunks_val.shape}, References: {references_val.shape}")
print(f"测试集 - Chunks: {chunks_test.shape}, References: {references_test.shape}")

# 保存分割后的数据
output_dir = base_path
print(f"\n保存数据到: {output_dir}")

# 创建子目录来组织文件
train_dir = os.path.join(base_path, "train")
val_dir = os.path.join(base_path, "validation")
test_dir = os.path.join(base_path, "test")

# 保存训练集
np.save(os.path.join(train_dir, "chunks_train.npy"), chunks_train)
np.save(os.path.join(train_dir, "references_train.npy"), references_train)

# 保存验证集
np.save(os.path.join(val_dir, "chunks_val.npy"), chunks_val)
np.save(os.path.join(val_dir, "references_val.npy"), references_val)

# 保存测试集
np.save(os.path.join(test_dir, "chunks_test.npy"), chunks_test)
np.save(os.path.join(test_dir, "references_test.npy"), references_test)

print("数据保存完成！")