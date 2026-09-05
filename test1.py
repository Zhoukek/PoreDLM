# 读取npy文件并绘图
import numpy as np
import matplotlib.pyplot as plt
from poreproc import SignalProcessor

# 读取 npy 文件
file_path = "/mnt/si002562jbsc/poregpt/datasets/DNA_S1_HG00200_MIX_250F701901011/basecall/eval/eval_00001/chunks_apple.npy"

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

# 提取第一个数据
if data.ndim == 1:
    # 如果是一维数组，直接使用
    signal = data
elif data.ndim == 2:
    # 如果是二维 (样本, 时间步)，取第一行
    signal = data[0]
elif data.ndim == 3:
    # 如果是三维 (样本, 时间步, 通道)，取第一个样本的第一个通道
    signal = data[0, :, 0]
    print(f"三维数据，取第一个样本的第一个通道，长度: {len(signal)}")
else:
    print(f"数据维度为 {data.ndim}，无法自动处理，请检查数据格式")
    signal = None

if signal is not None:
    total_len = len(signal)
    print(f"信号总长度: {total_len}")
    
    # 取中间大约500个点
    if total_len > 500:
        start_idx = (total_len - 500) // 2
        end_idx = start_idx + 500
        signal_cropped = signal[start_idx:end_idx]
        print(f"截取区间: {start_idx} 到 {end_idx} (共 {len(signal_cropped)} 个点)")
    else:
        signal_cropped = signal
        print(f"信号长度不足500，显示全部 {len(signal_cropped)} 个点")
    
    # 绘制截取后的信号
    plt.figure(figsize=(14, 6))
    plt.plot(signal_cropped, linewidth=1.5)
    plt.xlabel('Time Step (cropped)', fontsize=12)
    plt.ylabel('Signal Value', fontsize=12)
    plt.title(f'First Signal - Middle 500 Points (Original length: {total_len})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("cropped_signal_plot.png", dpi=300)
    plt.show()
    
    # 输出统计信息
    print(f"\n截取后信号统计:")
    print(f"  长度: {len(signal_cropped)}")
    print(f"  最小值: {np.min(signal_cropped):.4f}")
    print(f"  最大值: {np.max(signal_cropped):.4f}")
    print(f"  均值: {np.mean(signal_cropped):.4f}")
    print(f"  标准差: {np.std(signal_cropped):.4f}")