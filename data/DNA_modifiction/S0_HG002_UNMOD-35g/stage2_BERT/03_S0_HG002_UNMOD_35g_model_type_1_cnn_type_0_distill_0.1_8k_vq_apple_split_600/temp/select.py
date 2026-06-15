import gzip
import json

def extract_first_n_records(input_file, output_file, n=1000):
    """
    从jsonl.gz文件中提取前n条记录，保存到新的jsonl.gz文件中
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        n: 要提取的记录数量
    """
    count = 0
    
    with gzip.open(input_file, 'rt', encoding='utf-8') as f_in:
        with gzip.open(output_file, 'wt', encoding='utf-8') as f_out:
            for line in f_in:
                if count >= n:
                    break
                # 写入当前行到输出文件
                f_out.write(line)
                count += 1
    
    print(f"已成功提取前 {count} 条记录到 {output_file}")

# 使用示例
if __name__ == "__main__":
    input_file = "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/S0_HG002_UNMOD-35g/stage2_BERT/03_S0_HG002_UNMOD_35g_model_type_1_cnn_type_0_distill_0.1_8k_vq_apple_split_600/train/250F601844011_0_0_0_0_chunks.split.jsonl.gz"   # 替换为您的输入文件路径
    output_file = "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/S0_HG002_UNMOD-35g/stage2_BERT/03_S0_HG002_UNMOD_35g_model_type_1_cnn_type_0_distill_0.1_8k_vq_apple_split_600/temp/train/output_2000.jsonl.gz"  # 替换为输出文件路径
    
    extract_first_n_records(input_file, output_file, 2000)