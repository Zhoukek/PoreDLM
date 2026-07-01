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
    input_file = "/mnt/zzbnew/rnamodel/shenhaojie/signalDNAmodel/test-haojieshen-model-type26-cnn_type13_teacher_model_distill0.1_VQ_64k_lemon/basecall/validation_00001.jsonl.gz"   # 替换为您的输入文件路径
    output_file = "/mnt/zzbnew/rnamodel/shenhaojie/signalDNAmodel/test-haojieshen-model-type26-cnn_type13_teacher_model_distill0.1_VQ_64k_lemon/basecall_1000/1000_chunks.jsonl.gz"  # 替换为输出文件路径
    
    extract_first_n_records(input_file, output_file, 1000)