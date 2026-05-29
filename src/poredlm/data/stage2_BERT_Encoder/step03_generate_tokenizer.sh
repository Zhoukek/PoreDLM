#!/bin/bash

# Tokenizer生成脚本
# 功能：根据指定的K值生成tokenizer.json文件

# 设置K值（码本大小）- 请在此处修改K值
K=8196

# 设置输出文件路径 - 请在此处修改输出文件名
OUTPUT="tokenizer-8k.json"

# 打印执行信息
echo "生成tokenizer，K=$K，输出到 $OUTPUT"

# 执行Python生成脚本
python /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/data/stage2_BERT_Encoder/step03_generate_tokenizer.py --K $K --output $OUTPUT

# 打印完成信息
echo "完成!"
