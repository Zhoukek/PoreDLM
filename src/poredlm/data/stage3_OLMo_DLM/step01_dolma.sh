# 定义公共参数
TOKENIZER_PATH="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/data/stage2_BERT_Encoder/tokenizer-8k.json"
BASE_SRC="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/without_modifiction/stage2_BERT_Encoder/train"
BASE_DST="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/without_modifiction/stage2_BERT_Encoder/train"

# 对三个数据集分别运行
for split in train; do
    dolma tokens \
        --documents "${BASE_SRC}/${split}/*.gz" \
        --tokenizer.name_or_path "$TOKENIZER_PATH" \
        --destination "${BASE_DST}/${split}" \
        --dtype "uint16" \
        --tokenizer.pad_token_id 1 \
        --tokenizer.bos_token_id 2 \
        --tokenizer.eos_token_id 3 \
        --processes 32
done