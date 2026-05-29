# 定义公共参数
TOKENIZER_PATH="/mnt/si003067jezr/default/poregpt/dolma/tokenizers/pore_64k/tokenizer.json"
BASE_SRC="/mnt/zzbnew/rnamodel/zhoukexuan/poregpt/dataset/human_dna_032g_split1280_overlap256_baseline"
BASE_DST="/mnt/zzbnew/rnamodel/zhoukexuan/poregpt/dataset/human_dna_032g_split1280_overlap256_destination_baseline"

# 对三个数据集分别运行
for split in validation test train; do
    dolma tokens \
        --documents "${BASE_SRC}/${split}/*.gz" \
        --tokenizer.name_or_path "$TOKENIZER_PATH" \
        --destination "${BASE_DST}/${split}" \
        --dtype "uint32" \
        --tokenizer.pad_token_id 1 \
        --tokenizer.bos_token_id 2 \
        --tokenizer.eos_token_id 3 \
        --processes 32
done