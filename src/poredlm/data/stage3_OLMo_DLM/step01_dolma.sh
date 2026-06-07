# 定义公共参数
TOKENIZER_PATH="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/S0_HG002_UNMOD-35g/stage2_BERT/00_S0_HG002_UNMOD_35g_model_type_0_cnn_type_0_8k_vq_split_600/tokenizer-8k.json"
BASE_SRC="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/S0_HG002_UNMOD-35g/stage2_BERT/00_S0_HG002_UNMOD_35g_model_type_0_cnn_type_0_8k_vq_split_600"
BASE_DST="/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/S0_HG002_UNMOD-35g/stage3_OLMo_DLM/00_S0_HG002_UNMOD_35g_model_type_0_cnn_type_0_8k_vq"

# 对三个数据集分别运行
for split in test validation; do
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