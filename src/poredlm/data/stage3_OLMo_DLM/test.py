from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage3_OLMo_DLM/runs/test_zhou/base", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage3_OLMo_DLM/runs/test_zhou/base")

print(model)