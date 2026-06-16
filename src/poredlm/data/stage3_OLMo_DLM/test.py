from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("/mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage3_OLMo_DLM/runs/01_150_no_cond_8k_vq_test/hf", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("/mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage3_OLMo_DLM/runs/01_150_no_cond_8k_vq_test/hf")

print(model)