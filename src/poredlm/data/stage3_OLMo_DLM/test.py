from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("/mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage3_OLMo_DLM/runs/02_150m_no_cond_8k_vq/hf_bert", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("/mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage3_OLMo_DLM/runs/02_150m_no_cond_8k_vq/hf_bert")

print(model)