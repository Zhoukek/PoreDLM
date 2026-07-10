import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel

codec_name = "/path/to/stage1_codec"
bert_name = "/path/to/stage2_bert_hf"

codec = AutoModel.from_pretrained(codec_name, trust_remote_code=True).eval()
feat_ext = AutoFeatureExtractor.from_pretrained(codec_name, trust_remote_code=True)
bert = AutoModel.from_pretrained(bert_name, trust_remote_code=True).eval()

raw_signal = np.random.normal(70, 8, 1855).astype(np.float32)

with torch.no_grad():
    signal = feat_ext(raw_signal, return_tensors="pt")["signal"]
    codebook_ids = codec.encode_signal(signal, layer=0).long()

    token_offset = 128
    bos_token_id = 2
    eos_token_id = 3

    shifted_ids = codebook_ids + token_offset
    bert_input_ids = torch.cat(
        [
            torch.full((shifted_ids.size(0), 1), bos_token_id, dtype=torch.long),
            shifted_ids,
            torch.full((shifted_ids.size(0), 1), eos_token_id, dtype=torch.long),
        ],
        dim=1,
    )

    attention_mask = torch.ones_like(bert_input_ids)
    outputs = bert(input_ids=bert_input_ids, attention_mask=attention_mask)
    embeddings = outputs.last_hidden_state

print("Embedding shape:", embeddings.shape)