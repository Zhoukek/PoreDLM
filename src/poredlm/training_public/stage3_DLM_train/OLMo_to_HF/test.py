import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

model_dir = Path("/mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training_public/stage3_DLM_train/runs/test/hf_dlm")
elf_src = model_dir / "ELF-pytorch-port" / "src"
if str(elf_src) not in sys.path:
    sys.path.insert(0, str(elf_src))

model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
model.eval()

input_ids = torch.tensor([[2, 129, 130, 131, 132, 3]], dtype=torch.long)
attention_mask = torch.ones_like(input_ids)

with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_context=True,
        return_ode_hidden=True,
        ode_steps=2,
        ode_start_t=0.95,
        ode_self_cond_cfg_scale=0.0,
    )

hidden = outputs["last_hidden_state"]
context_hidden = outputs["context_hidden_state"]
ode_hidden = outputs["ode_hidden_state"]

print(hidden.shape)
print(context_hidden.shape)
print(ode_hidden.shape)

print(ode_hidden)

# Sequence-conditioned generation. max_length is the total length, including
# the condition prefix. For a padded batch, always pass condition_attention_mask.
condition_input_ids = torch.tensor([[2, 129, 130, 131]], dtype=torch.long)
condition_attention_mask = torch.ones_like(condition_input_ids)
with torch.no_grad():
    generated_ids = model.generate(
        condition_input_ids=condition_input_ids,
        condition_attention_mask=condition_attention_mask,
        max_length=64,
        num_steps=50,
        sampling_method="ode",
        cfg_scale=1.0,
        self_cond_cfg_scale=1.0,
        seed=6198,
    )

assert torch.equal(generated_ids[:, : condition_input_ids.shape[1]], condition_input_ids)
print(generated_ids.shape)
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=False))
