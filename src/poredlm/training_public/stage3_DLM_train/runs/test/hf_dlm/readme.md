# Stage3 DLM HF Model Usage

这个目录用于把 public Stage3 DLM 的 OLMo checkpoint 转成 Hugging Face custom model 格式，并测试转换后的 `hf_dlm` 模型。

## 5. Python 调用示例

推荐使用 Hugging Face 标准接口加载：

```python
import sys
from pathlib import Path

import torch
from transformers import AutoModel

model_dir = Path("/path/to/hf_dlm")
elf_src = model_dir / "ELF-pytorch-port" / "src"
if str(elf_src) not in sys.path:
    sys.path.insert(0, str(elf_src))

model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
model.eval()

input_ids = torch.tensor([[2, 129, 130, 131, 3]], dtype=torch.long)
attention_mask = torch.ones_like(input_ids)

with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_context=True,
        return_ode_hidden=True,
        ode_steps=2,
        ode_start_t=0.98,
        ode_self_cond_cfg_scale=0.0,
        return_sde_hidden=True,
        sde_steps=2,
        sde_start_t=0.98,
        sde_gamma=0.1,
        sde_seed=6198,
    )

hidden = outputs["last_hidden_state"]
context_hidden = outputs["context_hidden_state"]
ode_hidden = outputs["ode_hidden_state"]
sde_hidden = outputs["sde_hidden_state"]

print(hidden.shape)
print(context_hidden.shape)
print(ode_hidden.shape)
print(sde_hidden.shape)
```

输出含义：

- `last_hidden_state`: 单次 ELF denoiser forward 输出，默认 `t=1`。
- `context_hidden_state`: Stage2 BERT/context encoder 输出，需要 `return_context=True`。
- `ode_hidden_state`: 从 `context_hidden_state` 出发，经过 deterministic no-noise ELF ODE refinement 后的输出，需要 `return_ode_hidden=True`。
- `sde_hidden_state`: 从 `context_hidden_state` 出发，经过 stochastic/noisy ELF SDE-style refinement 后的输出，需要 `return_sde_hidden=True`。

`ode_steps` 控制 ODE 更新步数；`ode_start_t` 控制起始时间点，常用 `0.95` 或 `0.98`。步数越多，计算越慢，显存和时间开销也越大。

`sde_steps` 和 `sde_start_t` 控制 SDE 更新步数和起始时间点；`sde_gamma` 控制每一步额外注入噪声的强度，`0.0` 会退化成接近 ODE 的确定性更新；`sde_seed` 用于固定随机噪声，方便复现实验。
