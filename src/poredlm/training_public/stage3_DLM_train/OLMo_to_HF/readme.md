# Stage3 DLM HF Model Usage

这个目录用于把 public Stage3 DLM 的 OLMo checkpoint 转成 Hugging Face custom model 格式，并测试转换后的 `hf_dlm` 模型。

## 1. 转换模型

训练完成后运行：

```bash
bash /mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training_public/stage3_DLM_train/OLMo_to_HF/step02_olmo2_latest_to_hf.sh
```

默认输入 checkpoint：

```text
/mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training_public/stage3_DLM_train/runs/test/model/latest-unsharded
```

默认输出 HF 模型：

```text
/mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training_public/stage3_DLM_train/runs/test/hf_dlm
```

## 2. 打包 ELF 依赖

转换后的 `hf_dlm` 需要 `torch_elf` 代码。为了让模型移动到其他目录后也能运行，建议把整个 `ELF-pytorch-port` 文件夹复制到 `hf_dlm` 目录下：

```bash
cp -r /mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training_public/stage3_DLM_train/ELF-pytorch-port \
  /mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training_public/stage3_DLM_train/runs/test/hf_dlm/
```

最终目录结构应该类似：

```text
hf_dlm/
  config.json
  modeling_poredlm.py
  model.safetensors
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
  olmo_train_config.yaml
  ELF-pytorch-port/
    src/
      torch_elf/
```

移动或拷贝模型时，直接移动整个 `hf_dlm` 文件夹即可。

## 3. 设置环境变量

如果 `ELF-pytorch-port` 放在 `hf_dlm` 目录下，加载模型前可以设置：

```bash
export MODEL_DIR=/path/to/hf_dlm
export PYTHONPATH=${MODEL_DIR}/ELF-pytorch-port/src:${PYTHONPATH:-}
```

新版 `modeling_poredlm.py` 会优先自动查找 `${MODEL_DIR}/ELF-pytorch-port/src`，所以上面的 `PYTHONPATH` 是保险写法。

在当前服务器环境中，如果 `transformers` 或 `torch` 触发 `triton/metax` 相关导入问题，可以再设置：

```bash
export TORCHDYNAMO_DISABLE=1
```

## 4. 快速测试

可以用本目录下的测试脚本：

```bash
MODEL_DIR=/mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training_public/stage3_DLM_train/runs/test/hf_dlm \
TOKEN_IDS="2,129,130,131,3" \
ODE_STEPS=4 \
ODE_START_T=0.85 \
DEVICE=cuda \
bash /mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training_public/stage3_DLM_train/OLMo_to_HF/run_test_load_hf_dlm.sh
```

测试脚本会检查：

```text
last_hidden_state
context_hidden_state
ode_hidden_state
```

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
        ode_start_t=0.95,
        ode_self_cond_cfg_scale=0.0,
    )

hidden = outputs["last_hidden_state"]
context_hidden = outputs["context_hidden_state"]
ode_hidden = outputs["ode_hidden_state"]

print(hidden.shape)
print(context_hidden.shape)
print(ode_hidden.shape)
```

输出含义：

- `last_hidden_state`: 单次 ELF denoiser forward 输出，默认 `t=1`。
- `context_hidden_state`: Stage2 BERT/context encoder 输出，需要 `return_context=True`。
- `ode_hidden_state`: 从 `context_hidden_state` 出发，经过 deterministic no-noise ELF ODE refinement 后的输出，需要 `return_ode_hidden=True`。

`ode_steps` 控制 ODE 更新步数；`ode_start_t` 控制起始时间点，常用 `0.85` 或 `0.95`。步数越多，计算越慢，显存和时间开销也越大。
