# HF DLM：DNABERT V543 + ELF mixed（step 65000）

本目录对应 DNABERT V543 的 Stage 3 ELF DLM checkpoint，训练条件模式为 mixed。最大序列长度为 1536，词表大小为 2560。

## 输入格式与 CLS

训练序列布局为：

```text
[CLS=5, BOS=2, 内容 token..., EOS=3]
```

Hugging Face 模型的 `forward()` 和 `generate()` 不会自动添加 CLS。调用模型前请在最前面添加一次 CLS，并在 `attention_mask` 中将它标为有效 token。tokenizer 的 `cls_token` 字段未设置，因此也不会由 tokenizer 自动添加 CLS。不要重复添加。

本文中的区间坐标均按完整输入序列计算，使用 Python 左闭右开区间 `[start, end)`。内容第一个 token 的位置是 2，内容坐标 `[a, b)` 对应模型序列坐标 `[a + 2, b + 2)`。最大长度 1536 包含 CLS、BOS、EOS，所以最多容纳 1533 个内容 token。请使用本模型 tokenizer 或训练数据中的 token ID；旧波形模型的 `+128` codebook 偏移不适用于此模型。

## 加载模型

```python
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

model_dir = Path(
    "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training_public/"
    "stage3_DLM_train/runs/HF_RSQ741C12V523_DNABERT_V543/"
    "hf_dlm_elf_mixed_with_cls_token_step65000-unsharded"
)
elf_src = model_dir / "ELF-pytorch-port" / "src"
if str(elf_src) not in sys.path:
    sys.path.insert(0, str(elf_src))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(
    str(model_dir), trust_remote_code=True, torch_dtype="auto"
).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

# 示例内容 ID 仅演示序列布局；实际 ID 应来自本模型 tokenizer / 数据。
content_ids = [10, 11, 12, 13]
input_ids = torch.tensor(
    [[5, 2, *content_ids, 3]], dtype=torch.long, device=device
)
attention_mask = torch.ones_like(input_ids)
```

若原始数据已带 BOS/EOS，请保留它们，只在最前面添加 CLS。批次 padding 位置的 `attention_mask` 设为 0，CLS/BOS/内容/EOS 等真实 token 位置设为 1。

## 编码

```python
# 可选：context_hidden、ode_hidden、sde_hidden、last_hidden
encoding_type = "ode_hidden"

with torch.inference_mode():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_context=(encoding_type == "context_hidden"),
        return_ode_hidden=(encoding_type == "ode_hidden"),
        ode_steps=2,
        ode_start_t=0.98,
        ode_self_cond_cfg_scale=0.5,
        return_sde_hidden=(encoding_type == "sde_hidden"),
        sde_steps=2,
        sde_start_t=0.98,
        sde_gamma=0.1,
        sde_self_cond_cfg_scale=0.5,
        sde_seed=6198,
    )

output_key = {
    "context_hidden": "context_hidden_state",
    "ode_hidden": "ode_hidden_state",
    "sde_hidden": "sde_hidden_state",
    "last_hidden": "last_hidden_state",
}[encoding_type]
encoded = outputs[output_key]
print(encoded.shape)  # [batch, 完整序列长度, hidden_size]
content_hidden = encoded[:, 2:-1]  # 取内容位置，去掉 CLS、BOS、EOS
```

- `context_hidden_state`：DNABERT context encoder 的直接输出。
- `ode_hidden_state`：从 context 表征开始的确定性 ODE refinement。
- `sde_hidden_state`：带噪声的 SDE-style refinement，可通过 `sde_seed` 复现。
- `last_hidden_state`：给定 `t` 的 ELF denoiser 输出；省略 `t` 时默认为 1。

## 局部去噪

完整序列会作为上下文输入，只将指定内容区间的解码结果写回原序列。示例去噪内容坐标 `[1, 3)`：

```python
content_start, content_end = 1, 3
start, end = content_start + 2, content_end + 2  # 模型序列坐标 [3, 5)

with torch.inference_mode():
    context = model.context_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    ).last_hidden_state
    latents = model.ode_from_context_hidden(
        context,
        attention_mask=attention_mask,
        ode_start_t=0.97,
        ode_steps=1,
        self_cond_cfg_scale=0.5,
    )

    decoder_input = latents
    if int(getattr(model.elf_denoiser, "num_self_cond_cfg_tokens", 0)) > 0:
        decoder_input = torch.cat([latents, torch.zeros_like(latents)], dim=-1)
    decoder_scale = torch.ones(input_ids.shape[0], device=device, dtype=latents.dtype)
    _, logits = model.elf_denoiser(
        decoder_input,
        torch.ones(input_ids.shape[0], device=device, dtype=latents.dtype),
        attention_mask=attention_mask.bool(),
        self_cond_cfg_scale=decoder_scale,
        decoder_step_active=True,
    )
    proposed_ids = logits.argmax(dim=-1)

denoised_ids = input_ids.clone()
denoised_ids[:, start:end] = proposed_ids[:, start:end]
```

不要将 CLS 或 BOS 放入内容去噪区间。比较实验时固定 `ode_start_t`、`ode_steps` 和 `self_cond_cfg_scale`。

## 条件生成 / infill

`condition_token_mask=True` 表示固定条件，`False` 表示待生成位置。待生成区间仍是有效序列位置，因此 `condition_attention_mask` 在该区间也必须为 1。CLS、BOS 和 EOS 应保持为条件。下面生成内容坐标 `[1, 3)`：

```python
content_start, content_end = 1, 3
start, end = content_start + 2, content_end + 2
condition_token_mask = torch.ones_like(input_ids, dtype=torch.bool)
condition_token_mask[:, start:end] = False

with torch.inference_mode():
    result = model.generate(
        condition_input_ids=input_ids,
        condition_attention_mask=attention_mask,
        condition_token_mask=condition_token_mask,
        max_length=input_ids.shape[1],  # 总长度，包含 CLS/BOS/EOS
        num_steps=64,
        sampling_method="ode",
        cfg_scale=1.0,
        self_cond_cfg_scale=0.5,
        seed=6198,
        return_dict=True,
    )

generated_ids = result["sequences"]
print("generated content:", generated_ids[:, start:end])
assert torch.equal(
    generated_ids[condition_token_mask], input_ids[condition_token_mask]
)
```

`max_length` 是完整序列总长度，不是新增 token 数。若结果用于下游内容解码或评估，请按下游接口需要去除 CLS、BOS、EOS。

## 配置摘要

| 项目 | 值 |
|---|---|
| Stage 2 context encoder | DNABERT V543 (`porebert`) |
| Stage 3 目标 | ELF (`ELF-B`) |
| 条件模式 | mixed |
| CLS | 启用，ID `5`，在 BOS 前 |
| BOS / EOS / PAD | `2` / `3` / `1` |
| 词表大小 / 最大长度 | `2560` / `1536`（含特殊 token）|
| checkpoint | step `65000`，unsharded |

若服务器环境遇到 `triton` 或 `metax` 导入问题，可在运行 Python 前设置：

```bash
export TORCHDYNAMO_DISABLE=1
```
