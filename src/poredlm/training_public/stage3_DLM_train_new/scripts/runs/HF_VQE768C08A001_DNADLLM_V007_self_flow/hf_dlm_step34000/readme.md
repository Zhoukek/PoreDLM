# Stage 3 PoreLM HF Conversion and Inference

本目录提供 Stage 3 PoreLM checkpoint 到 Hugging Face custom model 的转换脚本。转换后的模型目录是自包含的：推理时只需要发布后的 HF 模型目录，不需要再依赖本地训练源码目录。

转换后的模型支持三类常用调用方式：

1. 对已知 token 序列编码，并可取 `context_hidden_state`、`ode_hidden_state`、`sde_hidden_state` 或 `last_hidden_state`。
2. 对已知 token 序列做短程 ODE refinement，并只替换指定区间的 token。
3. 固定指定区间之外的 token，从高斯噪声出发生成指定区间，也就是 infill。

区间均使用 Python 左闭右开坐标 `[start, end)`，坐标直接对应完整 token 序列位置。如果序列包含 BOS/EOS，它们也占位置。例如 `[BOS, 100 个 content token, EOS]` 的 content token 位于 `[1, 101)`。

## 1. 转换 Checkpoint

先在 `scripts/convert_porelm_to_hf.sh` 中设置：

```bash
INPUT_DIR=/path/to/step_xxx-unsharded
OUTPUT_DIR=/path/to/hf_porelm
TOKENIZER_JSON_PATH=/path/to/tokenizer-64k.json
```

然后运行：

```bash
bash /Users/kexuanzhou/project/PoreDLM/src/poredlm/training_public/stage3_DLM_train_new/scripts/convert_porelm_to_hf.sh
```

转换后的目录大致如下：

```text
hf_porelm/
  config.json
  modeling_poredlm.py
  model.safetensors 或 pytorch_model.bin
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
  porelm_train_config.yaml
  flow_matching_core/
    src/
      porelm/
        __init__.py
        flow_matching_core/
          model.py
          layers.py
          sampling.py
          checkpoints.py
```

如果训练使用了 `training_objective: self_flow`，转换后的 HF 权重不会包含 EMA teacher。推理只使用训练好的 student，也就是 `model.flow_denoiser`。

## 2. 公共加载代码

下面三个示例都从这段代码开始。`input_ids` 应放置模型词表中的 token ID；如果使用本项目的 waveform codebook，codec ID `k` 通常对应 DLM token ID `k + 128`。

```python
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

model_dir = Path("/path/to/hf_porelm")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(
    str(model_dir),
    trust_remote_code=True,
    torch_dtype="auto",
).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

input_ids = torch.tensor(
    [[2, 129, 130, 131, 132, 133, 134, 135, 136, 3]],
    dtype=torch.long,
    device=device,
)
attention_mask = torch.ones_like(input_ids)
```

如果服务器上的 `torch` 或 `transformers` 触发 `triton`/编译相关导入问题，可在运行 Python 前设置：

```bash
export TORCHDYNAMO_DISABLE=1
```

## 3. 对已知 Token 编码

`model(...)` 可以返回多种 hidden states。默认推荐 `ode_hidden_state`，即先由 Stage 2 context encoder 编码，再经过短程确定性 ODE refinement。

```python
encoding_type = "ode_hidden"  # context_hidden, ode_hidden, sde_hidden, last_hidden

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
print(encoded.shape)
```

各输出含义：

- `context_hidden_state`：Stage 2 context encoder 的直接输出，不运行 flow refinement。
- `ode_hidden_state`：从 `context_hidden_state` 出发，经过确定性的 ODE refinement。
- `sde_hidden_state`：从 `context_hidden_state` 出发，经过带随机扰动的 SDE-style refinement，可用 `sde_seed` 复现。
- `last_hidden_state`：在给定 `t` 下执行一次 `flow_denoiser` forward 的输出；未传入 `t` 时默认 `t=1`。

只会计算显式请求的附加分支。例如只设置 `return_ode_hidden=True` 时，不会额外计算 SDE hidden。

## 4. 指定区间短程去噪

下面示例先对完整序列编码，然后在 `t=0.98` 到 `t=1.0` 之间做 2 步 ODE refinement。最后只替换 `[denoise_start, denoise_end)` 区间，区间外 token 保持原值。

```python
denoise_start = 3
denoise_end = 7

with torch.inference_mode():
    context = model.context_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    ).last_hidden_state

    denoised_latents = model.ode_from_context_hidden(
        context,
        attention_mask=attention_mask,
        ode_start_t=0.98,
        ode_steps=2,
        self_cond_cfg_scale=0.5,
    )

    decoder_input = denoised_latents
    if int(getattr(model.flow_denoiser, "num_self_cond_cfg_tokens", 0)) > 0:
        decoder_input = torch.cat([denoised_latents, torch.zeros_like(denoised_latents)], dim=-1)

    decoder_scale = torch.ones(
        input_ids.shape[0],
        device=device,
        dtype=denoised_latents.dtype,
    )
    _, logits = model.flow_denoiser(
        decoder_input,
        torch.ones(input_ids.shape[0], device=device, dtype=denoised_latents.dtype),
        attention_mask=attention_mask.bool(),
        self_cond_cfg_scale=decoder_scale,
        decoder_step_active=True,
    )
    proposed_ids = logits.argmax(dim=-1)

denoised_ids = input_ids.clone()
denoised_ids[:, denoise_start:denoise_end] = proposed_ids[:, denoise_start:denoise_end]

print("before:", input_ids[:, denoise_start:denoise_end])
print("after: ", denoised_ids[:, denoise_start:denoise_end])
```

这个过程适合轻量修正已知序列，不是从高斯噪声重新生成整段信号。若希望实验可比，建议固定 `ode_start_t`、`ode_steps` 和 `self_cond_cfg_scale`。

## 5. 指定区间条件生成

区间 infill 使用 `model.generate()`。`condition_token_mask=True` 的位置作为固定条件，`condition_token_mask=False` 的位置从高斯噪声开始生成。`condition_attention_mask` 表示真实 token/非 padding 位置，不要用它代替 `condition_token_mask`。

```python
generate_start = 3
generate_end = 7

condition_token_mask = torch.ones_like(input_ids, dtype=torch.bool)
condition_token_mask[:, generate_start:generate_end] = False

masked_input_ids = input_ids.clone()
mask_token_id = tokenizer.mask_token_id
if mask_token_id is None:
    mask_token_id = int(getattr(model.config, "pad_token_id", 1) or 1)
masked_input_ids[:, generate_start:generate_end] = int(mask_token_id)

with torch.inference_mode():
    result = model.generate(
        condition_input_ids=masked_input_ids,
        condition_attention_mask=attention_mask,
        condition_token_mask=condition_token_mask,
        max_length=input_ids.shape[1],
        num_steps=64,
        sampling_method="ode",
        cfg_scale=1.0,
        self_cond_cfg_scale=0.5,
        seed=6198,
        return_dict=True,
    )

generated_ids = result["sequences"]
print("generated region:", generated_ids[:, generate_start:generate_end])

assert torch.equal(
    generated_ids[condition_token_mask],
    input_ids[condition_token_mask],
)
```

`max_length` 是完整序列总长度，不是新增 token 数。使用显式 `condition_token_mask` 时，`condition_input_ids.shape[1]` 必须等于 `max_length`，这样模型能保留原始位置并同时利用左右两侧上下文。

## 6. 限制生成到 Codebook Token

如果生成结果要送入 waveform codec，通常只应保留 codebook token。假设 codebook token 范围是 `[128, 128 + 65536)`，可以对 logits 做区间约束：

```python
token_offset = 128
codebook_size = 65536
codebook_end = token_offset + codebook_size

constrained_ids = result["logits"][..., token_offset:codebook_end].argmax(dim=-1)
constrained_ids = constrained_ids + token_offset

generated_ids[:, generate_start:generate_end] = constrained_ids[
    :, generate_start:generate_end
]
```

## 7. 常见注意点

- 推理时不需要 EMA teacher；Self-Flow 只改变训练目标，基本采样流程仍是普通 flow matching ODE/SDE。
- 转换后的 HF 模型目录已经包含最小 `flow_matching_core` runtime，不需要手动把训练源码加入 `sys.path`。
- 旧代码里的 `elf_denoiser` 已改名为 `flow_denoiser`。
- `condition_token_mask=True` 表示固定条件位置，生成后会原样保留。
- `condition_attention_mask=True` 表示有效序列位置；infill 区间虽然未知，但仍然应是有效位置，所以对应 `condition_attention_mask` 应为 `True`。
