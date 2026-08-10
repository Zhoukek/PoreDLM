# Stage 3 Conditional DLM：转换与 Hugging Face 推理

本目录用于将带条件生成能力的 Stage 3 DLM（OLMo checkpoint）转换为 Hugging Face custom model。转换后的模型支持三类用途：

1. 对已知 token 序列编码；
2. 对已知 token 序列做短程 ODE 去噪，并取出指定区间的去噪 token；
3. 固定指定区间之外的 token，从高斯噪声出发生成指定区间（infill）。

下面的区间均使用 Python 的左闭右开坐标 `[start, end)`，并且直接对应传给模型的完整 token 序列位置。如果序列包含 BOS/EOS，二者也占位置。例如 `[BOS, 100 个 content token, EOS]` 的 content token 位于 `[1, 101)`。

## 1. 转换当前 checkpoint

当前 OLMo checkpoint：

```text
/Users/kexuanzhou/project/PoreDLM/src/poredlm/training_public/stage3_DLM_train/runs/HF_VQE768C08A001_DNADLLM_V001_mix/model_mixed/step46000-unsharded
```

转换脚本：

```text
/Users/kexuanzhou/project/PoreDLM/src/poredlm/training_public/stage3_DLM_train/OLMo_to_HF/step02_olmo2_latest_to_hf.sh
```

运行脚本前，需要把脚本内的 `--input_dir` 改为上面的 `step46000-unsharded`，并把 `--output_dir` 设置为希望保存的 HF 模型目录，例如：

```text
/Users/kexuanzhou/project/PoreDLM/src/poredlm/training_public/stage3_DLM_train/runs/HF_VQE768C08A001_DNADLLM_V001_mix/hf_dlm_condition_step46000
```

然后执行：

```bash
bash /Users/kexuanzhou/project/PoreDLM/src/poredlm/training_public/stage3_DLM_train/OLMo_to_HF/step02_olmo2_latest_to_hf.sh
```

转换后的模型依赖 `torch_elf`。为了使模型目录可以独立移动，建议将整个 `ELF-pytorch-port` 复制到 HF 模型目录：

```bash
MODEL_DIR=/path/to/hf_dlm_condition_step46000
cp -r /Users/kexuanzhou/project/PoreDLM/src/poredlm/training_public/stage3_DLM_train/ELF-pytorch-port \
  "${MODEL_DIR}/"
```

目录结构应类似：

```text
hf_dlm_condition_step46000/
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

## 2. 公共加载代码

以下三个示例均先执行这段代码。`input_ids` 中应放置模型词表中的 token ID；本项目的波形 codebook token 默认偏移量为 128，即 codec ID `k` 对应 DLM token ID `k + 128`。

```python
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

model_dir = Path("/path/to/hf_dlm_condition_step46000")
elf_src = model_dir / "ELF-pytorch-port" / "src"
if str(elf_src) not in sys.path:
    sys.path.insert(0, str(elf_src))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(
    str(model_dir), trust_remote_code=True, torch_dtype="auto"
).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

# 示例完整序列；实际使用时替换为自己的 token。
input_ids = torch.tensor(
    [[2, 129, 130, 131, 132, 133, 134, 135, 136, 3]],
    dtype=torch.long,
    device=device,
)
attention_mask = torch.ones_like(input_ids)
```

若服务器上的 `torch`/`transformers` 触发 `triton` 或 `metax` 相关导入问题，可在运行 Python 前设置：

```bash
export TORCHDYNAMO_DISABLE=1
```

## 3. 功能一：对已知 token 编码

沿用之前的 `model(...)` 编码接口，可按需要选择 `context_hidden_state`、`ode_hidden_state`、`sde_hidden_state` 或 `last_hidden_state`。默认推荐并选择 `ode_hidden_state`，即先由 Stage 2 context encoder 编码，再经过确定性的 ODE refinement。输出 `encoded` 的形状为 `[batch, sequence_length, hidden_size]`。

```python
# 可选值："context_hidden"、"ode_hidden"、"sde_hidden"、"last_hidden"
encoding_type = "ode_hidden"

with torch.inference_mode():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_context=(encoding_type == "context_hidden"),
        return_ode_hidden=(encoding_type == "ode_hidden"),
        ode_steps=2,
        ode_start_t=0.98,
        ode_self_cond_cfg_scale=1.0,
        return_sde_hidden=(encoding_type == "sde_hidden"),
        sde_steps=2,
        sde_start_t=0.98,
        sde_gamma=0.1,
        sde_self_cond_cfg_scale=1.0,
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

各输出的含义如下：

- `ode_hidden_state`：默认选择。从 `context_hidden_state` 出发，经过确定性的无噪声 ODE refinement；由 `ode_steps` 和 `ode_start_t` 控制。
- `context_hidden_state`：Stage 2 BERT/context encoder 的直接输出，不运行扩散 refinement。
- `sde_hidden_state`：从 `context_hidden_state` 出发，经过带随机噪声的 SDE-style refinement；可用 `sde_seed` 复现。
- `last_hidden_state`：在给定 `t` 下执行单次 ELF denoiser forward 的输出；未显式传入 `t` 时默认为 `t=1`。

只会计算所选的附加编码分支。例如默认 `encoding_type="ode_hidden"` 时，`return_ode_hidden=True`，而 context 和 SDE 结果不会额外返回。若希望一次比较多种编码，可以同时将对应的 `return_*` 参数设为 `True`，再从 `outputs` 中分别读取。

## 4. 功能二：指定区间的信号去噪

去噪从已知 token 的 context hidden states 出发，在 `t=0.98` 到 `t=1.0` 之间执行 2 步确定性 ODE。这是很短的局部扩散过程，不是从高斯噪声重新生成整段信号。

下面示例对完整序列进行上下文编码和 ODE refinement，最后只采用 `[denoise_start, denoise_end)` 区间的新 token，区间外 token 保持原值。让模型在 refinement 时看到完整序列，可以利用待去噪区间左右两侧的条件。

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
        self_cond_cfg_scale=1.0,
    )

    # 将 ODE 后的 latent 解码回离散 token。
    decoder_input = denoised_latents
    if int(getattr(model.elf_denoiser, "num_self_cond_cfg_tokens", 0)) > 0:
        decoder_input = torch.cat(
            [denoised_latents, torch.zeros_like(denoised_latents)], dim=-1
        )
    decoder_scale = torch.ones(
        input_ids.shape[0], device=device, dtype=denoised_latents.dtype
    )
    _, logits = model.elf_denoiser(
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

这里的 `t=0.98` 表示输入已经非常接近数据端，因此 2 步 ODE 适合做轻量修正。若希望保持实验可比，应固定 `ode_start_t=0.98`、`ode_steps=2` 和 `self_cond_cfg_scale`。

## 5. 功能三：指定区间的条件信号生成

该功能使用 `model.generate()`。待生成区间从高斯噪声开始，区间之外的 token 作为固定条件；使用 `condition_token_mask` 可以保留 token 的原始位置，因此模型能够同时利用左、右两侧上下文。这与 `token_to_waveform` 示例中的 DLM infill 逻辑一致。

下面使用 ODE 采样 64 步生成 `[generate_start, generate_end)`：

```python
generate_start = 3
generate_end = 7

condition_token_mask = torch.ones_like(input_ids, dtype=torch.bool)
condition_token_mask[:, generate_start:generate_end] = False

# 未知位置的占位 token 不会被当作条件；使用模型的 mask token 更直观。
masked_input_ids = input_ids.clone()
mask_token_id = int(getattr(model.config, "mask_token_id", 1))
masked_input_ids[:, generate_start:generate_end] = mask_token_id

with torch.inference_mode():
    result = model.generate(
        condition_input_ids=masked_input_ids,
        condition_attention_mask=attention_mask,
        condition_token_mask=condition_token_mask,
        max_length=input_ids.shape[1],       # 总长度，不是新增 token 数
        num_steps=64,
        sampling_method="ode",
        cfg_scale=1.0,
        self_cond_cfg_scale=1.0,
        seed=6198,
        return_dict=True,
    )

generated_ids = result["sequences"]
print("generated region:", generated_ids[:, generate_start:generate_end])

# generate() 会原样保留所有条件位置。
assert torch.equal(
    generated_ids[condition_token_mask], input_ids[condition_token_mask]
)
```

若生成结果将送入 Stage 1 waveform codec，建议只允许波形 codebook ID。当前 DLM 的 codebook token 范围默认是 `[128, 128 + 65536)`；可以在待生成位置上约束 logits 后再取 `argmax`：

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

完整的 token 到波形解码、区间 MSE、绘图和批量样本处理可参考：

```text
/Users/kexuanzhou/project/PoreDLM/src/poredlm/training/satge_EX_Waveform_Reconstruction/token_to_waveform/README.md
```

其中 DLM 区间补全的关键关系是：

- `condition_token_mask=True`：固定的已知 token；
- `condition_token_mask=False`：从高斯噪声开始生成的 token；
- `max_length`：包含 BOS/EOS 在内的完整序列总长度；
- `num_steps=64, sampling_method="ode"`：64 步确定性 ODE 生成；
- `seed`：固定高斯初始噪声和随机 time schedule，便于复现实验。

注意：`condition_attention_mask` 表示真实 token/非 padding 位置，不能用它代替 `condition_token_mask`。做区间 infill 时，未知区间仍属于有效序列位置，因此其 `condition_attention_mask` 应为 1，而 `condition_token_mask` 应为 0。
