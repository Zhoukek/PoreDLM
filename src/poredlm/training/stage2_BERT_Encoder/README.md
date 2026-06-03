# Stage2 BERT Encoder Training

本目录用于训练 PoreDLM 的第二阶段 BERT 编码器。Stage2 不直接处理原始纳米孔信号，而是处理 Stage1 tokenizer 生成的离散 VQ/codebook token 序列。

当前重点训练入口有两个：

| 文件 | 作用 |
| --- | --- |
| `stage2_bert_encoder_train.py` | 原始版 Stage2 BERT MLM 训练，只预测 masked token id。 |
| `stage2_bert_encoder_train_v4.py` | 增强版训练，保留 v3 分阶段 mask，并加入 Stage1 codebook vector regression 辅助损失。 |

辅助文件：

| 文件 | 作用 |
| --- | --- |
| `dataset.py` | 读取 `.jsonl` / `.jsonl.gz` 里的 `<|bwav:id|>` token 文本，并 padding 成 batch。 |
| `bert_encoder_model.py` | 根据 YAML 配置构建 HuggingFace `BertForMaskedLM`。 |
| `runs/test_zhou/config.yaml` | 原始版训练示例配置。 |
| `runs/test_zhou/config_v4.yaml` | v4 训练示例配置。 |
| `runs/test_zhou/run.sh` | 原始版启动脚本。 |
| `runs/test_zhou/run_v4.sh` | v4 启动脚本。 |

本文只介绍训练逻辑，不展开 `eval/` 目录下的重建评估脚本。

## 数据格式

Stage2 训练数据来自 Stage1 tokenizer 的输出，通常保存为 `.jsonl.gz`。每一行是一个 JSON 对象，至少包含 `text` 字段：

```json
{"text": "<|bwav:123|><|bwav:456|><|bwav:789|>"}
```

`dataset.py` 会用正则提取所有 `<|bwav:id|>` token，并通过 `tokenizer_path` 映射到 BERT vocabulary id。

在当前配置中，Stage1 codebook id 会整体偏移 `129`：

```text
stage1_codebook_id = bert_vocab_id - 129
bert_vocab_id = stage1_codebook_id + 129
```

例如：

```text
Stage1 codebook id: 300
BERT vocab id:      429
```

这个偏移在 v4 的 codebook vector loss 中非常重要。

## 原始版训练流程

入口：

```text
stage2_bert_encoder_train.py
```

整体流程：

```text
jsonl.gz
  -> dataset.py 解析 <|bwav:id|>
  -> Stage2Collator padding
  -> mask_token_ids 随机选择 masked token
  -> BertForMaskedLM
  -> CE token loss
  -> 反向传播
```

原始版 mask 逻辑在 `mask_token_ids` 中：

```python
probability_matrix = torch.full(labels.shape, mask_probability)
masked_indices = torch.bernoulli(probability_matrix).bool() & valid_positions
```

含义是：每个有效 token 独立地以 `mask_probability` 概率被选为 MLM 目标。默认常用：

```yaml
mask_probability: 0.15
```

被选中的 token 采用标准 BERT 80/10/10 替换规则：

| 比例 | 输入给 BERT 的内容 | 训练标签 |
| --- | --- | --- |
| 80% | 替换成 `[MASK]`，当前配置中 `mask_token_id: 4` | 原始 token id |
| 10% | 替换成随机 token | 原始 token id |
| 10% | 保持原 token 不变 | 原始 token id |

未被选中的位置标签设为 `-100`，不会参与 CE loss。

原始版 loss：

```text
Loss = CE(pred_token_id, true_token_id)
```

它只关心 token id 是否预测正确。对于真实 token `100`，预测成 `101` 和预测成 `7000` 都被视为错误；CE 不知道两个错误 token 在 Stage1 codebook 空间里是否物理接近。

## v4 训练流程

入口：

```text
stage2_bert_encoder_train_v4.py
```

v4 在原始 MLM 训练上增加两件事：

1. 使用 v3 的分阶段 mask curriculum。
2. 增加 codebook vector regression head，让 BERT 学习 Stage1 codebook 的几何结构。

整体流程：

```text
jsonl.gz
  -> dataset.py 解析 <|bwav:id|>
  -> Stage2Collator padding
  -> 按 global_step 选择 mask 策略
  -> BertMlmWithCodebookRegression
       ├── MLM head: token id prediction
       └── vector_head: codebook vector prediction
  -> CE + MSE + CosineLoss
  -> 反向传播
```

### v4 mask curriculum

v4 根据当前 `global_step` 选择 mask 策略：

| 训练阶段 | mask 形态 |
| --- | --- |
| `0 <= step < 20000` | 100% 样本使用随机单点 mask |
| `20000 <= step < 60000` | 80% 样本随机单点 mask，20% 样本连续 short-span mask |
| `step >= 60000` | 70% 样本随机单点 mask，20% 样本 short-span mask，10% 样本 long-span mask |

默认 span 长度：

```yaml
short_span_min_length: 2
short_span_max_length: 5
long_span_min_length: 6
long_span_max_length: 20
```

每条样本仍然先按 `mask_probability` 计算 mask 数量：

```text
target_mask_count = round(valid_token_count * mask_probability)
```

也就是说，v4 改变的是 mask 的形态，不是默认 15% 的总 mask 比例。

### v4 codebook vector regression

v4 会从 Stage1 tokenizer checkpoint 中读取 VQ codebook embedding：

```yaml
stage1_codebook_ckpt: "/path/to/porepgt_vqe_tokenizer.final.pth"
codebook_vocab_offset: 129
```

读取到的 codebook 形状一般是：

```text
[codebook_size, codebook_dim]
```

例如：

```text
[8196, 768]
```

v4 的模型结构：

```text
BERT last_hidden_state
  ├── MLM head
  │     -> token logits
  │     -> CE token loss
  └── vector_head
        -> pred_codebook_vec
        -> MSE / Cosine loss
```

`vector_head` 当前设计为一个线性层：

```python
torch.nn.Linear(hidden_size, codebook_dim)
```

如果 BERT hidden size 和 codebook dim 都是 `768`，它就是：

```text
Linear(768 -> 768)
```

虽然维度一样，仍然需要这个映射层，因为 BERT hidden state 空间和 Stage1 VQ codebook 空间不是同一个语义空间。`vector_head` 的作用是学习从 BERT 上下文表示到 codebook 几何空间的对齐。

v4 中真实 codebook vector 的获取方式：

```text
masked label bert_vocab_id
  -> subtract codebook_vocab_offset
  -> stage1_codebook_id
  -> lookup Stage1 codebook embedding
  -> true_codebook_vec
```

预测向量来自：

```text
corrupted input_ids
  -> BERT
  -> last_hidden_state
  -> vector_head
  -> pred_codebook_vec
```

只在 masked token 位置计算 vector loss。

### v4 loss

v4 总损失：

```text
Loss = CE_token
     + codebook_mse_weight * MSE(pred_codebook_vec, true_codebook_vec)
     + codebook_cosine_weight * CosineLoss(pred_codebook_vec, true_codebook_vec)
```

默认配置：

```yaml
codebook_mse_weight: 0.1
codebook_cosine_weight: 0.05
```

三项含义：

| loss | 作用 |
| --- | --- |
| `CE_token` | 让模型预测正确 token id。 |
| `MSE` | 让预测 codebook vector 在坐标数值上接近真实 codebook vector。 |
| `CosineLoss` | 让预测向量方向接近真实 codebook vector，更关注几何方向。 |

v4 的目标不是替代 CE，而是在 CE 之外加入 codebook 几何约束。这样模型不会只把所有错误 token 都看成同等错误，而是能感知 codebook 空间中“物理上更接近”的 token。

## 原始版和 v4 对比

| 对比项 | 原始版 | v4 |
| --- | --- | --- |
| 训练入口 | `stage2_bert_encoder_train.py` | `stage2_bert_encoder_train_v4.py` |
| 模型 | `BertForMaskedLM` | `BertMlmWithCodebookRegression` |
| mask | token 独立随机 15% | step-based curriculum + 单点/span 混合 |
| Stage1 codebook | 不需要 | 需要读取 Stage1 tokenizer checkpoint |
| 输出 | token logits | token logits + predicted codebook vector |
| loss | CE | CE + MSE + CosineLoss |
| checkpoint | 普通 HF BERT checkpoint | HF BERT checkpoint + `codebook_vector_head.bin` |
| 更适合目标 | token id 预测 | token id 预测 + 后续信号重建 |

## 配置说明

### 通用 data 配置

```yaml
data:
  train_dir: "/path/to/train"
  valid_dir: "/path/to/validation"
  num_workers: 8
  prefetch_factor: 2
  pin_memory: true
  max_cache_size: 32
```

| 参数 | 含义 |
| --- | --- |
| `train_dir` | 训练 `.jsonl.gz` 所在目录。 |
| `valid_dir` | 验证 `.jsonl.gz` 所在目录。为空时不做验证。 |
| `num_workers` | DataLoader worker 数。 |
| `prefetch_factor` | 每个 worker 预取 batch 数。 |
| `pin_memory` | CUDA 训练时通常设为 `true`。 |
| `max_cache_size` | 当前 dataset 代码实际使用的是 `max_cache_files`，如需严格生效可以在配置里改成 `max_cache_files`。 |

### 通用 model 配置

```yaml
model:
  tokenizer_path: "/path/to/tokenizer-8k.json"
  vocab_size: 8325
  mask_token_id: 4
  pad_token_id: 1
  unk_token_id: 0
  random_token_min_id: 129
  random_token_max_id: 8325
  hidden_size: 768
  num_hidden_layers: 4
  num_attention_heads: 8
  intermediate_size: 3072
  max_position_embeddings: 4096
  mask_probability: 0.15
```

| 参数 | 含义 |
| --- | --- |
| `tokenizer_path` | Stage2 BERT tokenizer JSON。 |
| `vocab_size` | BERT vocabulary 大小。 |
| `mask_token_id` | `[MASK]` token id。当前训练中使用 `4`。 |
| `pad_token_id` | padding token id。 |
| `random_token_min_id` / `random_token_max_id` | 80/10/10 中随机 token 的采样范围。通常只在 codebook token 范围内采样。 |
| `mask_probability` | 每条样本的目标 mask 比例，默认 `0.15`。 |

### v4 专用 model 配置

```yaml
model:
  stage1_codebook_ckpt: "/path/to/stage1/checkpoint"
  codebook_key:
  codebook_vocab_offset: 129
  codebook_mse_weight: 0.1
  codebook_cosine_weight: 0.05
  short_span_start_step: 20000
  long_span_start_step: 60000
  short_span_min_length: 2
  short_span_max_length: 5
  long_span_min_length: 6
  long_span_max_length: 20
```

| 参数 | 含义 |
| --- | --- |
| `stage1_codebook_ckpt` | Stage1 tokenizer checkpoint 目录，用来读取 VQ codebook embedding。 |
| `codebook_key` | 可选。为空时自动寻找类似 `vq._codebook.embed` 的权重名；自动查找失败时再手动填写。 |
| `codebook_vocab_offset` | BERT vocab id 到 Stage1 codebook id 的偏移，当前为 `129`。 |
| `codebook_mse_weight` | MSE vector loss 权重。 |
| `codebook_cosine_weight` | Cosine vector loss 权重。 |
| `short_span_start_step` | 从哪个 step 开始加入 short-span mask。 |
| `long_span_start_step` | 从哪个 step 开始加入 long-span mask。 |

### training 配置

```yaml
training:
  max_steps: 200000
  learning_rate: 5.0e-5
  weight_decay: 0.01
  warmup_steps: 1000
  lr_scheduler_type: cosine
  device_micro_batch_size: 32
  gradient_accumulation_steps: 1
  mixed_precision: "no"
  gradient_clipping: 1.0
  output_dir: "models/stage2_BERT_Encoder_v4"
  log_every_steps: 10
  eval_every_steps: 1000
  max_eval_batches: 100
  save_every_steps: 50000
```

| 参数 | 含义 |
| --- | --- |
| `max_steps` | 最大 optimizer step 数。 |
| `device_micro_batch_size` | 每张 GPU 的 batch size。 |
| `gradient_accumulation_steps` | 梯度累积步数。 |
| `mixed_precision` | `no`、`fp16`、`bf16` 等。 |
| `output_dir` | checkpoint 输出目录。 |
| `eval_every_steps` | 每隔多少 step 验证一次。 |
| `save_every_steps` | 每隔多少 step 保存一次 checkpoint。 |

## 运行方式

进入运行目录：

```bash
cd /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage2_BERT_Encoder/runs/test_zhou
```

原始版：

```bash
bash run.sh
```

等价命令：

```bash
torchrun --nproc_per_node=2 --master_port 29501 \
  /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage2_BERT_Encoder/stage2_bert_encoder_train.py \
  --config config.yaml 2>&1 | tee run.log
```

v4 版：

```bash
bash run_v4.sh
```

等价命令：

```bash
torchrun --nproc_per_node=2 --master_port 29504 \
  /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/src/poredlm/training/stage2_BERT_Encoder/stage2_bert_encoder_train_v4.py \
  --config config_v4.yaml 2>&1 | tee run_v4.log
```

建议通过环境变量设置 W&B key，不要把真实 key 写进脚本：

```bash
export WANDB_API_KEY="your_wandb_key"
```

## 输出文件

原始版 checkpoint：

```text
models/stage2_BERT_Encoder/
  step_50000/
  step_100000/
  step_best/
  step_200000/
```

v4 checkpoint：

```text
models/stage2_BERT_Encoder_v4/
  step_50000/
    config.json
    model.safetensors 或 pytorch_model.bin
    codebook_vector_head.bin
    codebook_regression_config.json
  step_best/
  step_200000/
```

如果后续只用 MLM token logits，可以直接加载 v4 checkpoint 里的 HuggingFace BERT 部分。  
如果要使用 `vector_head` 的输出，需要额外加载 `codebook_vector_head.bin`。

## 日志指标

原始版主要日志：

| 指标 | 含义 |
| --- | --- |
| `train/loss` | MLM CE loss。 |
| `train/loss_log10` | loss 的 log10，方便观察大范围变化。 |
| `train/lr` | 当前学习率。 |
| `eval/top1_accuracy` | masked token 的 top1 准确率。 |
| `eval/top5_accuracy` | masked token 的 top5 准确率。 |

v4 额外日志：

| 指标 | 含义 |
| --- | --- |
| `train/token_loss` | MLM CE loss。 |
| `train/vector_mse_loss` | codebook vector MSE loss。 |
| `train/vector_cosine_loss` | codebook vector cosine loss。 |
| `train/mask_ratio` | 实际 mask token 数 / 有效 token 数。 |
| `train/mask_phase` | 当前 curriculum 阶段，`0/1/2`。 |
| `eval/vector_mse_loss` | 验证集 vector MSE。 |
| `eval/vector_cosine_loss` | 验证集 vector cosine loss。 |

## 注意事项

1. v4 的 `stage1_codebook_ckpt` 必须和生成 Stage2 token 数据时使用的 Stage1 tokenizer 对应，否则 codebook vector loss 会对错目标空间。

2. `codebook_vocab_offset` 当前应保持为 `129`。如果 tokenizer 词表规则改变，需要同步修改。

3. `codebook_key` 一般留空即可。只有自动找不到 Stage1 codebook embedding 时，才需要手动指定 state dict 里的 key。

4. v4 当前 `eval/perplexity` 使用总 loss 计算，因此包含 CE、MSE、CosineLoss，不再是严格的 token perplexity。如果需要严格 token perplexity，应使用 `token_loss` 计算。

5. v4 的 span mask 逻辑来自 v3。如果后续训练时在 span 阶段遇到 `permutation` 未定义相关错误，需要检查 `_mask_contiguous_spans` 中剩余 token 补齐分支的缩进。

6. 训练脚本只处理 token 序列，不直接调用 Stage1 decoder。Stage1 decoder 相关的信号重建评估在 `eval/` 目录中。
