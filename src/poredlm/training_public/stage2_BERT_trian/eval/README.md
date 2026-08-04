# 当前版 Stage2 BERT masked-token 评估

评估流程与 `train_stage2_bert_memmap.py` 对齐：

```text
原始 signal -> PoreVQCodec.encode_signal -> codebook id
-> 按 training_config.yaml 添加 token_offset、BOS/EOS、可选 CLS
-> mask 内容 token -> Stage2MaskedSignalLM 预测
-> 去除 offset 得到 codebook id -> PoreVQCodec.decode_token
```

## 运行

先修改或通过环境变量覆盖三个路径：

```bash
CODEC=/path/to/codec \
BERT=/path/to/stage2/models/step_xxx \
INPUT_NPY=/path/to/signal_chunks.npy \
bash run_eval_masked_tokens.sh
```

连续 mask 默认随机选择起点。指定起点和长度：

```bash
MASK_TOKEN_START=15 MASK_TOKEN_LENGTH=4 bash run_eval_masked_tokens.sh
```

随机 mask 15%：

```bash
MASK_MODE=random MASK_PROBABILITY=0.15 bash run_eval_masked_tokens.sh
```

脚本默认读取 `<BERT>/training_config.yaml`，也可向 Python 脚本传
`--training-config /path/to/train_config.yaml`。这里的配置决定 `token_offset`、
special token id 和是否加入 CLS，必须与该 checkpoint 的训练配置一致。

## 输出

- `metrics.json`：masked token accuracy、信号 MSE 等指标。
- `result.npz`：原始/预测 token、mask、原始/重建信号。
- `token_comparison.png`：目标 token 与 BERT 修复 token 对比。
- `signal_comparison.png`：原信号、codec baseline、BERT 修复结果对比。

若 BERT 序列超过 `max_position_embeddings`，脚本会明确报错；此时减小
`SIGNAL_LENGTH`。标准 6000-sample、stride=5 的输入约为 1202 个 token，能放入
当前 1536 长度模型。
