# 当前版 Stage2 BERT masked-token 评估

评估脚本直接读取 `runs/test/train_config.yaml`，并严格复用当前训练数据流程：

```text
data.valid_dir / data.train_dir
-> FlowMapDataset 读取 raw .npy memmap + .csv.gz 索引
-> normalize_flowmap_sample（BOS/EOS、可选 CLS、offset、截断、padding）
-> mask 内容 token -> Stage2MaskedSignalLM -> masked token accuracy
```

这里不再把 token shard 当成普通 `np.load()` 文件，也不要求输入原始 signal。
当前 FlowMap shard 只保存 token，因此波形对比是将“原始 token”和“BERT 修复 token”
分别送入同一个 Stage1 codec decoder 得到的两条重建波形；运行时需要提供 codec。

## 运行

```bash
cd /Users/kexuanzhou/project/PoreDLM/src/poredlm/training_public/stage2_BERT_trian/eval

BERT=/path/to/stage2/models/step_xxx \
CODEC=/path/to/pore_vq_codec_checkpoint \
TRAIN_CONFIG=../runs/test/train_config.yaml \
bash run_eval_masked_tokens.sh
```

默认读取 `data.valid_dir`。读取训练集或临时覆盖数据目录：

```bash
SPLIT=train bash run_eval_masked_tokens.sh

DATA_DIR=/path/to/flowmap/eval bash run_eval_masked_tokens.sh
```

评估第 100 条开始的 20 条数据，随机 mask 15%：

```bash
SAMPLE_INDEX=100 NUM_SAMPLES=20 \
MASK_MODE=random MASK_PROBABILITY=0.15 \
bash run_eval_masked_tokens.sh
```

连续 mask：

```bash
MASK_MODE=contiguous MASK_TOKEN_START=15 MASK_TOKEN_LENGTH=4 \
bash run_eval_masked_tokens.sh
```

连续 mask 的图片默认只显示 mask 区间左右各 20 个上下文 token。可调整为：

```bash
PLOT_CONTEXT_TOKENS=10 bash run_eval_masked_tokens.sh
```

设置 `PLOT_CONTEXT_TOKENS=-1` 才会显示整条 token 序列。

## 输出

- `metrics.json`：总体及逐样本 masked-token accuracy、非法 token 预测比例。
- `first_sample_result.npz`：第一条样本的 raw/normalized/corrupted/repaired token。
- `token_comparison.png`：第一条样本的目标 token 与 BERT 修复 token 对比。
- `waveform_comparison.png`：原始 token 与 BERT 修复 token 分别经过当前 codec decoder
  得到的重建波形对比；连续 mask 时只展示 mask 波形区间及两侧上下文。

`invalid_prediction_rate` 表示模型 top-1 预测落在配置中
`random_token_min_id:random_token_max_id` 内容 token 范围之外的比例。准确率使用模型
完整词表上的真实 top-1，不会通过限制 logits 范围人为提高结果。
