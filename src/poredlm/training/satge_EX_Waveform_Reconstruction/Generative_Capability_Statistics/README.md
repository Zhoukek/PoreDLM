# DLM Generative Capability Statistics

这个目录用于测量条件 DLM 在不同连续预测长度下的生成能力边界。对同一批 token chunks，分别在前部、中部和后部遮盖连续区间，从高斯噪声开始执行条件 ODE/SDE 生成，再用 Stage‑1 codec 将完整 token 序列还原为电信号，并计算预测区间的 waveform MSE。

## 评测设计

默认配置：

- 取 20 个 chunks，每个 chunk 使用 1200 个 content/codebook tokens；
- 前部、中部、后部的预测区间中心分别位于 chunk 的 20%、50%、80%；
- 扫描 `5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400` token；
- DLM 使用 64 步 ODE；
- 默认每次推理 2 个 chunks，可通过 `evaluation.batch_size` 根据显存调整；
- 所有长度和位置使用完全相同的 chunks；
- MSE 区间按照 codec 的实际 `downsample_rate` 计算，而不是按解码后长度做比例缩放。

区间长度增大时，区间以相同中心点向两侧扩张。例如 `total_length=1200`、`middle=0.5`、`mask_length=20` 时预测 `[590, 610)`。

## 运行

先检查并修改 `config.yaml` 中的数据集、DLM 和 tokenizer 路径，然后执行：

```bash
bash run.sh
```

常用命令行覆盖：

```bash
# 快速检查
bash run.sh --num-chunks 3 --mask-lengths 5,20,50

# 指定固定样本，便于不同 checkpoint 公平比较
bash run.sh \
  --sample-indices 0,10,25,50,100 \
  --mask-lengths 5,10,20,30,40,50,75,100,150,200,300,400 \
  --output-dir outputs/step46000
```

## 输出

输出目录包含：

- `per_sample_metrics.csv`：每个 chunk × 位置 × mask 长度的 MSE、token accuracy 和实际区间；
- `aggregate_metrics.csv`：按前/中/后及 overall 和 mask 长度聚合的 mean、median、std、四分位数和 95% CI；
- `run_summary.json`：模型路径、样本索引、完整配置摘要和聚合结果；
- `mse_vs_mask_length.png`：前/中/后平均 MSE 曲线及 95% CI；
- `mean_mse_heatmap.png`：位置 × mask 长度的平均 MSE 热力图；
- `mse_distributions.png`：各位置上不同 mask 长度的样本 MSE 箱线图。

判断“能力边界”时不建议只看平均值。可以先确定业务可接受的 MSE 阈值，再同时检查 median、95% CI 上界和箱线图长尾。不同 DLM checkpoint 对比时必须固定 `sample_indices`、`seed`、采样方法和步数。

注意：评测依赖转换后 HF 模型中修正过的条件 context attention mask。旧的 `modeling_poredlm.py` 会使推理条件结构与训练不一致，应先重新转换模型或替换模型目录中的代码。
