# Stage2 BERT Masked Reconstruction Eval 使用说明

这个目录用于评估 Stage1 tokenizer + Stage2 BERT 的 masked token 重建效果。整体流程是：

```text
输入原始信号 .npy -> 选择一条信号 -> 截取指定长度
-> Stage1 tokenizer 得到 codebook token id
-> 在 token 上做 mask
-> Stage2 BERT 预测被 mask 的 token
-> 用预测后的 token 经过 Stage1 decoder 重建信号
-> 保存 npz 结果和对比图
```

注意：这里的 mask 是发生在 Stage1 tokenizer 之后的 token id 上，不是直接在原始信号点上 mask。图片里标出的信号区域，是把被 mask 的 token 按 Stage1 的下采样率映射回原始信号坐标后得到的区域。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `masked_reconstruction_stage1_stage2.py` | v1 评估脚本：按自定义百分比随机 mask Stage1 token。 |
| `run_masked_reconstruction_v1.sh` | v1 的 Linux 运行脚本。 |
| `masked_reconstruction_stage1_stage2_v3.py` | v3 评估脚本：mask 一个连续 token 区间，并额外生成 token id 对比图。 |
| `run_masked_reconstruction_v3.sh` | v3 的 Linux 运行脚本。 |

## 快速运行

先进入 eval 目录：

```bash
cd /mnt/zzbnew/rnamodel/shenhaojie/PoreDLM/src/poredlm/training/stage2_BERT_Encoder/eval
```

运行随机比例 token mask：

```bash
bash run_masked_reconstruction_v1.sh
```

运行连续 token 区间 mask：

```bash
bash run_masked_reconstruction_v3.sh
```

运行前请先打开对应的 `.sh` 脚本，确认路径和参数是否符合当前服务器环境，尤其是 `STAGE1_CKPT`、`STAGE2_BERT`、`INPUT_NPY`、`PYTHON_SCRIPT` 和 `OUTPUT_DIR`。

## 两种评估方式

### v1：随机百分比 token mask

对应文件：

```text
masked_reconstruction_stage1_stage2.py
run_masked_reconstruction_v1.sh
```

适合评估“随机 mask 一定比例 token 后，BERT 整体恢复能力怎么样”。例如 `MASK_PERCENTAGE="15"` 表示随机 mask 15% 的 Stage1 token。

主要流程：

```text
信号截取 -> Stage1 tokenizer -> 随机选择一定比例 token 置为 [MASK]
-> BERT 预测 -> 替换被 mask 的 token -> Stage1 decoder -> 重建信号
```

### v3：连续 token 区间 mask

对应文件：

```text
masked_reconstruction_stage1_stage2_v3.py
run_masked_reconstruction_v3.sh
```

适合评估“指定某一段连续 token 被 mask 后，BERT 对这个局部区域的恢复能力”。例如：

```bash
MASK_TOKEN_START="9"
MASK_TOKEN_LENGTH="1"
```

表示从 token index 9 开始，mask 1 个 token。

v3 会额外生成一张 token id 对比图，对比：

```text
Stage1 原始 codebook id
BERT 预测后的 token id - 129
二者差值
```

## Shell 脚本参数

### 路径参数

| 参数 | 含义 |
| --- | --- |
| `STAGE1_CKPT` | Stage1 tokenizer checkpoint 路径，通常是 `porepgt_vqe_tokenizer.final.pth` 目录。 |
| `STAGE2_BERT` | Stage2 BERT checkpoint 路径，例如 `step_best`。 |
| `INPUT_NPY` | 输入信号 `.npy` 文件，可以是一条 1D 信号，也可以是多条信号组成的 2D chunks。 |
| `PYTHON_SCRIPT` | 要运行的 Python 评估脚本路径。 |
| `OUTPUT_DIR` | 输出目录。 |
| `OUTPUT_NPZ` | 保存重建结果、token、mask、指标的 `.npz` 文件。 |
| `OUTPUT_PLOT` | 保存信号重建对比图。 |

### 通用运行参数

| 参数 | 含义 |
| --- | --- |
| `INPUT_INDEX` | 当 `INPUT_NPY` 是二维 chunks 时，选择第几行作为输入信号。 |
| `INPUT_MODE` | 输入读取方式：`auto` 自动判断；`row` 按行取一条信号；`flatten` 将数组拉平成一条长信号。 |
| `DEVICE` | 推理设备，例如 `cuda:0` 或 `cpu`。 |
| `SIGNAL_START` | 从输入信号的哪个 sample index 开始截取。 |
| `SIGNAL_LENGTH` | 截取多少个信号点。v1 脚本中默认 Python 参数是 500；v3 脚本默认 Python 参数是 1000，但 shell 脚本可以覆盖。 |
| `MASK_TOKEN_ID` | BERT 使用的 `[MASK]` token id。当前训练代码通常使用 `4`。 |
| `MAX_LENGTH` | BERT 单次推理窗口长度，通常设为 `512`。 |
| `TOKEN_BATCH_SIZE` | Stage1 tokenizer 分块 token 数，长信号分块推理时使用。 |
| `PLOT_START` | 对比图从哪个信号 sample index 开始画。 |
| `PLOT_NUM_SAMPLES` | 对比图画多少个信号点；设为 `<=0` 表示画到结尾。 |
| `SEED` | 随机种子，控制随机 mask 位置。 |

### v1 专用参数

| 参数 | 含义 |
| --- | --- |
| `MASK_PERCENTAGE` | 随机 mask 的 token 百分比，范围 0 到 100。例如 `15` 表示 mask 15%。 |

对应 Python 参数是：

```bash
--mask-percentage "${MASK_PERCENTAGE}"
```

也可以不用百分比，直接传概率：

```bash
--mask-probability 0.15
```

如果同时提供 `--mask-percentage` 和 `--mask-probability`，脚本会优先使用 `--mask-percentage`。

### v3 专用参数

| 参数 | 含义 |
| --- | --- |
| `MASK_TOKEN_START` | 连续 mask 区间的起始 token index。设为负数或不传时随机选择。 |
| `MASK_TOKEN_LENGTH` | 连续 mask 的 token 数量。 |

例如：

```bash
MASK_TOKEN_START="9"
MASK_TOKEN_LENGTH="4"
```

表示 mask token `[9, 13)`，也就是 9、10、11、12 这 4 个 token。

## Python 脚本参数

### `masked_reconstruction_stage1_stage2.py`

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--stage1-ckpt` | 必填 | Stage1 tokenizer checkpoint。 |
| `--stage2-bert` | 必填 | Stage2 BERT checkpoint。 |
| `--input-npy` | 必填 | 输入信号 `.npy`。 |
| `--output-npz` | 必填 | 输出 `.npz`。 |
| `--output-plot` | `output-npz` 同名 `.png` | 信号对比图。 |
| `--device` | 自动选择 CUDA/CPU | 推理设备。 |
| `--input-index` | `0` | 多行 chunks 时选择第几行。 |
| `--input-mode` | `auto` | `auto`、`row` 或 `flatten`。 |
| `--signal-start` | `0` | 信号截取起点。 |
| `--signal-length` | `500` | 信号截取长度；`<=0` 表示截取到结尾。 |
| `--mask-probability` | `0.15` | token mask 概率，范围 0 到 1。 |
| `--mask-percentage` | 不启用 | token mask 百分比，范围 0 到 100，优先级高于 `--mask-probability`。 |
| `--mask-token-id` | `4` | BERT `[MASK]` token id。 |
| `--max-length` | BERT 配置值 | BERT 窗口长度。 |
| `--token-batch-size` | `8000` | Stage1 tokenizer 分块 token 数。 |
| `--plot-start` | `0` | 画图起点。 |
| `--plot-num-samples` | `500` | 画图长度。 |
| `--seed` | `42` | 随机种子。 |

### `masked_reconstruction_stage1_stage2_v3.py`

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--stage1-ckpt` | 必填 | Stage1 tokenizer checkpoint。 |
| `--stage2-bert` | 必填 | Stage2 BERT checkpoint。 |
| `--input-npy` | 必填 | 输入信号 `.npy`。 |
| `--output-npz` | 必填 | 输出 `.npz`。 |
| `--output-plot` | `output-npz` 同名 `.png` | 信号对比图。 |
| `--output-token-plot` | `result_token_compare.png` | token id 对比图。 |
| `--device` | 自动选择 CUDA/CPU | 推理设备。 |
| `--input-index` | `0` | 多行 chunks 时选择第几行。 |
| `--input-mode` | `auto` | `auto`、`row` 或 `flatten`。 |
| `--signal-start` | `0` | 信号截取起点。 |
| `--signal-length` | `1000` | 信号截取长度。 |
| `--mask-token-length` | `40` | 连续 mask 的 token 数量。 |
| `--mask-token-start` | 随机 | 连续 mask 的起始 token index；负数表示随机。 |
| `--mask-token-id` | `4` | BERT `[MASK]` token id。 |
| `--max-length` | BERT 配置值 | BERT 窗口长度。 |
| `--token-batch-size` | `8000` | Stage1 tokenizer 分块 token 数。 |
| `--plot-start` | `0` | 画图起点。 |
| `--plot-num-samples` | `1000` | 画图长度。 |
| `--seed` | `42` | 随机种子。 |

## 输出结果

运行完成后通常会得到：

| 输出 | 内容 |
| --- | --- |
| `result.npz` | 原始信号、Stage1 baseline 重建信号、BERT 重建信号、原始 token、修复后 token、mask 位置、MSE、token accuracy 等。 |
| `compare.png` | 原始信号、BERT 重建信号、Stage1 直接重建信号的对比图，并用黄色区域标注被 mask token 对应的信号位置。 |
| `result_token_compare.png` | v3 额外输出：Stage1 token id、BERT token id - 129、二者差值的柱状图。 |

`result.npz` 中常用字段：

| 字段 | 含义 |
| --- | --- |
| `original_signal` | 截取后的原始信号。 |
| `baseline_stage1_signal` | 不经过 BERT mask，直接 Stage1 encode/decode 的重建信号。 |
| `reconstructed_signal` | mask 后经过 BERT 预测，再 decode 得到的信号。 |
| `original_codebook_ids` | Stage1 tokenizer 输出的原始 codebook id。 |
| `repaired_codebook_ids` | BERT 修复 mask 后的 codebook id。 |
| `masked_positions` | 哪些 token 被 mask。 |
| `corrupted_bert_vocab_ids` | 送入 BERT 的 token id，其中 mask 位置被替换为 `MASK_TOKEN_ID`。 |
| `mask_signal_spans` | v1 中随机 mask token 映射回信号后的连续区间。 |
| `mask_token_start` / `mask_token_end` | v3 中连续 mask 的 token 起止位置。 |
| `token_accuracy_masked` | 只在被 mask token 上统计的预测准确率。 |
| `signal_mse` | BERT 重建信号与原始信号的 MSE。 |
| `stage1_baseline_mse` | Stage1 直接重建信号与原始信号的 MSE。 |

## 常见问题

### 1. `unrecognized arguments: --input-index`

说明服务器上实际运行的 Python 脚本还是旧版本，不支持 `--input-index` 等参数。检查 `PYTHON_SCRIPT` 指向的文件，把更新后的脚本同步到该路径，或者把 shell 里的 `PYTHON_SCRIPT` 改成当前 eval 目录下的脚本。

### 2. shell 脚本运行出现 `$'\r': command not found`

说明脚本是 Windows CRLF 换行。服务器上可以执行：

```bash
sed -i 's/\r$//' run_masked_reconstruction_v1.sh
sed -i 's/\r$//' run_masked_reconstruction_v3.sh
```

### 3. 输入 `.npy` 是二维数组

如果输入形状类似 `(29021, 5000)`，表示有 29021 条信号，每条 5000 个点。用：

```bash
INPUT_INDEX=1
INPUT_MODE="auto"
```

脚本会取第 1 行作为一条信号。

### 4. mask 位置预测出来都是同一个 token

这不一定是代码错误。常见原因包括：

- 使用 `argmax` 解码，模型会倾向选择最高概率 token。
- 连续 mask 太长，上下文不足时模型可能退化到高频 token。
- 训练数据 token 分布不均衡，某些 token 频率很高。
- `MASK_TOKEN_ID` 和训练时不一致。

建议先用 v3 把 `MASK_TOKEN_LENGTH` 设为 `1` 或 `2`，确认单 token mask 是否正常；再逐步增大连续 mask 长度。当前脚本默认使用 `MASK_TOKEN_ID=4`。

## 推荐检查顺序

1. 确认 `STAGE1_CKPT`、`STAGE2_BERT`、`INPUT_NPY` 都存在。
2. 确认 `PYTHON_SCRIPT` 指向的是你想运行的最新版脚本。
3. 先用较短信号测试，例如 `SIGNAL_LENGTH="500"`。
4. v1 先用 `MASK_PERCENTAGE="15"`。
5. v3 先用 `MASK_TOKEN_LENGTH="1"`，观察 token id 对比图。
6. 如果结果正常，再增大 `SIGNAL_LENGTH`、`MASK_PERCENTAGE` 或 `MASK_TOKEN_LENGTH`。
