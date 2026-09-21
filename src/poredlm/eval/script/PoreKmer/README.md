# PoreKmer

PoreKmer 0.3.0 是一个可独立安装的 **纳米孔事件上下文 5-mer 分类与电流建表工具包**。它复用冻结基座产生的 hidden，比较单事件、三事件和五事件对中心 5-mer 的识别；提供数据准备、审计、训练、预测封存、评估和模型比较的完整实验链路。

输入支持历史 dense hidden 缓存，以及 **NanoSignalPrep + NanoSignalAlign 的 aligned JSONL 和对应逐帧 hidden**。模型推理由上游完成；本包读取其输出，组织事件、训练分类头和评估，不读取 FAST5/POD5 或运行基座编码器。

历史 dense 入口沿用 `unverified` 标签注册。新 aligned 入口保留上游 `phase_calibrated` 状态、校准置信度和哈希，按已校准区间取参考方向标签；这一状态不表示对齐准确性已经独立验证。

## 安装

需要 Python 3.10+、NumPy 和 PyTorch。包本身不依赖 `motor`、Transformers 或 V003 权重。以下命令从本仓库根目录执行：

```bash
python -m pip install ./script/PoreKmer
porekmer --version
porekmer --help
porekmer doctor
```

安装会按包元数据解析依赖；离线环境应提前准备依赖或使用已配置的 Python 环境。CUDA 版 PyTorch 的安装需要匹配本机环境，工具不会自动安装 GPU 驱动。

也可使用 `python -m porekmer`，与 `porekmer` 等价。安装后可从任意工作目录启动，输入输出路径仍需正确。包尚未发布到公共包索引；不要将上述本地安装命令替换成未经确认的 `pip install porekmer`。公开名称可用性和开源许可证尚未确定。

拿到 wheel 后也可安装单个文件：

```bash
python -m pip install /path/to/porekmer-0.3.0-py3-none-any.whl
```

wheel 包含运行实现；对应版本的源码归档另含中文文档、配置示例和测试，适合完整交付。已有 `dist/` 中的旧版本文件不会随源码自动更新。分发边界见 [DISTRIBUTION.md](DISTRIBUTION.md)。

## 接入 NanoSignalPrep + NanoSignalAlign

```text
Prep + Align 的 aligned JSONL + 模型逐帧 hidden.npy
    → prepare-aligned → 事件 manifest + features/truth NPZ
    → train / predict / score / compare
```

从 [aligned.source.example.json](examples/aligned.source.example.json) 配置记录身份、split、JSONL 行号、hidden 路径和 SHA256；然后执行：

```bash
porekmer prepare-aligned --source-manifest /path/to/aligned.source.json \
  --output-dir /path/to/new_event_corpus
porekmer audit --manifest /path/to/new_event_corpus/manifest.json --verify-files
```

也可使用 [aligned.experiment.example.json](examples/aligned.experiment.example.json) 的 `aligned_manifest` 直接运行完整实验。新输入使用参考序列方向的中心 5-mer 标签，支持 `direct/reverse`，不会对校准区间二次平移。默认排除参与上游校准的物理 reads。输入 hidden 必须保留连续时间轴；已池化的单样本 `[N,D]` 距离评估数组不能替代逐帧 hidden。

输入信号可以保持 Apple 或其他预处理数值，输出通用 `current` 表。只有显式提供独立 `current_ref` 并声明 `stone_unclamped` 才输出 Stone 表，工具不推断或重新执行标准化。字段、坐标与示例见 [Prep/Align 接入指南](docs/ALIGNED_INPUT.zh-CN.md)。

## 一条命令运行完整实验

先检查配置和计划，再运行小规模冒烟测试：

```bash
porekmer run --config script/PoreKmer/examples/smoke.json --dry-run
porekmer run --config script/PoreKmer/examples/smoke.json
```

示例复用本项目的 Cyclone UNMOD dense hidden 缓存，每个 split 最多 2 个 read，各模型训练 1 个 epoch。它验证接口链路，**不是代表性抽样或性能结果**。源数据不随包分发；在其他机器上使用前应修改路径。

运行顺序为：准备并校验事件语料（或复用既有 manifest）、训练全部候选模型、冻结全部预测、封存候选集合、逐模型评分、统一比较。不会先评分一个模型再决定增加哪些候选模型。

全量探索性配置：

```bash
porekmer run --config script/PoreKmer/examples/full_experiment.json --dry-run
porekmer run --config script/PoreKmer/examples/full_experiment.json
```

该配置不限制 read 数，包含单事件 cosine/linear 对照和同维度的 1/3/5 事件 context 头。默认 CPU；确认 CUDA 可用后可同时修改 `train.device`、`predict.device` 为 `cuda:0`。全量运行时间和资源需求取决于语料与设备，此处没有性能保证。

配置中的相对路径以 **配置文件所在目录** 为基准，不以启动命令的工作目录为基准。历史示例的输入输出路径沿用原仓库布局，使用前应修改到当前实际目录。每次运行必须选择尚不存在的输出目录；不覆盖、自动清空或自动续训旧结果。失败后可能留下中间产物，应保留诊断信息并选择新目录重试。

`--dry-run` 用于检查配置与展示执行计划，不等于完整数据审计、训练成功或物理注册正确。

## 模型输入输出

已有 6000 点 chunk 的 V003 hidden 为 `[1200, 768]`。PoreKmer 在同一事件内对完整有效帧作算术平均，再按真实事件顺序构造 `[N, 5, 768]` 窗口。单事件和三事件模型屏蔽不用的邻居，所有模型使用完全相同的中心事件。

输入没有碱基字符或邻居标签；输出为中心事件的 `[N, 1024]` 类别分数。1024 类对应按 `ACGT` 字典序编码的全部 5-mer。`model.encode(x)` 可导出中心事件表示。

| 分类头 | 窗口 | 作用 |
|---|---|---|
| `cosine` | 1 | 直接从冻结事件 hidden 读取类别方向 |
| `linear` | 1 | 普通线性分类 `Wh+b` |
| `context` | 1、3、5 | 投影、固定位置编码、一层 Transformer、余弦分类 |

同一 `projection_dim` 下，三个 context 窗口的参数量相同。默认投影维度 128，所以 context 头导出的类别方向为 `[1024, 128]`，并非原始空间的 `[1024, 768]`。类别向量是训练得到的分类方向，不保证等于经验 hidden 簇中心。基座保持冻结；context 头会学习新的事件表示。

```python
import torch
from porekmer.models import EventContextClassifier

model = EventContextClassifier(
    hidden_dim=768, head="context", window_size=5, projection_dim=128
)
x = torch.zeros(2, 5, 768)  # 仅演示接口，不是实际实验数据
logits = model(x)          # [2, 1024]
embedding = model.encode(x)  # [2, 128]
```

## 命令与产物

| 命令 | 作用 |
|---|---|
| `doctor` | 检查运行环境；不能替代实际数据和模型验证 |
| `run` | 按 JSON 配置执行完整候选集合实验 |
| `prepare` | dense NPZ 转事件特征与真值分离的语料 |
| `prepare-aligned` | aligned JSONL + 逐帧 hidden 转事件语料 |
| `audit` | 检查数据契约，可加 `--verify-files` 校验内容 |
| `train` | 只使用 train/validation 训练和选模 |
| `predict` | 不打开真值 sidecar，生成 prediction/test 预测 |
| `freeze-family` | 评分前封存所有候选模型的预测 |
| `score` | 事件分类指标、read-bootstrap、电流表及参考表 |
| `compare` | 在相同中心事件与共同类别面板上比较模型 |

各阶段保留独立 CLI，例如：

```bash
porekmer prepare --help
porekmer train --help
porekmer score --help
```

主要产物包括语料 `manifest.json`、feature/truth NPZ、模型 `checkpoint.pt`、`class_vectors.npy`、训练配置与历史、`predictions.json`、family lock、`scores.json`、预测与参考电流 TSV、`comparison.json`。输入、特征、模型与预测记录内容哈希，便于核对来源。只加载可信来源的模型文件。

`run` 的输出布局为：

```text
output_dir/
├── config.input.json            原始配置快照，与 SHA256 绑定
├── config.normalized.json       解析默认值和绝对路径后的配置
├── plan.json                    实验执行计划
├── corpus/                      从 dense 源准备的语料；复用 manifest 时不复制
├── models/<name>/               checkpoint、类别向量和训练记录
├── predictions/<name>/          各候选的冻结预测
├── family.json                  候选预测集合封存
├── scores/<name>/               分类指标和电流 TSV
├── comparison.json              相同中心、共同类别面板比较
├── run_status.json              当前状态或失败阶段
└── run_report.json              完成后的实验汇总
```

Stone 表只收集**中心事件**的未 clamp 电流，先按 read/类别取中位数，再跨 read 取中位数；邻居事件的电流不混入中心类别。未预测出的类别保留缺失值，不强制补齐 1024 类。

## 结果能说明什么

- 这是已知 move 边界、经过 alignment 筛选的条件实验，即使预测阶段不读取标签，也不是纯信号端到端盲测。
- 五事件输入是五个连续状态，不是同一个 5-mer 的五次重复测量。更多上下文是否有效必须比较验证，不能假设窗口越大越好。
- 训练按 read 平衡：read 内事件损失取平均，再跨 read 平均；完整事件只计一次，不再叠加逐帧 inverse-dwell 权重。
- 同时看事件准确率、read-macro、类别召回、支持数和表误差。表相关性高不能单独证明序列分类准确或物理偏移正确。
- 旧逐帧 Top1 与本工具事件级 Top1 不是同一统计量，不可直接作差。

输入细节见 [数据格式](docs/INPUT_FORMAT.zh-CN.md)，配置、实验和解释边界见 [实验指南](docs/EXPERIMENT.zh-CN.md)，旧入口迁移见 [迁移指南](docs/MIGRATION.zh-CN.md)。

## 开发与分发

采用 `src/porekmer` 布局，可构建 wheel 和源码分发包。包安装后运行测试：

```bash
python -m unittest discover -s script/PoreKmer/tests -v
```

已有 setuptools、wheel 和运行依赖时，可离线运行仓库自带的构建脚本。默认先执行独立测试，再生成 wheel、源码归档和 SHA256 清单；不会发布或覆盖已有分发目录：

```bash
python script/PoreKmer/scripts/build_dist.py
```

也可使用通用构建前端：

```bash
python -m pip install build
python -m build script/PoreKmer
```

语料、预训练权重和实验结果不打包分发。合成测试证明代码契约，小规模真实数据测试证明兼容性；两者都不能代替正式模型效果评估。版本变更见 [CHANGELOG](CHANGELOG.md)。
