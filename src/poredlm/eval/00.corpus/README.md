# 固定电信号表征评测语料

语料按输入策略分为两个并列目录，统一通过 `corpus.py` 操作。

| 目录 | 信号策略 | 说明与图表 |
|---|---|---|
| `stone/` | 原版预计算信号，`float16` | [使用说明](stone/README.md) · [k-mer 折线图](stone/reports/PLOTS.md) |
| `apple/` | 原始 pA 完整 read 经 NanoSignalPrep Apple，再切 chunk，`float32` | [使用说明](apple/README.md) · [k-mer 折线图](apple/reports/PLOTS.md) |

两版均包含 Cyclone DNA、ONT R10 HG002、DNA 扩增子和 Cyclone RNA 四域，覆盖全部 **1024 种 5-mer**，每域 **98,164 条**、合计 **392,656 条**。固定样本清单和坐标相同，可用于比较模型及输入标准化策略。

`stone` 是原版语料的目录名；其处理模式仍为 `precomputed`，保留已有数值。各来源历史标准化未全部确认，尤其 ONT R10 的抽查结果呈固定均值/尺度的仿射标准化，不能将四域都视为已确认的 Stone MAD。

```text
00.corpus/
  README.md
  corpus.py                     # 唯一操作脚本
  stone/
    README.md
    SHA256SUMS
    config/ reports/ provenance/ logs/
    cyclone-dna/ ont-r10-hg002/ dna-amplicon/ cyclone-rna/
  apple/
    README.md
    SHA256SUMS
    config/ reports/ provenance/ logs/
    cyclone-dna/ ont-r10-hg002/ dna-amplicon/ cyclone-rna/
```

以下命令在项目根目录运行；`verify`、`plot` 默认选择 `stone`，也可用 `--root` 指定其他语料目录。

```bash
python 01.capability/00.corpus/corpus.py verify --strategy stone
python 01.capability/00.corpus/corpus.py verify --strategy apple
python 01.capability/00.corpus/corpus.py plot --strategy apple --overwrite
```

Apple 制备默认读取 `stone/`，输出到 `apple/`，原始 FAST5/POD5 映射位于 `apple/config/apple.sources.json`：

```bash
/root/miniconda3/envs/bonito_dna/bin/python 01.capability/00.corpus/corpus.py apple --workers 8
```

每版独立保存校验和、报告、日志及 NanoRepDist 配置模板。加载 manifest 内相对路径时，分别以 `stone/` 或 `apple/` 为语料根目录；模型推理缓存还需包含输入策略，不能只按共享的 sample_id 复用。模型表征和评测结果统一保存在 [01.model_eval](../01.model_eval/README.md)，按模型和运行批次分别归档。V003 使用 stone，V006、V007 使用 apple。

规划新实验时可使用 [实验规划清单](../docs/evaluation-checklist.md)；目录总览见 [能力评估](../README.md)。
