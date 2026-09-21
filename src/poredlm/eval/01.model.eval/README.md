# 模型测评

日常运行使用 **[evaluate_model.sh](evaluate_model.sh)**。通常只改脚本开头的
`MODEL_DIR`、`STRATEGY` 和 `GPUS`，然后在项目根目录执行：

```bash
bash 01.capability/01.model_eval/evaluate_model.sh
```

也可直接传参数，无需编辑脚本：

```bash
bash 01.capability/01.model_eval/evaluate_model.sh \
  --model-dir /mnt/zzbnew/poregpt/models/HF_VQE768C08A001_DNADLLM_V007 \
  --strategy apple --gpus 0 1 2 3
```

默认每域选择 ACGTA、AAAAA、CCCCC、GGGGG，每类 100 条；加 `--all-kmers`
统计全部 1024 类。相同配置可续跑；修改采样或模式时用 `--output-dir` 指定新批次。
只查看参数：`bash 01.capability/01.model_eval/evaluate_model.sh --help`。

## 三种模式

| 模式 | 输入与池化 |
|---|---|
| `chunk_center` | 完整 chunk，池化中心碱基区间 |
| `chunk_5mer` | 完整 chunk，池化整段 5-mer 区间 |
| `short_5mer` | 裁出 5-mer 单独推理，池化全部有效 token |

每模式输出距离报告和单/双/三类图，合计 **9 张图，各有 PNG/SVG**。
默认结果位置：`<模型名>/runs/<日期>_nanorepdist_modes/`，从其中的
`README.md` 和 `reports/PLOTS.md` 查看结果。

## 已有结果

| 内容 | 入口 |
|---|---|
| V007 / Apple 距离测评与三模式图 | [模型主页](HF_VQE768C08A001_DNADLLM_V007/README.md) · [指标汇总](HF_VQE768C08A001_DNADLLM_V007/reports/summary.md) |
| V003 / Stone 全1024类三策略空间距离 | [完整结果](HF_VQE768C08A001_DNADLLM_V003/runs/20260918_full1024_space/README.md) · [汇总图与1024类曲线](HF_VQE768C08A001_DNADLLM_V003/runs/20260918_full1024_space/reports/PLOTS.md) |
| V003 / Stone 距离测评 | [模型主页](HF_VQE768C08A001_DNADLLM_V003/README.md) · [全量报告](HF_VQE768C08A001_DNADLLM_V003/runs/20260916_full5mer/distance/report.md) · [chunk 图](HF_VQE768C08A001_DNADLLM_V003/runs/20260916_kmer_space/reports/PLOTS.md) · [短窗图](HF_VQE768C08A001_DNADLLM_V003/runs/20260917_short5mer/reports/PLOTS.md) |
| V006 / Apple 距离测评 | [模型主页](HF_VQE768C08A001_DNADLLM_V006/README.md) · [全量报告](HF_VQE768C08A001_DNADLLM_V006/runs/20260916_full5mer/distance/report.md) · [chunk 图](HF_VQE768C08A001_DNADLLM_V006/runs/20260916_kmer_space/reports/PLOTS.md) · [短窗图](HF_VQE768C08A001_DNADLLM_V006/runs/20260917_short5mer/reports/PLOTS.md) |

## 文件用途

- `evaluate_model.sh`：运行入口，设置模型、策略和 GPU。
- `scripts/evaluate_model.py`：调用 NanoRepDist，并补齐默认路径。
- `HF_*/runs/`：保留的模型距离测评结果；各模型首页提供图表索引。

跨域分类、分类三策略对照及相关临时审计结果已清理。这里保留已有的模型距离结果；
之后运行统一脚本，会生成新的 `<日期>_nanorepdist_modes` 批次。

V003 的全 1024 类距离报告复用了旧批次的表征，因此
`HF_…V003/runs/20260917_three_strategy/` 仅保留输入缓存，顶层
`20260917_three_strategy/` 仅保留提取配置和源码。
`evaluate.py`、`kmer_space.py`、`three_strategy.py`、`models.json` 是这些保留结果的
复现依赖，仍保留原路径；日常运行使用上面的 Shell 脚本。

[完整参数与协议](../../script/NanoRepDist/docs/model-evaluation.md) ·
[返回能力测评目录](../README.md)
