# 三种读出策略一起评测

三种策略应分别拟合和测评，使用相同样本、physical-read划分及分类配置，再比较结果。不能把三种表征混在同一个分类结果中解释。

| scheme | 模型输入 | 读取范围 |
|---|---|---|
| chunk_center | 完整chunk | 目标5-mer的中心碱基位置 |
| chunk_5mer | 同一个完整chunk | 目标5-mer完整区间 |
| short_5mer | 冻结信号裁出的5-mer短窗 | 全部内容token |

[configs/](configs/)保留 V003 的三份策略配置，以及 V006 的 `chunk_center` 配置。V003 三份配置使用同一冻结 Stone 语料的三个 DNA 域，阶段统一为连续 CNN、BERT、DLM；同中心 G 对照预选 AAGAA、CCGCC、TTGTT 三个标签。分类任务与指标定义见[工具说明](README.md)。

在已安装NanoRepProbe、具备对应表征文件的环境中，从项目根目录运行：

```bash
for scheme in chunk_center chunk_5mer short_5mer; do
  nanorepprobe run \
    --config "script/NanoRepProbe/configs/V003.${scheme}.json" \
    --output "/path/to/new_results/V003/${scheme}" --threads 4
done
```

更换模型时须导出三个策略各自的数组并更新配置，不能仅修改 scheme 字段却复用同一个数组。运行前可用 `nanorepprobe audit --config CONFIG --output AUDIT.json` 核对样本索引、物理 read 划分和输入文件；新结果写入独立目录。

从 center 到整段 5-mer 只改变读取范围；从整段输入到短窗还改变外部上下文、位置编号、CNN 边界和 stride 相位。这里 V003 短窗配置引用的冻结表征未额外 clamp 或标准化；使用其他提取结果时应记录其实际预处理。跨模型比较须保持样本和读出协议一致，同时改变模型与预处理时，不能将差异单独归因于模型。
