# 实验配置与验证策略

此文是运行协议，不是性能报告。目标是在冻结 V003、固定中心事件的条件下，比较事件内聚合与相邻事件读出是否帮助中心 5-mer 分类和 Stone 建表。

## 1. 配置文件

`porekmer run --config PATH` 接收 JSON 对象。相对路径均相对于配置文件目录解析。

| 配置项 | 含义 |
|---|---|
| `schema_version` | 当前为 1 |
| `source_manifest` | 历史 dense hidden manifest；三个输入字段选一个 |
| `aligned_manifest` | NanoSignalAlign JSONL + 逐帧 hidden 输入清单；见 [接入指南](ALIGNED_INPUT.zh-CN.md) |
| `manifest` | 已准备好的事件 manifest；跳过重新准备 |
| `output_dir` | 必须不存在的新实验目录 |
| `allow_unverified_register` | 默认 false；明确承认未校准注册风险后才能探索运行 |
| `max_reads_per_split` | 可选调试截取，仅适用于 dense 或 aligned 源语料准备；不是代表性采样 |
| `allow_calibration_reads` | 默认 false，仅用于 aligned 输入，显式保留参与上游校准的 reads |
| `models` | 至少两个候选；省略时为 context 窗口 1/3/5 |
| `train` | 训练参数 |
| `predict` | 推理参数 |
| `score` | 评分参数 |

每个模型包含唯一 `name`、`head`、`window_size` 和可选 `projection_dim`。`head=context` 支持窗口 1/3/5，`cosine` 和 `linear` 只支持窗口 1。同维度 context 组控制名义参数量；cosine/linear 是另一组简单读出基线，不宣称与 context 头参数完全匹配。

`run` 至少需要两个模型来完成 family 封存和比较；只训练一个模型时使用独立 `train` 命令。模型名称使用字母、数字、下划线或连字符，不能仅以大小写区分两个候选。未知字段、重复 JSON 键和不合法类型会被拒绝，不会静默忽略。

训练参数包括 `epochs`、`patience`、`batch_reads`、`learning_rate`、`weight_decay`、`class_balance`、`seed`、`device`、`threads`。推理参数包括 `device`、`batch_events`、`threads`。评分参数包括 `bootstrap_reps`、`seed`、`min_reads`。完整可运行例子见 [smoke.json](../examples/smoke.json) 和 [full_experiment.json](../examples/full_experiment.json)。

历史 dense 的两个示例都显式设置 `allow_unverified_register=true`；它们生成 schema 1 和 `register.status="unverified"`。新 [aligned 示例](../examples/aligned.experiment.example.json) 使用 schema 2 的 `phase_calibrated`，导入时核对上游校准，无需历史未校准开关。两条路径都不能通过手工改成 `verified` 宣称对齐已经独立验证。

`--dry-run` 先检查配置并显示执行计划。运行失败后不自动覆盖或续训；保存故障目录，用新的 `output_dir` 重试，或通过各阶段 CLI 明确复用已完成且经审计的产物。

## 2. 固定数据与监督坐标

正式模型比较前，应明确 k-mer 起点、中心位置和 hidden 帧坐标，使用独立校准数据冻结规则。历史 dense 入口记录 `start_offset_bases` 与 `center_offset_bases=start+2`，不重新求解物理偏移。aligned 入口直接使用校准后的 spans，标签统一到参考方向，`additional_offset_bases=0`；默认排除参与校准的物理 reads。

保持物理 read 独立的 train/validation/prediction/reference。上游先拆分物理 read，再构造事件窗口；本工具会拒绝重复 read 和跨 split 泄漏。同一 read 中重叠窗口不是独立重复样本。

准备后先执行 `audit --verify-files`，检查各 split read 数、保留事件数和共同中心覆盖。smoke 截取按 read ID 排序，只检验接口，不用于正式性能结论。

## 3. 候选模型与训练

推荐候选组：

| 模型 | 主要问题 |
|---|---|
| 单事件 cosine | 冻结事件 hidden 的简单归一化读出能力 |
| 单事件 linear | 模长与类别 bias 是否有用 |
| context 1 | 投影/非线性读出本身的收益 |
| context 3 | 最近邻事件是否提供额外信息 |
| context 5 | 更大事件窗口是否进一步改善 |

默认损失先对每个 read 的事件求均值，再跨 read 求均值；一个事件只计一次，不再使用逐帧 inverse-dwell 加权。默认类别不重加权；`inverse_sqrt` 权重只由 train 支持数计算，应作为单独消融。验证选模使用未加类别权重的 read-macro CE。

全部模型只用 train/validation，不读取 prediction/reference 标签作初始化、早停或类别重加权。先训练并预测全部候选，再封存 family，最后评分。family lock 约束产物一致性，但不能证明操作者在封存前从未查看过真值。

同一 context 维度、中心集合、优化预算和随机种子下比较窗口。计划中的多种子验证需分别运行独立输出目录；当前 `compare` 是候选描述性比较，不会自动执行多种子统计或显著性检验。若看测试结果后继续改模型，需要新的独立确认集。

## 4. 分类、建表与置信区间

分类同时报告事件 Top1/Top5、read-macro Top1/Top5、出现类别 macro-recall、全 1024 类 macro-recall、类别支持与零召回。缺失类别的逐类 recall 为 `null`；全类别宏平均将缺失类别记作 0，必须与只对出现类别平均的结果区分。

read-bootstrap 对物理 read 重采样，而不是把重叠窗口当独立样本。少于两个 read 或 `bootstrap_reps=0` 时区间为 `null`。不同模型点估计和各自区间不是成对差异检验。

预测电流表先按 read/预测类别对中心事件电流取中位数，再跨 read 聚合；参考表来自独立 reference reads 的中心事件真值。只用中心电流，邻居电流不混入。历史输入和显式 `stone_unclamped` 使用 Stone 表名；aligned 的 `input_signal` 使用通用 current 表，并记录单位。

查看 hard coverage、共同类别数、Pearson、Spearman、MAE、RMSE，以及两表满足 `min_reads` 的可靠支持指标。`compare` 额外在所有候选和参考表的共同类别上比较表误差，避免不同覆盖集合造成假象；共同交集为空时不硬算指标。

旧逐帧约 10% Top1 不是本工具的事件级指标。没有统一到相同事件集合和聚合口径之前，不应声称新方案提高了多少个百分点。

## 5. 必要的后续验证

当前工具没有自动完成以下实验，不能把它们写成已获得证据：

- current-only 基线与电流相近类别的混淆分析：区分“识别序列”与“按电流水平分组”。
- 独立 run、基因组区域或上下文留出：read 不重叠不等于序列/上下文独立。
- 事件边界扰动和注册敏感性：区分真实上下文收益与边界误差补偿。
- 计算效率：相同设备下记录候选参数量、训练/推理耗时、峰值资源和事件吞吐；不能只看准确率。
- 纯信号场景：需要另外解决未知事件边界，不能把已知 move 边界实验直接称为端到端 basecalling。

是否采用五事件作为主模型，应由类别召回、跨上下文稳定性、表误差和资源代价共同决定，不由最高相关系数单独决定。
