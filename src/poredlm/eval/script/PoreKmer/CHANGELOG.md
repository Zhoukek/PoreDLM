# Changelog

## 0.3.0

- 新增 `prepare-aligned` 与 `run` 的 `aligned_manifest` 输入，将 NanoSignalPrep + NanoSignalAlign 的记录和逐帧 hidden 转为事件语料。
- 按校准区间聚合 hidden，支持引用信号、特殊 token 切片、正反方向、参考方向 5-mer 标签与断点筛选；不重复施加相位偏移。
- 新增事件 schema 2，记录上游校准及输入哈希，默认排除校准物理 reads；保留 schema 1 与原 dense 缓存接口。
- 区分通用电流与显式声明的未限幅 Stone 电流，通用表使用 `current_median`，各阶段保留电流类型。
- 提供 aligned 接入说明、配置示例和契约及流程测试。

## 0.2.0

- 命名为 PoreKmer，独立 `src/porekmer` 布局、本地可安装包、`porekmer` 控制台入口与 `python -m porekmer`。
- 保留单阶段数据准备、审计、训练、预测封存、评分与比较接口。
- 新增版本查询、环境诊断、JSON 配置驱动的完整候选实验及 dry-run。
- 提供中文安装、数据契约、实验和迁移文档，以及小规模和全量探索配置。
- 保留 schema/checkpoint v1 语义和旧项目入口兼容；不因更名修改标签、旧语料或实验结果。
- 明确未校准注册、已知边界实验和非端到端 basecaller 的限制。

## 0.1.0 — 项目内模块

- 原入口 `motor.kmer_context`：完整事件聚合、共同中心集合、单/三/五事件读出、事件级评估和中心 Stone 建表。

版本条目记录软件变化，不表示已发布到公共包索引，也不代替测试或模型效果报告。
