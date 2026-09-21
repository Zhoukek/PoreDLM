# 输入与产物格式

PoreKmer 0.3.0 支持已有 dense hidden 语料、NanoSignalAlign JSONL + 逐帧 hidden，以及已准备好的事件语料。它不直接读取 FAST5、POD5、BAM 或运行基座模型。新 `prepare-aligned` 入口见 [Prep/Align 接入指南](ALIGNED_INPUT.zh-CN.md)；以下 dense 格式保持兼容。

## 1. dense hidden 输入

`prepare --source-manifest PATH` 接收 JSON manifest。源 manifest 必须 `status="complete"`，包含 `geometry` 和非空 `samples`。

`geometry` 中需要：

| 字段 | 含义 |
|---|---|
| `sequence_offset_bases` | 已有标签消费代码使用的 k-mer 起点偏移，不代表重新校准 |
| `model_stride_samples` | 每个 nominal hidden 格的信号步长，当前 V003 为 5 |
| `frame_lag` | 当前只支持 0；缺省按 0 处理 |
| `central_supervised_samples` | chunk 内监督区间 `[start, end)`，当前典型为 `[1500, 4500]` |

每个 sample 至少包含 `read_id`、`split`、安全相对路径 `path` 和文件 SHA-256 `npz_sha256`。可提供 `physical_read_id`、`record_id`/`query_name` 及来源元数据。没有 `physical_read_id` 时使用 `read_id` 作为物理 read 身份；调用者必须保证该身份真实，而非不同 chunk 的别名。

支持 `train`、`validation`、`prediction`、`reference` 和 `test` 划分；完整默认流水线使用前四种。一个物理 read 仅允许一个 chunk，不能在任意 split 重复。调试限流前也会检查完整 manifest 的身份冲突。

每个 dense NPZ 的必需数组如下。`T` 为 hidden 帧数、`D` 为 hidden 维度、`S` 为信号长度；现有缓存通常 `T=1200, D=768, S=6000`。

| 数组 | 形状 | 用途 |
|---|---|---|
| `hidden` | `[T, D]` | 冻结基座上下文表示 |
| `labels` | `[T]` | 5-mer 整数标签，完整有效事件为 `0..1023` |
| `weights` | `[T]` | 原 inverse-dwell 权重，校验完整事件 |
| `purity_mask`、`owned_mask` | 各 `[T]` | 标注格纯度与中央监督区域归属 |
| `state_ids` | `[T]` | 来源事件状态身份；不是 k-mer 类别 |
| `query_positions`、`reference_positions` | 各 `[T]` | 同事件一致性和真实邻接关系 |
| `sample_starts`、`sample_ends` | 各 `[T]` | nominal 信号格半开区间 |
| `signal_stone_unclamped` | `[S]` | 未 clamp 的 Stone 信号，仅用于中心事件建表 |

数组必须满足实现的长度、类型、有限数值和坐标一致性校验。PoreKmer 不重复进行 Stone 标准化，也不平移旧标签；更改注册定义需在上游重建标签、掩码和对应来源记录。

## 2. 事件生成规则

按状态身份、query/reference 位置和连续帧划分事件，不能按相同 k-mer 标签合并。同聚物中两个连续同标签状态仍然是两个不同事件。

仅保留完整、纯净、位于中央监督区域且 inverse-dwell 权重和约为 1 的事件。事件 hidden 是该事件有效帧的原始 hidden 算术平均。这里的“纯净”只针对标注采样格，不表示 V003 感受野不含邻居。

共同中心要求前后各两个真实相邻事件：query 位置逐次增加 1，事件信号区间首尾相接。缺失状态、部分事件和对齐缺口切断窗口；不填充，不跨缺口拼接。

1/3/5 事件模型使用同一批共同中心。单事件模型也不使用额外的、没有四个邻居的事件，因此这是固定中心集合的公平比较，不是单事件最大覆盖实验。

## 3. 准备后的事件语料

历史 dense 生成的事件 manifest 保持 `schema_version=1`、`kind="kmer_event_context"`。新 aligned 入口使用 schema 2，增加上游校准和电流类型契约，仍使用下面的六数组 feature/truth 结构。软件版本与数据 schema 版本不同，不应手工同步修改。

每个 read 一个 feature NPZ，且只能包含以下六个数组：

```text
features       [E, D]  完整事件的平均 hidden
centers        [C]     共同中心在 E 中的索引
center_ids     [C]     中心事件在 chunk 内的起始采样坐标
currents       [C]     中心事件电流中位数（schema 1 为 unclamped Stone，schema 2 显式声明类型）
event_starts   [E]     事件起点
event_ends     [E]     事件终点，不包含
```

与之分离的 truth NPZ 只含 `labels[C]`。模型窗口由 `make_windows()` 生成 `[C,5,D]`，对应 `e-2,e-1,e,e+1,e+2`。预测阶段只需 feature sidecar，不打开 truth 文件；但事件划分与筛选本身已经依赖 move/alignment，仍存在明确的条件实验边界。

manifest 记录安全相对路径、内容 SHA-256、来源 manifest 哈希、注册定义、样本计数和过滤说明。路径不可逃出语料根目录；移动语料时应保持内部目录结构，不要编辑 manifest 来绕过哈希检查。

```bash
porekmer audit --manifest /path/to/corpus/manifest.json --verify-files
```

审计通过只表示数据契约与哈希一致，不证明序列—信号配准正确。

## 4. 输出解释

- `checkpoint.pt`：验证集选出的最佳分类头，包含重建模型所需配置。只使用可信来源文件。
- `class_vectors.npy`：最佳模型的类别方向，或 linear 权重；linear bias 保留在 checkpoint。它不是原始信号表，也不保证是经验簇中心。
- `predictions.json`：预测记录、模型/输入哈希与中心事件指纹。
- `scores.json`：分类、read-bootstrap 和表指标，附注册和信息边界。
- `predicted_stone_5mer.tsv`：预测类别对应的中心电流聚合；无支持类别缺失。
- `reference_stone_5mer.tsv`：独立 reference reads、按真值中心类别聚合的参考表。
- schema 2 的 `input_signal` 使用 `predicted_current_5mer.tsv`、`reference_current_5mer.tsv` 和 `current_median`，不会将通用电流标为 Stone。
- `comparison.json`：同一中心集合上的候选比较，包括共同类别表指标。

分类器输出的类别与电流表是两个不同层级的产物。邻居提供分类上下文，但建表只使用中心电流；不能将五个事件的电流混成一个“更长的同类信号”。
