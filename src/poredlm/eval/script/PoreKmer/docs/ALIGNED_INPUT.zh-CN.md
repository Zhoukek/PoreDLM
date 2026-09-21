# NanoSignalPrep + NanoSignalAlign 接入

`prepare-aligned` 读取已经校准的 JSONL 和模型逐帧 hidden，构造 PoreKmer 的事件语料。基座模型、checkpoint 和 hidden 层由调用方选择；适配器不运行模型、不重复标准化信号，也不重新估计相位。

## 1. 准备输入清单

复制 [输入模板](../examples/aligned.source.example.json)，填写真实路径、身份和 SHA256。模板中的四条记录分别演示完整实验的 train、validation、prediction、reference；模板没有附带数据，零值哈希需要替换。

```json
{
  "schema_version": 1,
  "kind": "nanosignalalign_hidden",
  "status": "complete",
  "geometry": {"model_stride_samples": 5, "frame_origin_samples": 0},
  "current": {"normalization": "input_signal", "units": "normalized_signal"},
  "samples": [{
    "read_id": "read-a:0:0",
    "physical_read_id": "read-a",
    "split": "train",
    "aligned_jsonl": "inputs/aligned.jsonl",
    "record_index": 0,
    "hidden_path": "hidden/read-a.npy",
    "hidden_sha256": "替换为该 NPY 文件的 SHA256"
  }]
}
```

输入清单中的文件路径相对清单所在目录解析，也可用绝对路径。JSONL 内部的 `signal_ref.path`、`calibration_path` 则相对 JSONL 所在目录解析。输出事件语料的内部路径保持相对路径。

| 字段 | 要求 |
|---|---|
| `read_id` / `physical_read_id` | 必须与选中 JSONL 记录一致；不从 chunk 名称猜测物理身份 |
| `record_index` | JSONL 中非空记录的零基序号；支持 `.jsonl.gz` |
| `split` | `train`、`validation`、`prediction`、`reference` 或 `test` |
| `hidden_path` / `hidden_sha256` | 对应整条有效信号的二维实数 NPY 及文件校验和 |
| `hidden_slice` | 可选 `[start,end)`，沿 NPY 第 0 维截取内容帧，排除 BOS/EOS 或 padding |
| `current_ref` | 可选外部电流 NPY 引用：`path`、`start`、`end`，二维 NPY 还需 `row`；长度须等于有效信号长度 |

一个物理 read 只能出现一次，包括同一 split 内；应在上游先选一个完整 chunk、再按物理 read 分组。`max_reads_per_split` 是按身份排序的调试限流，限流前仍检查整个清单的身份冲突。

aligned 记录必须含 `seq`、`ref`、`align.seq_aligned/ref_aligned`、`base_sample_span_ref_calibrated` 和 `signal_alignment`。注释需要 `stage=phase_calibrated`、`query_orientation`、整数 `offset_bases`、`calibration_path`、`calibration_sha256` 和校准角色。校准文件的方向、偏移与记录必须一致，文件哈希必须匹配。信号支持 inline `signal` 或二维 NPY 的 `signal_ref`。

## 2. hidden 时间轴和事件聚合

先应用 `hidden_slice`；此后第 `j` 帧对应信号锚点：

```text
frame_origin_samples + j * model_stride_samples
```

`model_stride_samples` 必须为正整数，`frame_origin_samples` 在 `[0,stride)`，默认 0。切片后的帧数必须等于有效信号内锚点数；长度 6000、stride 5、origin 0 时为 1200 帧。若 BERT 输出 `[1202,D]` 且含首尾特殊 token，可设置 `hidden_slice: [1,1201]`。该规则约束名义帧坐标，不宣称模型感受野只包含一个事件。

每个事件直接使用已校准的参考碱基信号区间 `[start,end)`，平均锚点落在区间内的所有 hidden，FP64 累积后保存 FP32。没有锚点的事件被过滤；不通过补零或相邻帧插值制造事件。原有相位已经体现在 span 中，适配器不会再添加 `offset_bases`。

事件按照信号时间顺序排列。仅使用与参考精确匹配的 A/C/G/T 事件；五事件窗口要求信号首尾相接、参考和 query 位置按标注方向逐碱基相邻。错配、插入、缺失、无效映射和无 hidden 的事件会切断上下文；相邻同聚物事件仍保留为不同事件。

中心标签固定取参考序列 `ref[center_ref-2:center_ref+3]`，按 A/C/G/T 字典序编码为 `0..1023`。对于 `reverse`，特征窗口仍按信号时间排列，标签仍按参考字符串方向；不将信号翻转、不自动做反向互补。因此 RNA 或负向记录的信号方向 5-mer 可能与输出标签顺序不同，这一约定会保存在来源记录中。

## 3. 电流类型

`current.normalization` 支持：

- `input_signal`：默认使用 aligned 信号，也可以显式提供 `current_ref`；保留已有数值，不重新标准化。输出 `predicted_current_5mer.tsv`、`reference_current_5mer.tsv`，数值列为 `current_median`。
- `stone_unclamped`：每条记录必须显式提供 `current_ref`，调用方确认是未限幅的 Stone 信号。输出沿用 `predicted_stone_5mer.tsv`、`reference_stone_5mer.tsv` 和 `stone_median`。

`current.units` 为必填说明，例如 `normalized_signal`。工具记录声明和来源，不能仅从数值证明标准化历史。`00.corpus/stone` 是历史目录名，不能单凭该名字声明 `stone_unclamped`；Apple 模型输入也不能自动视为 Stone 电流。需要 Stone 表时，应另行提供长度和时间轴一致的电流引用。

中心电流对完整事件区间的全部采样点取中位数；建表仍先按 read/类别取中位数，再跨 read 汇总。特征可以来自 Apple 模型输入，建表电流可以来自独立 `current_ref`，两者的采样坐标必须相同。候选比较检查电流类型和单位一致。

## 4. 校准来源和数据划分

新事件 manifest 为 `schema_version=2`、`kind=kmer_event_context`，注册状态是 `phase_calibrated`，并保存：

- `label_coordinate_system=reference`、`kmer_pattern=XXNXX`；
- `additional_offset_bases=0`、`independently_validated=false`；
- 校准文件路径、哈希和置信度，以及逐记录方向、偏移和源数据信息。

默认排除标为校准角色的物理 reads，也检查校准文件 `stats.calibration_read_ids`。`--allow-calibration-reads` 可显式保留这些 reads；这一选择会进入 provenance。低或未知置信度会保留，`phase_calibrated` 不等于独立验证准确，也不会被改成 `verified`。

历史 dense 入口及 schema 1 不变，仍要求其原有 `allow_unverified_register`。新 schema 2 可以直接训练；导入时校验校准来源，不要求使用历史标签的未校准开关。

## 5. 执行

在当前项目根目录，可安装本地包，或使用源码入口：

```bash
export PYTHONPATH=script/PoreKmer/src
python -m porekmer prepare-aligned --source-manifest /path/to/aligned.source.json \
  --output-dir /path/to/new_events
python -m porekmer audit --manifest /path/to/new_events/manifest.json --verify-files
```

完成的事件目录可作为已有实验配置的 `manifest`。也可从 [实验模板](../examples/aligned.experiment.example.json) 一次执行准备、训练全部候选、冻结全部预测、评分和比较：

```bash
python -m porekmer run --config /path/to/aligned.experiment.json --dry-run
python -m porekmer run --config /path/to/aligned.experiment.json
```

配置使用 `aligned_manifest`；与历史 `source_manifest`、事件 `manifest` 三选一。配置中的相对路径以配置文件目录为准，输出目录必须不存在。dry-run 只检查配置和元数据，实际导入时才读取信号、hidden 和校准文件。

现有 `01.capability/01.model_eval` 的池化向量服务于距离评估；接入本工具时，需要模型导出完整连续 hidden，并明确层 stride、特殊 token 和预处理。此适配器不重新生成基座表征或覆盖现有评估语料。
