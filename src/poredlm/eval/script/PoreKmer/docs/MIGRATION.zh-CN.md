# 从 motor.kmer_context 迁移到 PoreKmer

PoreKmer 0.2.0 将原项目内模块整理为独立 `src` 布局包。分发名为 `PoreKmer`，Python import 和命令名均为 `porekmer`。这不是公共索引发布或名称所有权声明。

## 安装与入口替换

```bash
python -m pip install ./script/PoreKmer
python -m porekmer --help
porekmer --version
```

| 原来 | 现在 |
|---|---|
| `python -m motor.kmer_context prepare ...` | `porekmer prepare ...` |
| `python -m motor.kmer_context train ...` | `porekmer train ...` |
| `from motor.kmer_context.models import EventContextClassifier` | `from porekmer.models import EventContextClassifier` |
| 手动逐阶段执行 | 保留阶段命令，新增 `porekmer run --config ...` |

原有 `prepare/audit/train/predict/freeze-family/score/compare` 参数保持兼容。新增 `--version`、`doctor` 和配置驱动的 `run`。仓库内旧 `motor.kmer_context` 入口作为兼容转发层保留；新的外部项目应直接依赖安装后的 `porekmer`，不再导入 `motor`。

## 已有模型与数据

软件版本更新不等于实验数据更新。已有事件 manifest 和 checkpoint 的版本 1 契约保持不变，结构和哈希校验仍然执行；旧 dense hidden 仍可交给 `prepare`。不要为了迁移手工编辑 manifest、改写 NPZ 或重命名 checkpoint 内部参数。

兼容表示可在相同数据契约下读取，并不表示修正了历史标签的物理偏移。`register.status` 仍为 `unverified`，探索运行仍需明确允许。之前的实验结果不能因工具更名而改写成已校准结果。

可直接复用已准备的事件语料，在运行配置中用 `manifest` 替代 `source_manifest`；两者不能同时提供。新的运行目录必须不存在，旧结果不会被覆盖。

0.3.0 新增 `prepare-aligned` 和第三种配置输入 `aligned_manifest`，见 [接入指南](ALIGNED_INPUT.zh-CN.md)。该入口写入 schema 2、保留上游相位校准和电流类型；旧 schema 1 文件无需迁移。三种配置输入仍互斥。

## 路径变化

独立阶段命令中的相对 CLI 路径按通常方式相对于当前工作目录。`run` 配置里的路径则相对于配置文件所在目录：移动配置文件后，应同步检查其相对路径。

安装后从其他目录运行，不再需要把本项目根目录加入 `PYTHONPATH`。但外部数据与权重不会自动复制，源数据路径仍由用户提供。

## 迁移后检查

```bash
porekmer doctor
porekmer run --config script/PoreKmer/examples/smoke.json --dry-run
python -m unittest discover -s script/PoreKmer/tests -v
```

真实数据试跑应使用新的输出路径。命令成功只证明相应接口或契约通过，不证明全量训练、GPU、跨平台或生物学结论已经验证。
