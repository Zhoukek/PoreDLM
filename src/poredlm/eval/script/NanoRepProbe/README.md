# NanoRepProbe

冻结模型表征的跨域识别评测工具。NanoRepDist 描述空间距离；NanoRepProbe 检验序列信息能否被简单分类器读出，以及同一识别规则能否跨数据域使用。不训练编码器，也不把 PCA 重合当作成功指标。

## 安装与入口

```bash
python -m pip install -e script/NanoRepProbe
nanorepprobe --help
nanorepprobe audit --config script/NanoRepProbe/configs/V003.chunk_center.json --output /tmp/probe.audit.json
nanorepprobe run --config script/NanoRepProbe/configs/V003.chunk_center.json --output /path/to/new_run --threads 4
nanorepprobe validate /path/to/new_run
```

不安装也可以：`PYTHONPATH=script/NanoRepProbe/src python -m nanorepprobe ...`。仅需 CPU，无 PyTorch/CUDA 或 pandas 依赖。配置路径相对配置文件所在目录解析。

中断后用相同命令加 `--resume`。恢复会校验配置、工具代码、所有输入 SHA256 和已完成任务产物；输入或代码变化必须建立新 run。默认拒绝覆盖非空输出。`report RUN` 只重绘报告并刷新输出快照，不重训分类器。

## 输入契约

JSON 配置包含模型名称、来源 run、读出策略、表征阶段、各域 manifest 与 NPY。[配置示例](configs/)保留 V003 的三种读出策略和 V006 的 `chunk_center`，均引用现有冻结表征。新增模型只需导出相同样本协议下的 `[N,D]` 浮点 NPY 并写新配置；三策略的配置方法见[三种读出策略](THREE_STRATEGIES.md)。

```json
{
  "model_id": "my_model",
  "source_run": "../frozen_run",
  "scheme": "chunk_center",
  "stages": ["vqe_cnn", "bert", "dlm_ode"],
  "domains": [
    {"name": "domain-a", "manifest": "../a/manifest.csv", "representations": {
      "vqe_cnn": "../a/cnn.npy", "bert": "../a/bert.npy", "dlm_ode": "../a/dlm.npy"}},
    {"name": "domain-b", "manifest": "../b/manifest.csv", "representations": {
      "vqe_cnn": "../b/cnn.npy", "bert": "../b/bert.npy", "dlm_ode": "../b/dlm.npy"}}
  ],
  "tasks": ["kmer", "center_base", "same_center_g"],
  "same_center_kmers": ["AAGAA", "CCGCC", "TTGTT"],
  "regularization": [0.0001, 0.01, 1, 100],
  "bootstrap_repeats": 200,
  "seed": 20260917
}
```

manifest 必须包含 `sample_index,sample_id,kmer,physical_read_id,probe_split`；每行通过 `sample_index` 访问 NPY 的原始行，允许 manifest 重排或子集。`kmer` 为五位 A/C/G/T；`probe_split` 为 train/validation/test。若有 `corpus` 列，必须与域名称一致。检查样本唯一、数组有限、维度一致，以及全局物理 read 不跨划分。`split=evaluation` 不能用于训练/验证/测试。

## 评测协议

- `kmer`：独立完整5-mer分类器；当前语料训练覆盖1024类。
- `center_base`：独立的中心A/C/G/T分类器。
- `same_center_g`：预先固定 AAGAA/CCGCC/TTGTT 三类分类器；此任务仅检验这三类上下文，不能外推为所有同中心G序列。
- 每域单独训练后测试所有域，得到训练域×测试域矩阵；三个及以上域另做“所有其他域训练、留一域测试”。
- 比较各表征阶段，连续CNN作为较早阶段基线；给出均匀随机猜测的理论期望。

使用线性 ridge 分类器（平方损失+L2正则），并非逻辑回归或神经网络分类头。目标为 `sum(w_i * ||onehot(y_i)-b-z_i W||²) + lambda*||W||²`。源域等权、每源域内各训练类别等权，权重总和1；均值/标准差仅用相同源训练权重计算，截距不正则化。各候选lambda复用相同加权协方差特征分解，通过各源域 validation 的 macro recall 等权平均选择；同分选更大lambda。一旦选定，同一分类器原样用于所有目标test。保存的分数是线性判别得分，不是概率。

默认候选 `[1e-4,1e-2,1,100]` 预先固定。测试集不参与标准化、拟合、调参或重训；线性读出得分不等于最优下游能力上限。

`macro_recall/macro_f1` 在该目标域测试集有真实支持的类别上计算；`common_macro_*` 在所有配置域共同具有测试支持的固定类别上计算，供主图比较。预测词表始终保留完整训练类别；F1保留全目标测试集产生的假阳性。缺失类别报告为support=0，不虚构零召回后混入宏平均。逐类表的零support行不代表测得了召回率。

95%区间采用 **按物理read分组的贝叶斯bootstrap**：每read一次Exp(1)权重，它的所有窗口共享该权重，200次重加权的2.5/97.5百分位。其归一化与Dirichlet(1,...,1)等价，可保留稀有类别支持。该区间反映固定分类器下测试read权重的不确定性，不含probe重训或预训练不确定性，也不宣称频率学覆盖率。同任务/目标域的各模型阶段使用相同种子；当前不输出配对提升显著性结论。

## 输出

```text
run/
  README.md                    # 结果与解释边界
  config.resolved.json          # 解析后的完整配置
  audit.json                   # 每域划分、覆盖、read隔离与输入SHA
  metrics.csv                  # 每训练域×测试域×阶段×任务
  validation.csv               # 所有lambda的源域验证结果
  per_class.csv                # 每类support/召回率/F1
  jobs/<stage>__<task>__<sources>/
    probe.npz                  # 标准化参数、线性系数、截距、类别表
    fit.json                   # 训练数量、lambda选择与耗时
    <target>.predictions.npz    # 原sample_index/id/read、真实/预测类别及top5
    <target>.confusion.npz      # 完整混淆矩阵
    complete.json              # 原子任务完成记录与SHA
  reports/PLOTS.md              # 450dpi PNG与SVG图入口
  figures/                     # 图及源数据清单
  plan/experiment-protocol.md
  tables/table-schema.md
  provenance/nanorepprobe/      # 此次工具源码快照
  environment.json
  status.json
  SHA256SUMS
```

`nanorepprobe validate` 校验输出快照，从逐样本预测重算主要指标并比对混淆矩阵。区间与拟合参数的独立审计另见实际run的验收报告。

## 当前能力边界

本工具从冻结表征训练分类器，支持两个及以上数据域和配置中的任务子集。表征提取由上游工具完成；`chunk_center`、`chunk_5mer`、`short_5mer` 分别配置和运行，各策略的结果单独报告。

read隔离不等于参考位点隔离；预训练训练集重叠和phase校准置信度应随输入记录。数据域可能混有平台、文库、来源差异，不能直接解释为纯马达效应。同时改变模型和预处理时，不能将差异单独归因于模型。RNA、已知/未知修饰信息、未见位点泛化和跨马达少样本适配需要独立对照。

## 测试

```bash
python -m pip install -e 'script/NanoRepProbe[dev]'
python -m pytest script/NanoRepProbe/tests -q
```

测试覆盖读出行索引、物理read泄漏、输入/输出篡改、线性解与直接加权求解一致、缺失类宏指标、同read重复窗口的不确定性处理，以及端到端训练/恢复/报告。所有合成测试在临时目录生成，不混入真实结果。
