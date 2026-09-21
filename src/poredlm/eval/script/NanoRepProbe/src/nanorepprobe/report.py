"""Auditable figures and reports for frozen cross-domain linear probes.

The reporter never fits a probe or selects a regularization parameter.  It only
renders metrics.csv after the runner has completed inference and evaluation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_REQUIRED = {
    'model_id', 'scheme', 'stage', 'task', 'train_domains', 'test_domain',
    'experiment', 'n_test', 'n_classes_evaluated', 'accuracy', 'macro_recall',
    'macro_f1', 'top5_accuracy', 'common_macro_recall', 'common_macro_f1',
    'macro_recall_ci_low', 'macro_recall_ci_high', 'selected_lambda',
}
_SCORES = (
    'accuracy', 'macro_recall', 'macro_f1', 'top5_accuracy',
    'common_macro_recall', 'common_macro_f1', 'macro_recall_ci_low',
    'macro_recall_ci_high', 'common_macro_recall_ci_low',
    'common_macro_recall_ci_high',
)
_COUNTS = ('n_test', 'n_classes_evaluated', 'n_classes_train', 'common_n_classes', 'n_test_reads')
_TASKS = ('kmer', 'center_base', 'same_center_g')
_TASK_LABELS = {'kmer': '5-mer identity', 'center_base': 'Center base',
                'same_center_g': 'Same-center-G: AAGAA / CCGCC / TTGTT'}
_DOMAIN_LABELS = {'cyclone-dna': 'Cyclone DNA', 'ont-r10-hg002': 'ONT R10 HG002',
                  'dna-amplicon': 'DNA amplicon'}
_STAGE_LABELS = {'vqe_cnn': 'VQE CNN (baseline)', 'vqe_code': 'VQE code',
                 'bert': 'BERT context', 'dlm_ode': 'DLM ODE'}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _slug(value: str) -> str:
    cleaned = re.sub(r'[^A-Za-z0-9_-]+', '_', value).strip('_')
    return (cleaned or 'unnamed')[:130]


def _text(value: Any) -> str:
    return str(value).replace('|', '\\|').replace('\n', ' ')


def _join_markdown(parts: list[str]) -> str:
    """Keep table rows contiguous while preserving spacing between sections."""
    document = '\n'.join(parts)
    return re.sub(r'(?m)(^\|[^\n]*\n)(?:[ \t]*\n)+(?=\|)', r'\1', document)


def _number(value: Any) -> str:
    return 'NA' if value is None else f'{float(value):.3f}'


def _range(rows: list[dict], column: str) -> str:
    values = sorted({int(row[column]) for row in rows if row.get(column) is not None})
    return '未记录' if not values else str(values[0]) if len(values) == 1 else f'{values[0]}–{values[-1]}'


def _load_metrics(path: Path) -> tuple[list[dict], list[str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        missing = sorted(_REQUIRED - set(fields))
        if missing:
            raise ValueError(f'metrics.csv is missing columns: {missing}')
        rows = list(reader)
    if not rows:
        raise ValueError('metrics.csv has no observations; an empty experiment cannot be reported as complete')
    seen = set()
    for row in rows:
        if row['task'] not in _TASKS or row['experiment'] not in ('single_domain', 'leave_one_domain_out'):
            raise ValueError('Unsupported task or experiment in metrics.csv')
        for name in ('model_id', 'scheme', 'stage', 'train_domains', 'test_domain'):
            if not row[name]:
                raise ValueError(f'Empty identity field: {name}')
        for name in (*_SCORES, 'selected_lambda'):
            if name not in row:
                continue
            value = row[name]
            if value in ('', 'NA', 'NaN', 'nan', 'None', 'null'):
                row[name] = None
                continue
            number = float(value)
            if not np.isfinite(number):
                raise ValueError(f'Nonfinite metric: {name}')
            if name in _SCORES and not 0 <= number <= 1:
                raise ValueError(f'{name} must be in [0, 1]')
            if name == 'selected_lambda' and number < 0:
                raise ValueError('selected_lambda must be nonnegative')
            row[name] = number
        for name in ('accuracy', 'macro_recall', 'macro_f1', 'selected_lambda'):
            if row.get(name) is None:
                raise ValueError(f'Missing required completed-result metric: {name}')
        for name in _COUNTS:
            if name not in row or row[name] == '':
                row[name] = None
                continue
            value = float(row[name])
            if not np.isfinite(value) or value < 0 or value != int(value):
                raise ValueError(f'Invalid count: {name}')
            row[name] = int(value)
        if not row['n_test'] or not row['n_classes_evaluated']:
            raise ValueError('Evaluation rows require nonzero test samples and evaluated classes')
        if row.get('common_n_classes') is not None and row['common_n_classes'] > row['n_classes_evaluated']:
            raise ValueError('Common test classes exceed evaluated test classes')
        for prefix in ('macro_recall', 'common_macro_recall'):
            low, high = row.get(prefix + '_ci_low'), row.get(prefix + '_ci_high')
            if (low is None) != (high is None) or (low is not None and low > high):
                raise ValueError(f'Invalid interval endpoints for {prefix}')
        train = row['train_domains'].split('+')
        if len(train) != len(set(train)) or any(not domain for domain in train):
            raise ValueError('Invalid training-domain identity')
        if row['experiment'] == 'single_domain' and len(train) != 1:
            raise ValueError('single_domain requires one training domain')
        if row['experiment'] == 'leave_one_domain_out' and (len(train) < 2 or row['test_domain'] in train):
            raise ValueError('leave_one_domain_out requires at least two source domains excluding the target')
        identity = tuple('+'.join(sorted(train)) if name == 'train_domains' else row[name] for name in ('model_id', 'scheme', 'stage', 'task', 'train_domains', 'test_domain', 'experiment'))
        if identity in seen:
            raise ValueError(f'Duplicate metric cell: {identity}')
        seen.add(identity)
    return rows, fields


def _orders(config: dict, rows: list[dict]) -> tuple[list[str], list[str]]:
    configured = config.get('domains', [])
    domains = [str(item['name'] if isinstance(item, dict) else item) for item in configured]
    observed_domains = set(row['test_domain'] for row in rows)
    observed_domains.update(domain for row in rows for domain in row['train_domains'].split('+'))
    if not domains:
        domains = [domain for domain in _DOMAIN_LABELS if domain in observed_domains]
        domains.extend(sorted(observed_domains - set(domains)))
    if len(domains) < 2 or len(set(domains)) != len(domains) or observed_domains - set(domains):
        raise ValueError('The report requires at least two distinct configured domains')
    stages = [str(stage) for stage in config.get('stages', [])]
    observed_stages = {row['stage'] for row in rows}
    if not stages:
        stages = [stage for stage in _STAGE_LABELS if stage in observed_stages]
        stages.extend(sorted(observed_stages - set(stages)))
    if not stages or len(set(stages)) != len(stages) or observed_stages - set(stages):
        raise ValueError('Stage order is inconsistent with metrics.csv')
    return domains, stages


def _task_order(config: dict, rows: list[dict]) -> list[str]:
    observed = {row['task'] for row in rows}
    tasks = list(config.get('tasks') or [task for task in _TASKS if task in observed] or _TASKS)
    if len(set(tasks)) != len(tasks) or any(task not in _TASKS for task in tasks) or observed - set(tasks):
        raise ValueError('Task selection is inconsistent with metrics.csv')
    return tasks


def _task_label(task: str, config: dict) -> str:
    if task == 'same_center_g':
        return 'Same-center-G: ' + ' / '.join(config.get('same_center_kmers', ['AAGAA', 'CCGCC', 'TTGTT']))
    return _TASK_LABELS[task]


def _validate_experiments(rows: list[dict], domains: list[str]) -> None:
    all_domains = set(domains)
    for row in rows:
        if row['experiment'] == 'leave_one_domain_out':
            if len(domains) < 3:
                raise ValueError('Two-domain evaluation does not include leave_one_domain_out rows')
            if set(row['train_domains'].split('+')) != all_domains - {row['test_domain']}:
                raise ValueError('leave_one_domain_out must train on every domain except its target')


def _primary_metric(rows: list[dict]) -> str:
    # Missing common-support cells remain NA.  Never mix support definitions
    # within a heatmap by silently substituting full-target recall cell by cell.
    return 'common_macro_recall' if any(row.get('common_macro_recall') is not None for row in rows) else 'macro_recall'


def _write_data(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _heatmaps(rows: list[dict], domains: list[str], stages: list[str], metric: str,
              prefix: Path, synthetic: bool) -> list[Path]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    cmap = LinearSegmentedColormap.from_list('probe_recall', ['#F4F6F8', '#76B7B2', '#2E86AB', '#17405C'])
    cmap.set_bad('#DEDEDE')
    style = {'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
             'axes.labelsize': 9, 'xtick.labelsize': 7.8, 'ytick.labelsize': 8,
             'svg.fonttype': 'none', 'savefig.dpi': 450, 'savefig.bbox': 'tight',
             'axes.spines.top': False, 'axes.spines.right': False}
    lookup = {(row['stage'], row['train_domains'], row['test_domain']): row
              for row in rows if row['experiment'] == 'single_domain'}
    model, scheme, task = rows[0]['model_id'], rows[0]['scheme'], rows[0]['task']
    n_domains = len(domains)
    with plt.rc_context(style):
        panel_inches = max(3.7, n_domains + .7)
        fig, axes = plt.subplots(1, len(stages), figsize=(panel_inches * len(stages) + .55, max(4.75, 1.05 * n_domains + 1.6)), squeeze=False)
        for axis, stage in zip(axes[0], stages):
            matrix = np.full((n_domains, n_domains), np.nan)
            for i, train in enumerate(domains):
                for j, test in enumerate(domains):
                    row = lookup.get((stage, train, test))
                    if row is not None and row.get(metric) is not None:
                        matrix[i, j] = row[metric]
            image = axis.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap=cmap, aspect='equal')
            for i, train in enumerate(domains):
                for j, test in enumerate(domains):
                    row = lookup.get((stage, train, test))
                    value = None if row is None else row.get(metric)
                    label = _number(value)
                    if row is not None:
                        classes = row.get('common_n_classes') if metric == 'common_macro_recall' else row['n_classes_evaluated']
                        label += f"\nn={row['n_test']}"
                        if classes is not None:
                            label += f"\nC={classes}"
                    axis.text(j, i, label, ha='center', va='center', fontsize=7.5,
                              color='white' if value is not None and value > .60 else '#202B33')
                axis.add_patch(Rectangle((i - .5, i - .5), 1, 1, fill=False, edgecolor='#59616C', linewidth=1.0))
            axis.set_xticks(range(n_domains), [_DOMAIN_LABELS.get(d, d).replace(' HG002', '\nHG002') for d in domains], rotation=25, ha='right')
            axis.set_yticks(range(n_domains), [_DOMAIN_LABELS.get(d, d) for d in domains])
            axis.set_title(_STAGE_LABELS.get(stage, stage))
            axis.set_xlabel('Test domain')
            if axis is axes[0, 0]:
                axis.set_ylabel('Train domain')
            axis.set_xticks(np.arange(-.5, n_domains, 1), minor=True)
            axis.set_yticks(np.arange(-.5, n_domains, 1), minor=True)
            axis.grid(which='minor', color='white', linewidth=1)
            axis.tick_params(which='minor', bottom=False, left=False)
        color_axis = fig.add_axes([.92, .30, .013, .43])
        label = 'Macro recall (common test classes)' if metric == 'common_macro_recall' else 'Macro recall (target-supported classes)'
        fig.colorbar(image, cax=color_axis, ticks=[0, .25, .5, .75, 1]).set_label(label, fontsize=8)
        title = f'{model} / {scheme}\n{rows[0].get("_task_label", _TASK_LABELS[task])} · frozen linear ridge probe'
        if synthetic:
            title = '[SYNTHETIC — REPLACE WITH REAL DATA]\n' + title
        fig.suptitle(title, fontsize=10.5, y=.985)
        fig.text(.48, .035, 'Diagonal: held-out reads from the training domain.  Off-diagonal: a different test domain.\n'
                 'n = test samples; C = evaluated classes for the displayed metric.  All panels use the same 0–1 color scale.',
                 fontsize=7.8, ha='center', linespacing=1.5)
        fig.subplots_adjust(left=.105, right=.90, top=.76 if synthetic else .79, bottom=.25, wspace=.55)
        outputs = []
        prefix.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ('.png', '.svg'):
            path = prefix.with_suffix(suffix)
            fig.savefig(path, dpi=450)
            outputs.append(path)
        plt.close(fig)
    return outputs


def _interval(row: dict, metric: str) -> str:
    prefix = 'common_macro_recall' if metric == 'common_macro_recall' else 'macro_recall'
    low, high = row.get(prefix + '_ci_low'), row.get(prefix + '_ci_high')
    return '未记录' if low is None or high is None else f'[{low:.3f}, {high:.3f}]'


def _protocol_notes(audit: dict) -> str:
    notes = audit.get('protocol_notes', {})
    if not isinstance(notes, dict):
        raise ValueError('audit.protocol_notes must be an object when supplied')
    result = ['### 审计记录的协议与解释范围\n']
    if notes:
        result.append('| 字段 | 审计记录 |\n|---|---|\n')
        for key, value in notes.items():
            rendered = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, bool)) else str(value)
            result.append(f'| {_text(key)} | {_text(rendered)} |\n')
    else:
        result.append('audit.json 未提供 protocol_notes；未据此推断相位校准、参考位点隔离或预训练重叠状态。\n')
    if notes.get('phase_confidence') is not None:
        result.append('相位校准状态按上述来源记录呈现；该置信度不是逐碱基标签正确率，标签不确定性可能影响 probe。\n')
    split = str(notes.get('split', ''))
    if 'physical_read' in split or 'physical read' in split:
        result.append('按 physical read 分组使测试 reads 与训练划分区分；这种划分本身不能证明参考位点也未出现在训练中。\n')
    if notes.get('reference_locus_holdout') is False:
        result.append('审计记录未实施参考位点隔离，不能将当前结果解释为未见参考位点的泛化。\n')
    elif notes.get('reference_locus_holdout') is True:
        result.append('审计记录 reference_locus_holdout=true；具体位点划分及证据以来源审计为准。\n')
    elif 'reference_locus_holdout' not in notes:
        result.append('参考位点隔离未提供记录。\n')
    if str(notes.get('pretraining_overlap', '')).lower() == 'unknown':
        result.append('审计记录预训练重叠未知，不能声称与编码器预训练数据完全无重叠。\n')
    elif 'pretraining_overlap' not in notes:
        result.append('预训练重叠未提供记录。\n')
    result.append('跨数据域比较可能同时涉及样本来源、实验流程与输入处理；除非另有控制实验，不能自动归因于单一马达因素。若跨运行同时改变模型与预处理，应按组合差异解读。\n')
    return _join_markdown(result)


def _markdown_report(rows, audit, config, domains, stages, groups, figure_records, synthetic, completeness):
    warning = '\n**[待真实实验替换] 本报告为 synthetic smoke 测试，所有数值均为合成数据。**\n' if synthetic else ''
    tasks = _task_order(config, rows)
    state = '指标组合完整' if completeness else '指标组合不完整；缺失项保留为 NA，不能视为完成测评'
    descriptions = {
        'kmer': '识别当前评估支持的 k-mer 类别；若包含 CNN 阶段，可将其作为较早表征阶段的基线。',
        'center_base': '独立预测中心碱基，检查是否主要读出了中心位点。',
        'same_center_g': '针对 ' + '、'.join(config.get('same_center_kmers', ['AAGAA', 'CCGCC', 'TTGTT']))
             + f" 独立训练 {len(config.get('same_center_kmers', ['AAGAA', 'CCGCC', 'TTGTT']))} 类 probe；这些类别中心相同，用于检查上下文分辨能力。样本较少时区间可能较宽。",
    }
    result = ['# NanoRepProbe：跨域线性探针\n', warning,
              f'{state}。已读取 {len(rows):,} 行指标，输入及运行审计见 [配置](config.resolved.json)、[audit.json](audit.json) 和 [完整指标](metrics.csv)。\n',
              '本工具的评估协议固定编码器表征，只拟合线性 ridge probe；正则化参数使用训练/验证数据选择，测试集不参与模型或超参数选择。它衡量冻结表征的线性可读出程度，不等同于训练完整 basecaller 的准确率；本次执行记录见下方审计字段。\n',
              '## 评估任务\n',
              '\n'.join(f'- `{task}`：{descriptions[task]}' for task in tasks) + '\n',
              '## 跨域迁移矩阵\n',
              f'本批次配置 {len(domains)} 个数据域。图中行是训练域、列是测试域；对角线也只评估 held-out reads。所有图统一使用 0–1 色标。优先显示共同测试类别上的 `common_macro_recall`；如果整组缺少该指标，改用目标域支持类别上的 `macro_recall` 并在色条注明。部分缺失的共同类别指标保留 NA，不混用两种口径。\n']
    for figure in figure_records:
        result.append(f"- {figure['model_id']} / {figure['scheme']} / {figure['task']}：[PNG]({figure['png']}) · [SVG]({figure['svg']}) · [绘图数据]({figure['data']})\n")
    result += ['\n## 类别覆盖与测试规模\n',
               '完整目标域宏指标包含该目标测试集中有样本的类别；共同类别宏指标限制在约定的共同支持集合，不能替代全部目标类别的覆盖报告。下表汇总各阶段和训练域条件的取值范围，精确逐行值见 metrics.csv；共同类别身份及划分支持见 audit.json。\n',
               '| 模型 / 方案 | 任务 | 测试域 | 测试样本 | 测试 reads | 目标类数 | 共同类数 | 训练类数 |\n|---|---|---|---:|---:|---:|---:|---:|\n']
    coverage = defaultdict(list)
    for row in rows:
        coverage[(row['model_id'], row['scheme'], row['task'], row['test_domain'])].append(row)
    for key, group in sorted(coverage.items()):
        model, scheme, task, domain = key
        result.append(f'| {_text(model)} / {_text(scheme)} | {task} | {_text(domain)} | ' + ' | '.join(_range(group, col) for col in ('n_test', 'n_test_reads', 'n_classes_evaluated', 'common_n_classes', 'n_classes_train')) + ' |\n')
    lodo = [row for row in rows if row['experiment'] == 'leave_one_domain_out']
    result.append('\n## 留一域评估\n')
    if len(domains) < 3:
        result.append('本批次只有两个域，按配置不运行 leave-one-domain-out；双向跨域及同域参考已包含在上方矩阵。\n')
    else:
        result += [f'每行用其余 {len(domains) - 1} 个域训练，在被留出的一个域测试；每任务、每阶段预期 {len(domains)} 行。以下并列保留全部目标类别宏召回及共同类别宏召回。\n',
                   '| 模型 / 方案 | 任务 | 阶段 | 训练域 → 测试域 | 全目标 macro recall | 共同 macro recall | 共同类别区间 | λ |\n|---|---|---|---|---:|---:|---|---:|\n']
        for row in sorted(lodo, key=lambda x: (x['model_id'], x['scheme'], tasks.index(x['task']), stages.index(x['stage']), domains.index(x['test_domain']))):
            result.append(f"| {_text(row['model_id'])} / {_text(row['scheme'])} | {row['task']} | {row['stage']} | {_text(row['train_domains'])} → {_text(row['test_domain'])} | {_number(row['macro_recall'])} | {_number(row['common_macro_recall'])} | {_interval(row, 'common_macro_recall')} | {row['selected_lambda']:.3g} |\n")
        if not lodo:
            result.append('| — | — | — | 留一域指标缺失 | — | — | — | — |\n')
    methods = sorted({str(row.get('interval_method')) for row in rows if row.get('interval_method')})
    result += ['\n## 区间与审计说明\n',
               '区间方法记录为 ' + (', '.join(f'`{method}`' for method in methods) if methods else '未提供，不能从列名 ci 自动推断方法') + '。\n']
    if 'read_cluster_bayesian_bootstrap' in methods:
        result.append('`read_cluster_bayesian_bootstrap` 对每个测试 physical read 分配独立 Exp(1) 权重，归一化等价于 Dirichlet 权重；同一 read 的样本共同加权。其区间描述固定 probe 对测试 read 权重的敏感性，保持已有类别支持；不是普通频率学置信区间，也不包含 probe 重训练、表征模型训练或预训练变化的不确定性。区间分位数与重采样次数以配置为准。\n')
    conflicts = audit.get('group_split_conflicts')
    if isinstance(conflicts, list):
        result.append(f'审计记录中的跨划分 physical-read 冲突数为 {len(conflicts)}；具体 read 分组、类别支持和输入指纹见 audit.json。\n')
    result.append(_protocol_notes(audit))
    result.append('\n[图表索引](reports/PLOTS.md) · [图表数据清单](figures/data-manifest.md)\n')
    return _join_markdown(result)


def render_report(output_dir: Path) -> dict:
    """Render completed CSV metrics, audit metadata and resolved configuration.

    ``config.resolved.json`` must record the domain and stage order.  Optional
    ``data_kind='synthetic'`` marks smoke data in filenames and visible captions.
    Figures display common_macro_recall when supplied.  CI columns are never
    relabeled as frequentist confidence intervals without a recorded method.
    """
    output_dir = Path(output_dir).resolve()
    metrics_path, audit_path, config_path = [output_dir / name for name in ('metrics.csv', 'audit.json', 'config.resolved.json')]
    inputs = [{'path': path.name, 'sha256': _sha(path), 'bytes': path.stat().st_size} for path in (metrics_path, audit_path, config_path)]
    rows, fields = _load_metrics(metrics_path)
    audit, config = json.loads(audit_path.read_text()), json.loads(config_path.read_text())
    if not isinstance(audit, dict) or not isinstance(config, dict):
        raise ValueError('audit.json and config.resolved.json must contain objects')
    domains, stages = _orders(config, rows)
    tasks = _task_order(config, rows)
    _validate_experiments(rows, domains)
    data_kind = str(config.get('data_kind', audit.get('data_kind', 'real'))).lower()
    synthetic = data_kind in ('synthetic', 'mock', 'planning') or config.get('synthetic') is True
    if data_kind not in ('real', 'synthetic', 'mock', 'planning'):
        raise ValueError(f'Unknown data_kind: {data_kind}')
    groups = defaultdict(list)
    for row in rows:
        groups[(row['model_id'], row['scheme'], row['task'])].append(row)
    figures_dir, reports_dir = output_dir / 'figures', output_dir / 'reports'
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    records, coverage = [], []
    for (model, scheme, task), group in sorted(groups.items()):
        stem = ('synthetic_' if synthetic else '') + _slug(model) + '__' + _slug(scheme) + '__' + task + '_transfer'
        data_path = figures_dir / 'data' / (stem + '.csv')
        _write_data(data_path, group, fields)
        metric = _primary_metric(group)
        complete = True
        for stage in stages:
            single = [r for r in group if r['stage'] == stage and r['experiment'] == 'single_domain']
            lodo = [r for r in group if r['stage'] == stage and r['experiment'] == 'leave_one_domain_out']
            complete &= len(single) == len(domains) ** 2 and len(lodo) == (len(domains) if len(domains) >= 3 else 0)
            coverage.append({'model_id': model, 'scheme': scheme, 'task': task, 'stage': stage,
                             'single_domain_cells': len(single), 'leave_one_domain_out_cells': len(lodo)})
        records.append({'model_id': model, 'scheme': scheme, 'task': task, 'metric': metric, 'complete': bool(complete),
                        'data': str(data_path.relative_to(output_dir)), 'data_sha256': _sha(data_path),
                        'png': str((figures_dir / stem).with_suffix('.png').relative_to(output_dir)),
                        'svg': str((figures_dir / stem).with_suffix('.svg').relative_to(output_dir))})
    all_models = {(row['model_id'], row['scheme']) for row in rows}
    complete = all(record['complete'] for record in records) and all((model, scheme, task) in groups for model, scheme in all_models for task in tasks)
    source_file = Path(__file__).resolve()
    manifest = ['# NanoRepProbe 图表数据清单\n',
                '**[待真实实验替换] synthetic smoke 数据，不能当作实验结果。**\n' if synthetic else '绘图数据来自本批次实际 metrics.csv；报告不拟合或重新选择 probe。\n',
                '所有绘图 CSV 与本清单先于图像落盘。热图统一取值范围 [0, 1]；共同类别及全部目标类别宏指标分开记录。\n',
                '| Figure | Data file | Real/mock | Source | Script | Outputs |\n|---|---|---|---|---|---|\n']
    for record in records:
        manifest.append(f"| {_text(record['model_id'])} / {record['task']} | [{Path(record['data']).name}]({Path(record['data']).relative_to('figures')}) | {'synthetic [待真实实验替换]' if synthetic else 'real'} | ../metrics.csv; {record['metric']} | `{source_file}` | [PNG]({Path(record['png']).name}) / [SVG]({Path(record['svg']).name}) |\n")
    manifest.append('\n## 输入 SHA-256\n\n' + '\n'.join(f"- `{entry['path']}`: `{entry['sha256']}`" for entry in inputs) + f'\n- report.py: `{_sha(source_file)}`\n')
    manifest.append('\n## 绘图数据 SHA-256\n\n' + '\n'.join(f"- `{record['data']}`: `{record['data_sha256']}`" for record in records) + '\n')
    manifest_path = figures_dir / 'data-manifest.md'
    manifest_path.write_text(_join_markdown(manifest), encoding='utf-8')
    outputs = []
    for record in records:
        group = [dict(row, _task_label=_task_label(record['task'], config)) for row in groups[(record['model_id'], record['scheme'], record['task'])]]
        outputs.extend(_heatmaps(group, domains, stages, record['metric'], output_dir / Path(record['png']).with_suffix(''), synthetic))
    (output_dir / 'README.md').write_text(_markdown_report(rows, audit, config, domains, stages, groups, records, synthetic, complete), encoding='utf-8')
    plot_index = ['# NanoRepProbe 图表\n', '**[待真实实验替换] synthetic smoke 数据。**\n' if synthetic else '',
                  '颜色显示共同类别 macro recall，若整组缺少该指标则明确改用全部目标类别宏召回；所有热图均使用 0–1 色标。行是训练域，列是测试域，方框对角线仍使用 held-out reads。留一域表及解释范围见 [主报告](../README.md)。\n']
    for record in records:
        plot_index.append(f"- {record['model_id']} / {record['scheme']} / {record['task']}：[PNG](../{record['png']}) · [SVG](../{record['svg']}) · [数据](../{record['data']})\n")
    plot_index.append('\n[数据清单](../figures/data-manifest.md)\n')
    (reports_dir / 'PLOTS.md').write_text(_join_markdown(plot_index), encoding='utf-8')
    for entry in inputs:
        if _sha(output_dir / entry['path']) != entry['sha256']:
            raise ValueError(f"Report input changed during rendering: {entry['path']}")
    generated = outputs + [output_dir / record['data'] for record in records] + [manifest_path, output_dir / 'README.md', reports_dir / 'PLOTS.md']
    result = {'status': 'passed' if complete else 'partial', 'data_kind': 'synthetic' if synthetic else 'real',
              'metric_rows': len(rows), 'domains': domains, 'stages': stages, 'tasks': tasks, 'figure_count': len(records),
              'png_count': len(outputs) // 2, 'svg_count': len(outputs) // 2,
              'dpi': 450, 'color_limits': [0, 1], 'coverage': coverage, 'inputs': inputs,
              'figures': records, 'artifacts': [{'path': str(path.relative_to(output_dir)), 'sha256': _sha(path), 'bytes': path.stat().st_size} for path in generated]}
    _write_json(reports_dir / 'report.validation.json', result)
    return result
