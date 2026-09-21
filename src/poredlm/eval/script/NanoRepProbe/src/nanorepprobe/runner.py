"""Reproducible cross-domain evaluation from frozen row-indexed features."""
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from datetime import datetime, timezone

import numpy as np
import scipy
from threadpoolctl import threadpool_limits, threadpool_info
from .data import load_config, load_domains, audit_domains
from .linear import fit_system, predict
from .metrics import classification_metrics, read_bootstrap


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def digest(path):
    value = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def csv_write(path, rows):
    if not rows:
        raise ValueError(f'no rows for {path}')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def task_data(domain, task, config):
    labels = domain.labels
    if task == 'kmer':
        return np.ones(len(labels), bool), labels
    if task == 'center_base':
        return np.ones(len(labels), bool), np.array([k[len(k)//2] for k in labels])
    if task == 'same_center_g':
        chosen = config.get('same_center_kmers', ['AAGAA', 'CCGCC', 'TTGTT'])
        if len(set(chosen)) < 2 or any(k[len(k)//2] != 'G' for k in chosen):
            raise ValueError('same_center_kmers requires at least two different kmers centered on G')
        return np.isin(labels, chosen), labels
    raise ValueError(f'unknown task {task}')


def protocol(config):
    return '''# NanoRepProbe 评测协议\n\n冻结输入表征；不训练编码器。训练源域的 physical_read_id 分组 train，使用源域 validation 选择正则化，最终仅在 test 上报告。任何测试域数据均不参与 scaler/系数/超参数拟合。\n\n采用有截距的线性 ridge 分类器：sum_i w_i ||onehot(y_i)-b-standardize(x_i)W||² + lambda ||W||²；每个源域等权、域内各训练类别等权，sum(w)=1。均值和方差仅用这些源域训练权重计算，截距不正则化。lambda 用各源域验证集 macro recall 的等权平均选择，平分时选更大的 lambda。此处为线性可读性测试，不是充分训练后的最优分类上限。\n\n三域互测包含同域参考；leave-one-domain-out 用两个域训练并只测试第三域。比较连续 CNN/BERT/DLM；CNN 是较早表征阶段基线，并报告均匀随机类别期望。kmer 为完整标签分类，center_base 为独立中心碱基分类器，same_center_g 为预先固定的中心G三类独立分类器。\n\nmacro recall/macro F1 对目标测试集有真实支持的类别取平均；common_macro_* 对所有配置域共同有测试支持的固定类别取平均。模型仍输出完整训练类别，评测不丢弃不属于共同集的预测。F1 假阳性保留全测试集贡献；缺失标签不虚构测试数据。完整混淆矩阵与逐样本 top5 预测保留。\n\n区间为 physical_read_id 分组 Bayesian bootstrap 的2.5/97.5百分位：每条物理read获得独立Exp(1)权重，其所有窗口共享权重。保持稀有类别支持，仅反映固定分类器下测试read重加权不确定性，不含编码器或probe重训练，不宣称频率学覆盖率。\n\n限制：read隔离并非参考位点隔离，预训练重叠未知；相位校准置信度以config/audit元信息为准，缺少证据时不能推断逐碱基定位准确。DNA数据集可能包含平台、文库与来源差异，结果是跨数据集迁移而非纯马达因果效应。RNA、修饰保留能力、未见参考位点、其他读出策略需独立实验。本轮不据PCA重合判定能力，不根据测试结果重选lambda。V003+Stone与V006+Apple为模型及预处理组合比较。\n'''


def snapshot_source(output):
    root = Path(__file__).resolve().parent
    target = output / 'provenance' / 'nanorepprobe'
    target.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for source in sorted(root.glob('*.py')):
        hashes[source.name] = digest(source)
        shutil.copy2(source, target / source.name)
    return hashes


def source_hashes():
    return {p.name: digest(p) for p in sorted(Path(__file__).resolve().parent.glob('*.py'))}


def run(config_path, output_dir, resume=False, threads=4):
    config = load_config(config_path)
    config.setdefault('tasks', ['kmer', 'center_base', 'same_center_g'])
    config.setdefault('same_center_kmers', ['AAGAA', 'CCGCC', 'TTGTT'])
    config.setdefault('regularization', [0.0001, 0.01, 1.0, 100.0])
    config.setdefault('bootstrap_repeats', 200)
    config.setdefault('seed', 20260917)
    if threads < 1:
        raise ValueError('threads must be positive')
    if config['bootstrap_repeats'] < 20:
        raise ValueError('at least 20 bootstrap repeats required')
    if not config['regularization'] or any(not np.isfinite(v) or v <= 0 for v in config['regularization']):
        raise ValueError('regularization candidates must be positive finite numbers')
    if len(set(config['tasks'])) != len(config['tasks']):
        raise ValueError('duplicate tasks')
    output = Path(output_dir).resolve()
    code_hashes = source_hashes()
    fingerprint = hashlib.sha256(json.dumps({'config': config, 'source': code_hashes}, sort_keys=True).encode()).hexdigest()
    identity_path = output / 'identity.json'
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise ValueError('output is nonempty; use a new run directory or --resume')
        if not identity_path.exists() or json.loads(identity_path.read_text())['fingerprint'] != fingerprint:
            raise ValueError('resume refused: config/tool source changed')
    output.mkdir(parents=True, exist_ok=True)
    dump(identity_path, {'fingerprint': fingerprint, 'source_hashes': code_hashes})
    dump(output / 'config.resolved.json', config)
    snapshot_source(output)
    (output / 'plan').mkdir(exist_ok=True)
    (output / 'plan' / 'experiment-protocol.md').write_text(protocol(config))
    (output / 'plan' / 'review').mkdir(exist_ok=True)
    (output / 'plan' / 'review' / 'method-experiment-traceability.md').write_text(
        '| 检验内容 | 证据 | 允许结论 |\n|---|---|---|\n'
        '| 序列可读性 | 同域test probe | 当前线性读出能保留多少序列信息 |\n'
        '| 跨域迁移 | 单源互测及双源留出 | 固定识别规则能否跨数据集使用 |\n'
        '| 上下文信息 | 同中心G三类probe | 当前三种上下文能否区分；不外推全体序列 |\n'
        '| 修饰/位点泛化 | 未运行 | 无相关性能结论 |\n')
    (output / 'tables').mkdir(exist_ok=True)
    (output / 'tables' / 'table-schema.md').write_text(
        'metrics.csv：每行一个stage/task/训练域组合/测试域；指标来自test预测。'
        'common_macro_*为固定共同测试类别宏平均；*_ci_*为read分组贝叶斯bootstrap区间。'
        'validation.csv仅源域验证集，用于选lambda；per_class.csv为逐类test计数/召回/F1。\n')
    print('Loading and auditing frozen manifests/features', flush=True)
    domains = load_domains(config)
    audit = audit_domains(domains)
    old_audit = output / 'audit.json'
    if resume and old_audit.exists():
        previous = json.loads(old_audit.read_text())
        previous_inputs = {entry['path']: entry['sha256'] for entry in previous['inputs']}
        current_inputs = {entry['path']: entry['sha256'] for entry in audit['inputs']}
        if previous_inputs != current_inputs:
            raise ValueError('resume refused: input content changed')
    audit['protocol_notes'] = {'encoder_frozen': True, 'test_used_for_selection': False,
        'split': 'probe_split by physical_read_id', 'normalization': config.get('normalization'),
        'phase_confidence': config.get('phase_confidence', 'unknown'),
        'pretraining_overlap': config.get('pretraining_overlap', 'unknown'),
        'reference_locus_holdout': config.get('reference_locus_holdout', False)}
    dump(old_audit, audit)
    started = time.monotonic()
    names = list(domains)
    if len(names) < 2:
        raise ValueError('cross-domain evaluation requires at least two domains')
    experiments = [('single_domain', [name], names) for name in names]
    if len(names) >= 3:
        experiments += [('leave_one_domain_out', [n for n in names if n != target], [target]) for target in names]
    all_metrics, all_validation, all_per_class = [], [], []
    jobs_total = len(config['stages']) * len(config['tasks']) * len(experiments)
    done = 0
    with threadpool_limits(limits=threads):
        dump(output / 'environment.json', {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
            'platform': platform.platform(), 'threadpools': threadpool_info(), 'threads': threads,
            'command': sys.argv, 'started_at': datetime.now(timezone.utc).isoformat()})
        for stage in config['stages']:
            features = {name: domains[name].load_features(stage) for name in names}
            dimensions = {values.shape[1] for values in features.values()}
            if len(dimensions) != 1:
                raise ValueError('feature dimensions differ across domains')
            for task in config['tasks']:
                task_rows = {name: task_data(domains[name], task, config) for name in names}
                classes = np.array(sorted(set(itertools.chain.from_iterable(
                    labels[mask].tolist() for mask, labels in task_rows.values()))))
                label_index = {label: i for i, label in enumerate(classes)}
                indices = {name: {split: np.flatnonzero(mask & (domains[name].splits == split))
                                 for split in ['train', 'validation', 'test']}
                           for name, (mask, _) in task_rows.items()}
                labels = {name: np.array([label_index[label] if label in label_index else -1
                                         for label in task_rows[name][1]]) for name in names}
                common = sorted(set.intersection(*(set(task_rows[n][1][indices[n]['test']]) for n in names)))
                if not common:
                    raise ValueError(f'{task} has no shared test labels')
                for experiment, sources, targets in experiments:
                    source_label = '+'.join(sources)
                    job_name = f'{stage}__{task}__{source_label}'
                    job_dir = output / 'jobs' / job_name
                    complete_path = job_dir / 'complete.json'
                    if resume and complete_path.exists():
                        saved = json.loads(complete_path.read_text())
                        for relative, expected in saved['artifacts'].items():
                            path = job_dir / relative
                            if not path.is_file() or digest(path) != expected:
                                raise ValueError(f'resume artifact hash mismatch: {path}')
                        rows, val_rows, class_rows = saved['metrics'], saved['validation'], saved['per_class']
                        print(f'Resumed {job_name}', flush=True)
                    else:
                        job_dir.mkdir(parents=True, exist_ok=True)
                        job_start = time.monotonic()
                        train_x = np.concatenate([features[n][indices[n]['train']] for n in sources])
                        train_y = np.concatenate([task_rows[n][1][indices[n]['train']] for n in sources])
                        train_domains = np.concatenate([np.repeat(n, len(indices[n]['train'])) for n in sources])
                        system = fit_system(train_x, train_y, train_domains, classes)
                        del train_x
                        val_rows, candidates = [], []
                        for regularization in sorted(set(config['regularization'])):
                            coef = system.coefficients(regularization)
                            domain_scores = []
                            for source in sources:
                                idx = indices[source]['validation']
                                predicted, top = predict(system, features[source][idx], coef)
                                score, _, _ = classification_metrics(labels[source][idx], predicted, top, classes, classes)
                                domain_scores.append(score['macro_recall'])
                                val_rows.append({'stage': stage, 'task': task, 'experiment': experiment,
                                    'train_domains': source_label, 'validation_domain': source,
                                    'regularization': regularization, 'macro_recall': score['macro_recall'],
                                    'n_validation': len(idx), 'n_classes_validation': score['n_classes_evaluated']})
                            candidates.append((float(np.mean(domain_scores)), regularization))
                        selected_score, selected = max(candidates, key=lambda pair: (pair[0], pair[1]))
                        coef = system.coefficients(selected)
                        np.savez_compressed(job_dir / 'probe.npz', mean=system.mean, scale=system.scale,
                            coefficients=coef, intercept=system.target_mean, classes=classes,
                            regularization=selected, n_train=system.n_samples)
                        dump(job_dir / 'fit.json', {'sources': sources, 'targets': targets, 'task': task, 'stage': stage,
                            'classes': classes.tolist(), 'common_test_classes': common,
                            'selected_lambda': selected, 'validation_macro_recall': selected_score,
                            'n_train': system.n_samples, 'constant_features': system.constant_features,
                            'candidates': [{'lambda': value, 'source_validation_macro_recall': score}
                                           for score, value in candidates],
                            'seconds_to_fit_and_select': time.monotonic() - job_start})
                        rows, class_rows = [], []
                        for target in targets:
                            idx = indices[target]['test']
                            predicted, top = predict(system, features[target][idx], coef)
                            values, matrix, per_class = classification_metrics(labels[target][idx], predicted, top, classes, common)
                            # Same target/task seed across source models enables paired read reweighting.
                            seed_key = f"{config['seed']}:{task}:{target}"
                            draw_seed = int(hashlib.sha256(seed_key.encode()).hexdigest()[:8], 16)
                            intervals = read_bootstrap(labels[target][idx], predicted, domains[target].groups[idx],
                                                      classes, common, config['bootstrap_repeats'], draw_seed)
                            row = {'model_id': config['model_id'], 'scheme': config['scheme'], 'stage': stage,
                                'task': task, 'train_domains': source_label, 'test_domain': target,
                                'experiment': experiment, **values, **intervals,
                                'selected_lambda': selected, 'source_validation_macro_recall': selected_score,
                                'n_train': system.n_samples}
                            rows.append(row)
                            for item in per_class:
                                class_rows.append({'stage': stage, 'task': task, 'train_domains': source_label,
                                                   'test_domain': target, 'experiment': experiment, **item})
                            np.savez_compressed(job_dir / f'{target}.predictions.npz',
                                sample_index=domains[target].sample_indices[idx],
                                sample_id=domains[target].sample_ids[idx], physical_read_id=domains[target].groups[idx],
                                true_index=labels[target][idx], predicted_index=predicted, top5_indices=top, classes=classes)
                            np.savez_compressed(job_dir / f'{target}.confusion.npz', counts=matrix, classes=classes)
                        artifacts = {str(p.relative_to(job_dir)): digest(p) for p in sorted(job_dir.glob('*'))
                                     if p.is_file() and p.name != 'complete.json'}
                        dump(complete_path, {'artifacts': artifacts, 'metrics': rows, 'validation': val_rows,
                                             'per_class': class_rows})
                        print(f'{job_name}: lambda={selected:g}, val={selected_score:.4f}, '
                              f'test common recall=' + ', '.join(f"{r['test_domain']}:{r['common_macro_recall']:.4f}" for r in rows)
                              + f' ({time.monotonic()-job_start:.1f}s)', flush=True)
                    all_metrics.extend(rows)
                    all_validation.extend(val_rows)
                    all_per_class.extend(class_rows)
                    done += 1
                    csv_write(output / 'metrics.csv', all_metrics)
                    csv_write(output / 'validation.csv', all_validation)
                    csv_write(output / 'per_class.csv', all_per_class)
                    dump(output / 'status.json', {'status': 'running', 'completed_jobs': done,
                         'total_jobs': jobs_total, 'elapsed_seconds': time.monotonic() - started})
    dump(output / 'status.json', {'status': 'metrics_complete', 'completed_jobs': done,
        'total_jobs': jobs_total, 'metrics_rows': len(all_metrics), 'elapsed_seconds': time.monotonic() - started})
    from .report import render_report
    report = render_report(output)
    if report['status'] != 'passed':
        raise ValueError('report indicates incomplete experiment coverage')
    dump(output / 'status.json', {'status': 'complete', 'completed_jobs': done, 'total_jobs': jobs_total,
        'metrics_rows': len(all_metrics), 'elapsed_seconds': time.monotonic() - started, 'report': report})
    write_snapshot(output)
    print(f'Complete: {output} ({len(all_metrics)} metric rows)', flush=True)
    return output


def write_snapshot(output):
    output = Path(output)
    paths = sorted(p for p in output.rglob('*') if p.is_file() and p.name != 'SHA256SUMS'
                   and p.suffix not in {'.log', '.tmp', '.pyc'})
    (output / 'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.relative_to(output)}\n' for p in paths))


def validate_run(output):
    output = Path(output).resolve()
    snapshot = output / 'SHA256SUMS'
    if not snapshot.exists():
        raise ValueError('run has no SHA256SUMS')
    checked = 0
    for line in snapshot.read_text().splitlines():
        expected, relative = line.split('  ', 1)
        path = (output / relative).resolve()
        if not path.is_relative_to(output):
            raise ValueError('snapshot entry escapes output directory')
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'output checksum mismatch: {relative}')
        checked += 1
    status = json.loads((output / 'status.json').read_text())
    if status['status'] != 'complete' or status['completed_jobs'] != status['total_jobs']:
        raise ValueError('run is incomplete')
    with (output / 'metrics.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    seen = set()
    for row in rows:
        key = tuple(row[k] for k in ['stage','task','train_domains','test_domain'])
        if key in seen:
            raise ValueError(f'duplicate metrics: {key}')
        seen.add(key)
        job_dir = output / 'jobs' / '__'.join(key[:3])
        with np.load(job_dir / f'{row["test_domain"]}.predictions.npz', allow_pickle=False) as data:
            fit = json.loads((job_dir / 'fit.json').read_text())
            values, matrix, _ = classification_metrics(data['true_index'], data['predicted_index'],
                                                       data['top5_indices'], data['classes'], fit['common_test_classes'])
            for name, value in values.items():
                if not np.isclose(float(row[name]), value, rtol=1e-10, atol=1e-12):
                    raise ValueError(f'metric mismatch: {key} {name}')
            with np.load(job_dir / f'{row["test_domain"]}.confusion.npz', allow_pickle=False) as stored:
                if not np.array_equal(matrix, stored['counts']):
                    raise ValueError(f'confusion mismatch: {key}')
    return {'status': 'passed', 'checked_files': checked, 'checked_metric_rows': len(rows)}
