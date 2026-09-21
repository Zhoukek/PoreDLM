#!/usr/bin/env python3
"""Paired k-mer spatial views using frozen corpora and frozen model checkpoints."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tarfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('HF_HOME', '/tmp/model_eval/hf_cache')
os.environ.setdefault('HF_MODULES_CACHE', '/tmp/model_eval/hf_cache/modules')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/model_eval/mpl_cache')
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')

CASE_BASE = Path(__file__).resolve().parent
CASE_PROJECT = CASE_BASE.parent.parent
CASE_RUN = '20260916_kmer_space'
CASE_DOMAINS = ('cyclone-dna', 'ont-r10-hg002', 'dna-amplicon', 'cyclone-rna')
CASE_STAGES = ('vqe_cnn', 'bert', 'dlm_ode')
CASE_MODELS = ('V003', 'V006')


def case_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def case_dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def case_save(path, array):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('wb') as handle:
        np.save(handle, array, allow_pickle=False)
    temporary.replace(path)


def case_helper():
    path = CASE_BASE / 'evaluate.py'
    spec = importlib.util.spec_from_file_location('frozen_evaluation_helpers', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module


def case_location(model):
    return CASE_BASE / f'HF_VQE768C08A001_DNADLLM_{model}' / 'runs' / CASE_RUN


def case_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def case_prepare(args):
    helper = case_helper()
    selected = list(dict.fromkeys([args.single_kmer, *args.multi_kmers]))
    if len(args.multi_kmers) != 3 or len(selected) != 4:
        raise ValueError('Choose three distinct multi-kmer labels and a different single-kmer label')
    if any(len(k) != 5 or set(k) - set('ACGT') for k in selected):
        raise ValueError('All selected labels must be canonical DNA-style 5-mers')
    schemes = ['chunk_center', args.comparison]
    if args.comparison not in ('chunk_5mer', 'short_5mer'):
        raise ValueError('Unsupported comparison')
    for model in CASE_MODELS:
        source_config, source_run = helper.load_run(model)
        helper.verify_sources(source_config)
        run = case_location(model)
        config = {
            'schema_version': 1, 'model': model, 'model_id': source_config['model_id'],
            'strategy': source_config['strategy'], 'schemes': schemes,
            'single_kmer': args.single_kmer, 'multi_kmers': args.multi_kmers,
            'selected_kmers': selected, 'seed': 20260916,
            'source_run': str(source_run), 'corpus_root': source_config['corpus_root'],
            'source_run_config_sha256': case_digest(source_run / 'config/run.json'),
            'source_run_snapshot_sha256': case_digest(source_run / 'SHA256SUMS'),
            'source_corpus_snapshot_sha256': source_config['corpus_checksums_sha256'],
            'source_runner_sha256': source_config['runner_sha256'],
            'runner_sha256': case_digest(Path(__file__)),
            'stages': list(CASE_STAGES), 'domains': list(CASE_DOMAINS),
            'signal_preprocessing': 'Use frozen source arrays directly; no extra normalization or clipping',
            'label_orientation': 'Reference sequence order in frozen manifest, including RNA; do not reverse signal',
            'pooling': {
                'chunk_center': 'Full 6000-point chunk; average token centers in [center_signal_start,center_signal_end)',
                'chunk_5mer': 'Full 6000-point chunk; average token centers in [signal_start,signal_end)',
                'short_5mer': 'Crop frozen [signal_start,signal_end), encode without external context, average all ceil(length/5) content tokens',
                'token_centers': '5*j within the model input',
                'special_tokens': 'BOS/EOS excluded', 'accumulation': 'float64; saved float32',
            },
            'inference': {'precision': 'float32', 'tf32': False, 'batch_size': 32,
                          'vq_tile_tokens': source_config['vq_tile_tokens'], 'ode': source_config['ode']},
            'center_replay_tolerance': {'atol': 3e-5, 'rtol': 3e-4},
            'pca': 'L2-normalized representations; one shared full-SVD PCA per model/case/stage across both schemes, all selected kmers and all four corpora',
            'datasets': {},
        }
        manifest_frames = {}
        for domain in CASE_DOMAINS:
            original = Path(source_config['corpus_root']) / domain / 'manifest.csv'
            rows = [dict(row) for row in case_csv(original) if row['kmer'] in selected]
            if not rows:
                raise RuntimeError(f'No selected samples in {domain}')
            indices = [int(row['sample_index']) for row in rows]
            if indices != sorted(set(indices)):
                raise RuntimeError('Source sample indices are not unique and sorted')
            for i, row in enumerate(rows):
                row['case_index'] = str(i)
                a,b,c,d = (int(row[x]) for x in ('signal_start','signal_end','center_signal_start','center_signal_end'))
                if not 0 <= a <= c < d <= b <= 6000:
                    raise RuntimeError('Center and whole-kmer signal intervals do not nest')
                helper.pooling_bounds([a], [b]); helper.pooling_bounds([c], [d])
            counts = {k: sum(row['kmer'] == k for row in rows) for k in selected}
            if any(n != 100 for n in counts.values()):
                raise RuntimeError('This paired protocol requires exactly 100 frozen samples per requested group')
            buffer = io.StringIO(newline='')
            writer = csv.DictWriter(buffer, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            payload = buffer.getvalue()
            manifest_frames[domain] = (rows, payload)
            config['datasets'][domain] = {
                'source_manifest_sha256': case_digest(original),
                'selected_manifest_sha256': hashlib.sha256(payload.encode()).hexdigest(), 'samples': len(rows),
                'kmer_counts': counts, 'unique_chunks': len({row['signal_row'] for row in rows}),
            }
        old = run / 'config/run.json'
        if old.exists() and json.loads(old.read_text()) != config:
            raise RuntimeError(f'Existing spatial run has a different protocol: {run}; preserve it and choose a new CASE_RUN')
        case_dump(old, config)
        for domain, (rows, payload) in manifest_frames.items():
            path = run / 'manifests' / f'{domain}.csv'; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload.encode())
        if not (run / 'reports/status.json').exists():
            case_dump(run / 'reports/status.json', {'status': 'prepared', 'schemes': schemes})
        print(model, 'prepared', sum(x['samples'] for x in config['datasets'].values()), 'paired observations per scheme', flush=True)


def case_load(model):
    run = case_location(model)
    config = json.loads((run / 'config/run.json').read_text())
    if config['runner_sha256'] != case_digest(Path(__file__)):
        raise RuntimeError('Spatial runner changed after registration')
    helper = case_helper(); source_config, source_run = helper.load_run(model)
    checks = [(source_run/'config/run.json', config['source_run_config_sha256']),
              (source_run/'SHA256SUMS', config['source_run_snapshot_sha256']),
              (Path(source_config['corpus_root'])/'SHA256SUMS', config['source_corpus_snapshot_sha256'])]
    for path, expected in checks:
        if case_digest(path) != expected:
            raise RuntimeError(f'Frozen source changed: {path}')
    return helper, config, run, source_config, source_run


def case_extract(args):
    started = time.monotonic()
    helper, config, run, source_config, source_run = case_load(args.model)
    domain = args.dataset
    helper.verify_sources(source_config, domain)
    config_sha = case_digest(run / 'config/run.json')
    completion = run / 'reports' / f'extract.{domain}.json'
    if case_digest(run / 'manifests' / f'{domain}.csv') != config['datasets'][domain]['selected_manifest_sha256']:
        raise RuntimeError('Selected manifest changed after registration')
    if completion.exists():
        complete = json.loads(completion.read_text())
        if complete['config_sha256'] != config_sha:
            raise RuntimeError('Previous extraction belongs to another configuration')
        for rel, expected in complete['outputs_sha256'].items():
            if case_digest(run / rel) != expected:
                raise RuntimeError('Previously completed spatial representation changed')
        print(args.model, domain, 'complete; verified existing outputs', flush=True)
        return
    rows = case_csv(run / 'manifests' / f'{domain}.csv')
    source_manifest = Path(source_config['corpus_root']) / domain / 'manifest.csv'
    if case_digest(source_manifest) != config['datasets'][domain]['source_manifest_sha256']:
        raise RuntimeError('Source manifest changed')
    selected = np.asarray([int(row['sample_index']) for row in rows], dtype=np.int64)
    if [int(row['case_index']) for row in rows] != list(range(len(rows))):
        raise RuntimeError('Invalid case row order')
    original_completion = json.loads((source_run/'representations'/domain/'complete.json').read_text())
    cached = {}
    for stage in CASE_STAGES:
        path = source_run/'representations'/domain/(stage+'.npy')
        if case_digest(path) != original_completion['sha256'][stage]:
            raise RuntimeError('Existing center representation differs from completed source run')
        cached[stage] = np.array(np.load(path, mmap_mode='r')[selected], dtype=np.float32)
    torch = helper.setup_torch(args.device)
    codec, dlm = helper.load_models(source_config, args.device)
    signals = np.load(Path(source_config['corpus_root'])/domain/'prep/signals.npy', mmap_mode='r')
    results = {scheme: {stage: np.full((len(rows),768), np.nan, np.float32) for stage in CASE_STAGES}
               for scheme in config['schemes'] if scheme != 'chunk_center'}
    parity = {stage: {'max_abs_error': 0.0, 'checked_samples': 0} for stage in CASE_STAGES}
    center_checked = np.zeros(len(rows), bool)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[int(row['signal_row'])].append(i)
    chunk_rows = sorted(groups)
    # Recompute center for every selected sample while obtaining the wider readout.
    # This checks source membership, local bounds, and the effect of re-batching.
    if 'chunk_5mer' in config['schemes']:
        for offset in range(0, len(chunk_rows), 32):
            picked = chunk_rows[offset:offset+32]
            maps, _ = helper.forward_maps(codec, dlm, np.array(signals[picked], dtype=np.float32), source_config)
            local, batch_rows = [], []
            for batch_index, signal_row in enumerate(picked):
                local.extend(groups[signal_row]); batch_rows.extend([batch_index]*len(groups[signal_row]))
            chosen_rows = [rows[i] for i in local]
            center_left, center_right = helper.pooling_bounds([int(x['center_signal_start']) for x in chosen_rows], [int(x['center_signal_end']) for x in chosen_rows])
            whole_left, whole_right = helper.pooling_bounds([int(x['signal_start']) for x in chosen_rows], [int(x['signal_end']) for x in chosen_rows])
            for stage, features in maps.items():
                center = helper.mean_pool_intervals_fp64(features, batch_rows, center_left, center_right).cpu().numpy()
                target = cached[stage][local]
                error = float(np.max(np.abs(center - target)))
                parity[stage]['max_abs_error'] = max(parity[stage]['max_abs_error'], error)
                parity[stage]['checked_samples'] += len(local)
                if not np.allclose(center, target, **config['center_replay_tolerance']):
                    raise RuntimeError(f'{stage}: re-extracted center differs from frozen result (max abs {error})')
                results['chunk_5mer'][stage][local] = helper.mean_pool_intervals_fp64(features, batch_rows, whole_left, whole_right).cpu().numpy()
            center_checked[local] = True
            print(args.model, domain, f'{min(offset+32,len(chunk_rows))}/{len(chunk_rows)} chunks', f'{time.monotonic()-started:.1f}s', flush=True)
        if not center_checked.all():
            raise RuntimeError('Some selected centers were not checked')
    if 'short_5mer' in config['schemes']:
        by_length = defaultdict(list)
        for i,row in enumerate(rows):
            by_length[int(row['signal_end'])-int(row['signal_start'])].append(i)
        count = 0
        for length, same_length in sorted(by_length.items()):
            for offset in range(0,len(same_length),32):
                local=same_length[offset:offset+32]
                crops=np.stack([np.asarray(signals[int(rows[i]['signal_row']),int(rows[i]['signal_start']):int(rows[i]['signal_end'])], dtype=np.float32) for i in local])
                maps,_=helper.forward_maps(codec,dlm,crops,source_config)
                for stage,features in maps.items():
                    results['short_5mer'][stage][local]=features.double().mean(dim=1).float().cpu().numpy()
                count+=len(local)
        if count != len(rows):raise RuntimeError('Short-window sample coverage incomplete')
        parity={'note':'Center scheme copied exactly from completed source run; no extra full-chunk pass for short-window comparison'}
    outputs={}
    for scheme in config['schemes']:
        for stage in CASE_STAGES:
            array = cached[stage] if scheme == 'chunk_center' else results[scheme][stage]
            if array.shape != (len(rows),768) or not np.isfinite(array).all() or np.any(np.linalg.norm(array,axis=1)<=0):
                raise RuntimeError('Invalid selected representations')
            path=run/'representations'/scheme/domain/(stage+'.npy');case_save(path,array)
            outputs[str(path.relative_to(run))]=case_digest(path)
    case_dump(completion, {
        'status':'passed','model':args.model,'domain':domain,'samples':len(rows),'unique_chunks':len(chunk_rows),
        'config_sha256':config_sha,'source_run_config_sha256':config['source_run_config_sha256'],
        'center_replay':parity,'outputs_sha256':outputs,'seconds':time.monotonic()-started,
    })
    print(args.model,domain,'spatial extraction PASSED',flush=True)


def case_plot(model):
    helper,config,run,source_config,source_run=case_load(model)
    config_sha=case_digest(run/'config/run.json')
    for domain in CASE_DOMAINS:
        if case_digest(run/'manifests'/f'{domain}.csv') != config['datasets'][domain]['selected_manifest_sha256']:
            raise RuntimeError('Selected manifest changed after registration')
        report=json.loads((run/'reports'/f'extract.{domain}.json').read_text())
        if report['status']!='passed' or report['config_sha256']!=config_sha:
            raise RuntimeError('Spatial extraction incomplete or configuration mismatch')
        for rel,expected in report['outputs_sha256'].items():
            if case_digest(run/rel)!=expected:raise RuntimeError('Spatial extraction output changed')
    render_cases(run)
    case_dump(run/'reports/status.json',{'status':'completed','model':model,'strategy':config['strategy'],
        'schemes':config['schemes'],'samples_per_scheme':sum(d['samples'] for d in config['datasets'].values()),
        'single_kmer':config['single_kmer'],'multi_kmers':config['multi_kmers'],'stages':list(CASE_STAGES)})
    (run/'provenance').mkdir(parents=True, exist_ok=True)
    with tarfile.open(run/'provenance/source.tar.gz','w:gz') as archive:
        archive.add(Path(__file__),arcname='kmer_space.py')
        archive.add(CASE_BASE/'evaluate.py',arcname='frozen_evaluate.py')
    case_snapshot(run)
    print(model,'ALL SPATIAL FIGURES PASSED',flush=True)


def case_snapshot(run):
    paths=sorted(p for p in run.rglob('*') if p.is_file() and p.name!='SHA256SUMS' and p.suffix!='.log' and not p.name.endswith('.partial'))
    (run/'SHA256SUMS').write_text(''.join(f'{case_digest(p)}  {p.relative_to(run)}\n' for p in paths))


def case_run_all(args):
    jobs=queue.Queue()
    for model in CASE_MODELS:
        for domain in CASE_DOMAINS:jobs.put((model,domain))
    def worker(gpu):
        while True:
            try:model,domain=jobs.get_nowait()
            except queue.Empty:return
            run=case_location(model);log=run/'logs'/f'extract.{domain}.log';log.parent.mkdir(parents=True,exist_ok=True)
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
            with log.open('a') as handle:
                subprocess.run([sys.executable,str(Path(__file__)),'extract','--model',model,'--dataset',domain,'--device','cuda:0'],env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
            print(model,domain,'complete',flush=True)
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures=[executor.submit(worker,gpu) for gpu in args.gpus]
        for future in futures:future.result()
    for model in CASE_MODELS:case_plot(model)


def case_main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='command',required=True)
    prep=subs.add_parser('prepare')
    prep.add_argument('--single-kmer',default='ACGTA')
    prep.add_argument('--multi-kmers',nargs='+',default=['AAAAA','CCCCC','GGGGG'])
    prep.add_argument('--comparison',choices=['chunk_5mer','short_5mer'],default='chunk_5mer')
    extract=subs.add_parser('extract');extract.add_argument('--model',choices=CASE_MODELS,required=True)
    extract.add_argument('--dataset',choices=CASE_DOMAINS,required=True);extract.add_argument('--device',default='cuda:0')
    plot=subs.add_parser('plot');plot.add_argument('--model',choices=CASE_MODELS,required=True)
    run=subs.add_parser('run');run.add_argument('--gpus',type=int,nargs='+',default=[0,1,2,3])
    args=parser.parse_args()
    if args.command=='prepare':case_prepare(args)
    elif args.command=='extract':case_extract(args)
    elif args.command=='plot':case_plot(args.model)
    else:case_run_all(args)

"""Selected-kmer cross-domain plots, using paired-scheme PCA and 768-D metrics.

This module only reads completed representations.  It never loads a model or
changes an input manifest.  The public entry point is ``render_cases(run)``.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/model-eval-kmer-space-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


DOMAINS = ("cyclone-dna", "ont-r10-hg002", "dna-amplicon", "cyclone-rna")
STAGES = ("vqe_cnn", "bert", "dlm_ode")
COLORS = {"cyclone-dna": "#009988", "ont-r10-hg002": "#EE3377",
          "dna-amplicon": "#CCBB44", "cyclone-rna": "#EE7733"}
DOMAIN_LABELS = {"cyclone-dna": "Cyclone DNA", "ont-r10-hg002": "ONT R10 HG002",
                 "dna-amplicon": "DNA amplicon", "cyclone-rna": "Cyclone RNA"}
STAGE_LABELS = ("Stage 1 · VQE CNN", "Stage 2 · BERT context", "Stage 3 · DLM ODE")
SCHEME_LABELS = {"chunk_center": "Full chunk / center-base pooling",
                 "chunk_5mer": "Full chunk / whole-5-mer pooling",
                 "short_5mer": "Short 5-mer input / valid-token pooling"}
MARKERS = ("o", "^", "s")
PAIR_COLUMNS = ["model", "strategy", "scheme", "case", "stage", "corpus",
                "left_kmer", "right_kmer", "centroid_cosine_distance"]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _normalize(values: np.ndarray) -> np.ndarray:
    # Same float32 convention as NanoRepDist.metrics.l2_normalize.
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("Representations must be finite and have nonzero norms")
    return values / norms


def _centroid(values: np.ndarray) -> np.ndarray:
    center = np.asarray(values, dtype=np.float64).mean(axis=0)
    norm = np.linalg.norm(center)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("A selected group has a zero or invalid centroid")
    return center / norm


def _distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.clip(1.0 - np.dot(left, right), 0.0, 2.0))


def _cases(config: dict) -> dict[str, list[str]]:
    single = config.get("single_kmer", "ACGTA")
    multi = config.get("multi_kmers", ["AAAAA", "CCCCC", "GGGGG"])
    if len(multi) != 3 or len(set([single, *multi])) != 4:
        raise ValueError("Expected one single kmer and three distinct additional multi_kmers")
    if any(len(k) != 5 or not set(k) <= set("ACGT") for k in [single, *multi]):
        raise ValueError("All selected classes must be canonical ACGT 5-mers")
    return {f"single_{single}": [single], "pair_" + "_".join(multi[:2]): multi[:2],
            "triple_" + "_".join(multi): multi}


def _load_inputs(run: Path, config: dict, cases: dict):
    schemes = config.get("schemes", ["chunk_center", "chunk_5mer"])
    if not schemes or len(schemes) != len(set(schemes)) or any(s not in SCHEME_LABELS for s in schemes):
        raise ValueError("schemes must contain distinct supported scheme names")
    targets = set(k for values in cases.values() for k in values)
    expected_per_class = int(config.get("samples_per_kmer", 100))
    if expected_per_class < 2:
        raise ValueError("Covariance plots require at least two samples per group")
    manifests, arrays, inventory = {}, {}, []
    paths = [run / "config/run.json"]
    for domain in DOMAINS:
        path = run / "manifests" / f"{domain}.csv"
        frame = pd.read_csv(path, keep_default_na=False)
        for column in ("kmer", "sample_index", "case_index"):
            if column not in frame:
                raise ValueError(f"Missing {column} in {path}")
        for column in ("sample_index", "case_index"):
            values = pd.to_numeric(frame[column], errors="raise").to_numpy()
            if not np.isfinite(values).all() or np.any(values != values.astype(np.int64)) or np.any(values < 0):
                raise ValueError(f"Invalid integer {column} in {path}")
            frame[column] = values.astype(np.int64)
        frame = frame.sort_values("case_index").reset_index(drop=True)
        if not np.array_equal(frame.case_index.to_numpy(), np.arange(len(frame))):
            raise ValueError(f"case_index must be a permutation of 0..n-1 in {path}")
        if frame.sample_index.duplicated().any():
            raise ValueError(f"Duplicate original sample_index in {path}")
        if set(frame.kmer) != targets or any(frame.kmer.value_counts().get(k, 0) != expected_per_class for k in targets):
            raise ValueError(f"Expected all four selected classes, {expected_per_class} observations each: {path}")
        if "corpus" in frame and not frame.corpus.eq(domain).all():
            raise ValueError(f"Manifest corpus disagrees with path: {path}")
        manifests[domain] = frame
        paths.append(path)
        for scheme in schemes:
            for stage in STAGES:
                rep_path = run / "representations" / scheme / domain / f"{stage}.npy"
                raw = np.load(rep_path, mmap_mode="r", allow_pickle=False)
                if raw.shape != (len(frame), 768) or raw.dtype != np.float32:
                    raise ValueError(f"Expected {(len(frame), 768)} float32: {rep_path}: {raw.shape}, {raw.dtype}")
                values = _normalize(raw)
                if not np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=2e-6, rtol=0):
                    raise ValueError(f"L2 normalization check failed: {rep_path}")
                raw_norms = np.linalg.norm(raw, axis=1)
                arrays[(scheme, domain, stage)] = values
                inventory.append({"scheme": scheme, "corpus": domain, "stage": stage,
                                  "shape": list(raw.shape), "dtype": str(raw.dtype),
                                  "min_raw_norm": float(raw_norms.min()),
                                  "max_raw_norm": float(raw_norms.max()),
                                  "normalized_max_error": float(np.abs(np.linalg.norm(values, axis=1) - 1).max())})
                paths.append(rep_path)
    fingerprints = [{"path": str(path.relative_to(run)), "sha256": _sha(path),
                     "bytes": path.stat().st_size} for path in paths]
    return schemes, manifests, arrays, inventory, fingerprints


def _compute_metrics(config, schemes, cases, manifests, arrays):
    within, cross, between, summaries = [], [], [], []
    for scheme in schemes:
        for case, kmers in cases.items():
            for stage in STAGES:
                key = dict(model=config["model"], strategy=config["strategy"], scheme=scheme, case=case, stage=stage)
                centers, sizes, group_spreads = {}, {}, []
                for domain in DOMAINS:
                    # NanoRepDist's cosine metric path normalizes the L2 input again.
                    values = _normalize(arrays[(scheme, domain, stage)])
                    for kmer in kmers:
                        group = values[manifests[domain].kmer.eq(kmer).to_numpy()]
                        center = _centroid(group)
                        distances = 1.0 - group @ center
                        centers[(domain, kmer)], sizes[(domain, kmer)] = center, len(group)
                        group_spreads.append(float(distances.mean()))
                        within.append({**key, "corpus": domain, "molecule": "rna" if domain.endswith("rna") else "dna",
                                       "kmer": kmer, "observations": len(group),
                                       "mean_cosine_distance": float(distances.mean()),
                                       "median_cosine_distance": float(np.median(distances)),
                                       "std_cosine_distance": float(np.std(distances, ddof=1))})
                cross_distances, between_distances = [], []
                for kmer in kmers:
                    for left, right in combinations(DOMAINS, 2):
                        distance = _distance(centers[(left, kmer)], centers[(right, kmer)])
                        cross_distances.append(distance)
                        cross.append({**key, "kmer": kmer, "left_corpus": left, "right_corpus": right,
                                      "left_observations": sizes[(left, kmer)], "right_observations": sizes[(right, kmer)],
                                      "centroid_cosine_distance": distance})
                for domain in DOMAINS:
                    for left, right in combinations(kmers, 2):
                        distance = _distance(centers[(domain, left)], centers[(domain, right)])
                        between_distances.append(distance)
                        between.append({**key, "corpus": domain, "left_kmer": left, "right_kmer": right,
                                        "centroid_cosine_distance": distance})
                mean_within = float(np.mean(group_spreads))
                mean_between = float(np.mean(between_distances)) if between_distances else None
                summaries.append({**key, "kmers": ";".join(kmers), "groups": len(group_spreads),
                                  "cross_pairs": len(cross_distances), "between_pairs": len(between_distances),
                                  "mean_within_group_distance": mean_within,
                                  "mean_cross_corpus_distance": float(np.mean(cross_distances)),
                                  "mean_between_kmer_distance": mean_between,
                                  "between_within_ratio": mean_between / mean_within if mean_between is not None and mean_within > 0 else None})
    return {"within_groups": pd.DataFrame(within), "corpus_pair_distances": pd.DataFrame(cross),
            "kmer_pair_distances": pd.DataFrame(between, columns=PAIR_COLUMNS), "stage_summary": pd.DataFrame(summaries)}


def _fit_projections(run, config, schemes, cases, manifests, arrays):
    coordinates, pca_rows = [], []
    for case, kmers in cases.items():
        for stage in STAGES:
            parts, metadata = [], []
            for scheme in schemes:
                for domain in DOMAINS:
                    frame = manifests[domain]
                    selected = frame.kmer.isin(kmers).to_numpy()
                    cols = [c for c in ("case_index", "sample_index", "physical_read_id", "read_id", "kmer") if c in frame]
                    part = frame.loc[selected, cols].copy()
                    part.insert(0, "corpus", domain)
                    part.insert(0, "scheme", scheme)
                    parts.append(arrays[(scheme, domain, stage)][selected])
                    metadata.append(part)
            values = np.concatenate(parts)
            pca = PCA(n_components=2, svd_solver="full")
            pca.fit(values)
            points = pca.transform(values)
            if not np.isfinite(points).all() or not np.isfinite(pca.explained_variance_ratio_).all():
                raise ValueError(f"Nonfinite PCA: {case}/{stage}")
            frame = pd.concat(metadata, ignore_index=True)
            pca_id = f"{case}/{stage}"
            for name, value in reversed(list(dict(model=config["model"], strategy=config["strategy"], case=case,
                                                   stage=stage, pca_id=pca_id).items())):
                frame.insert(0, name, value)
            frame["pc1"], frame["pc2"] = points[:, 0], points[:, 1]
            path = run / "analysis/pca" / case / f"{stage}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, components=pca.components_, mean=pca.mean_,
                                explained_variance=pca.explained_variance_,
                                explained_variance_ratio=pca.explained_variance_ratio_,
                                singular_values=pca.singular_values_, n_samples=np.array(len(values)),
                                schemes=np.asarray(schemes), kmers=np.asarray(kmers),
                                domains=np.asarray(DOMAINS), pca_id=np.array(pca_id))
            coordinates.append(frame)
            pca_rows.append({"model": config["model"], "case": case, "stage": stage, "pca_id": pca_id,
                             "schemes": ";".join(schemes), "kmers": ";".join(kmers), "corpora": ";".join(DOMAINS),
                             "observations": len(values), "variance_pc1": float(pca.explained_variance_ratio_[0]),
                             "variance_pc2": float(pca.explained_variance_ratio_[1]),
                             "axis_limit": max(1e-6, 1.10 * float(np.abs(points).max())),
                             "basis_path": str(path.relative_to(run)), "basis_sha256": _sha(path)})
    return pd.concat(coordinates, ignore_index=True), pd.DataFrame(pca_rows)


def _ellipse(axis, points, color, linestyle="-"):
    if len(points) < 2:
        return
    covariance = np.cov(points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * np.sqrt(np.maximum(eigenvalues, 0) * (-2 * np.log(0.20)))
    axis.add_patch(Ellipse(points.mean(axis=0), width, height, angle=angle, facecolor=color,
                           edgecolor=color, alpha=0.075, linewidth=1.2, linestyle=linestyle, zorder=1))


def _plot_case(run, config, scheme, case, kmers, coordinates, pca_summary, summaries):
    style = {"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10.5,
             "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
             "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
             "legend.frameon": False, "grid.alpha": 0.18, "grid.linewidth": 0.6,
             "savefig.dpi": 450, "savefig.bbox": "tight", "svg.fonttype": "none"}
    rows = summaries.loc[(summaries.scheme == scheme) & (summaries.case == case)].set_index("stage")
    basis = pca_summary.loc[pca_summary.case == case].set_index("stage")
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(11.8, 9.2))
        for index, (axis, stage, title) in enumerate(zip([axes[0, 0], axes[0, 1], axes[1, 0]], STAGES, STAGE_LABELS)):
            panel = coordinates.loc[(coordinates.scheme == scheme) & (coordinates.case == case) & (coordinates.stage == stage)]
            for domain in DOMAINS:
                for ki, kmer in enumerate(kmers):
                    points = panel.loc[(panel.corpus == domain) & (panel.kmer == kmer), ["pc1", "pc2"]].to_numpy()
                    _ellipse(axis, points, COLORS[domain], ("-", "--", ":")[ki])
                    axis.scatter(points[:, 0], points[:, 1], s=24 if len(kmers) == 1 else 17,
                                 marker=MARKERS[ki], color=COLORS[domain], alpha=0.64 if len(kmers) == 1 else 0.52,
                                 edgecolor="white", linewidth=0.35, zorder=2)
                    center = points.mean(axis=0)
                    axis.scatter(*center, marker="X", s=74, color=COLORS[domain], edgecolor="white", linewidth=0.8, zorder=3)
                    if len(kmers) > 1:
                        axis.annotate(str(ki + 1), center, xytext=(5, 4), textcoords="offset points",
                                      fontsize=7, color=COLORS[domain], weight="bold", zorder=4,
                                      bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.2})
            info = basis.loc[stage]
            limit = info.axis_limit
            axis.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel="PC1", ylabel="PC2")
            axis.set_aspect("equal", adjustable="box")
            axis.set_title(f"{title}\nPC variance: {100 * info.variance_pc1:.1f}% + {100 * info.variance_pc2:.1f}%")
            axis.grid()
            axis.text(-0.13, 1.09, chr(65 + index), transform=axis.transAxes, fontsize=12, weight="bold", va="top")
        axis = axes[1, 1]
        x = np.arange(3)
        trends = [("mean_cross_corpus_distance", "Same kmer / cross-domain", "#CC3311", "o", "-"),
                  ("mean_within_group_distance", "Within domain × kmer", "#0077BB", "s", "--")]
        if len(kmers) > 1:
            trends.append(("mean_between_kmer_distance", "Different kmers / same domain", "#009988", "^", "-"))
        trend_handles = []
        for column, label, color, marker, line in trends:
            handle, = axis.plot(x, rows.loc[list(STAGES), column], color=color, marker=marker,
                                linestyle=line, linewidth=1.8, label=label)
            trend_handles.append(handle)
        axis.set_xticks(x, ["VQE CNN", "BERT", "DLM ODE"])
        axis.set_ylim(bottom=0)
        axis.set_ylabel("Cosine distance (768-D)")
        axis.set_title("Selected-kmer distance decomposition\nEqual weight per group or centroid pair")
        axis.grid(axis="y")
        axis.text(-0.13, 1.09, "D", transform=axis.transAxes, fontsize=12, weight="bold", va="top")
        domain_handles = [Line2D([], [], color=COLORS[d], marker="o", linestyle="", markersize=7, label=DOMAIN_LABELS[d]) for d in DOMAINS]
        fig.legend(handles=domain_handles, title="Signal type (color)", loc="upper left", bbox_to_anchor=(0.795, 0.87), fontsize=8.3)
        kmer_handles = [Line2D([], [], color="#444444", marker=MARKERS[i], linestyle="", markersize=7,
                               label=f"{i + 1}: {k}" if len(kmers) > 1 else k) for i, k in enumerate(kmers)]
        kmer_handles.append(Line2D([], [], color="#666666", marker="X", linestyle="", markersize=8, label="2D group mean"))
        fig.legend(handles=kmer_handles, title="Kmer (shape)", loc="upper left", bbox_to_anchor=(0.795, 0.65), fontsize=8.3)
        fig.legend(handles=trend_handles, title="Panel D: 768-D metrics", loc="upper left", bbox_to_anchor=(0.795, 0.40), fontsize=7.6)
        fig.text(0.805, 0.22, "Ellipse: nominal 80%\nGaussian covariance contour;\nnot a mean confidence interval.\n\nNo train/test distinction.\nAll selected observations shown.", fontsize=7.8, va="top", linespacing=1.5)
        fig.suptitle(f"{config['model']} ({config['strategy']}) · {SCHEME_LABELS[scheme]}\n" + " / ".join(kmers) + " across four signal types",
                     fontsize=12, weight="semibold", y=0.985)
        fig.text(0.44, 0.035, "Same model / case / stage: one PCA basis and axis range shared by all schemes.\nStages and models have independent PCAs; panel D uses original 768-D vectors.",
                 ha="center", fontsize=8.2, linespacing=1.5)
        fig.subplots_adjust(left=0.075, right=0.77, top=0.87, bottom=0.115, hspace=0.43, wspace=0.30)
        output = run / "figures" / scheme / case
        output.parent.mkdir(parents=True, exist_ok=True)
        files = []
        for suffix in (".png", ".svg"):
            path = output.with_suffix(suffix)
            fig.savefig(path, dpi=450)
            files.append(path)
        plt.close(fig)
    return files


def render_cases(run) -> dict:
    """Create selected-kmer descriptive plots and a complete data audit trail.

    Inputs are config/run.json, manifests/<domain>.csv and
    representations/<scheme>/<domain>/<stage>.npy.  Each manifest retains the
    original sample_index and maps case_index to representation rows.
    """
    run = Path(run).resolve()
    config = json.loads((run / "config/run.json").read_text())
    if config.get("model") not in ("V003", "V006") or config.get("strategy") not in ("stone", "apple"):
        raise ValueError("Expected model V003/V006 and strategy stone/apple")
    if {"V003": "stone", "V006": "apple"}[config["model"]] != config["strategy"]:
        raise ValueError("Model and signal strategy do not match")
    cases = _cases(config)
    schemes, manifests, arrays, inventory, inputs = _load_inputs(run, config, cases)
    analysis, reports = run / "analysis", run / "reports"
    analysis.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    tables = _compute_metrics(config, schemes, cases, manifests, arrays)
    coordinates, pca_summary = _fit_projections(run, config, schemes, cases, manifests, arrays)
    tables.update(coordinates=coordinates, pca_summary=pca_summary)
    data_paths = []
    for name, frame in tables.items():
        path = analysis / f"{name}.csv"
        frame.to_csv(path, index=False, float_format="%.17g")
        data_paths.append(path)
    data_paths.extend(run / p for p in pca_summary.basis_path)
    # This manifest is committed before any rendering: figures are derived from
    # inspectable real coordinates and metrics, with original sample identifiers.
    manifest = {"schema_version": 1, "model": config["model"], "strategy": config["strategy"],
                "schemes": schemes, "cases": cases, "stages": list(STAGES), "domains": list(DOMAINS),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "metric_space": "768-D sample L2; float64 mean normalized to unit centroid; cosine=1-dot",
                "pca_scope": "one pooled full-SVD PCA per model/case/stage, jointly fit all schemes/domains/selected kmers",
                "pca_sample_weighting": "equal observation weight; paired scheme membership identical",
                "ellipse": "nominal 80% Gaussian covariance contour, not a mean confidence interval",
                "centroid_display": "X is arithmetic mean of projected sample coordinates; metric centroid is normalized in 768-D",
                "aggregation": "equal weight per within group, same-kmer domain pair, or within-domain kmer pair",
                "inputs": inputs, "representation_validation": inventory,
                "rows": {name: len(frame) for name, frame in tables.items()},
                "data_artifacts": [{"path": str(p.relative_to(run)), "sha256": _sha(p), "bytes": p.stat().st_size} for p in data_paths]}
    _json(analysis / "data-manifest.json", manifest)
    figures = []
    for scheme in schemes:
        for case, kmers in cases.items():
            figures.extend(_plot_case(run, config, scheme, case, kmers, coordinates, pca_summary, tables["stage_summary"]))
    for entry in inputs:
        if _sha(run / entry["path"]) != entry["sha256"]:
            raise ValueError(f"An input changed during rendering: {entry['path']}")
    expected_counts = {"within_groups": len(schemes) * 3 * 4 * 6,
                       "corpus_pair_distances": len(schemes) * 3 * 6 * 6,
                       "kmer_pair_distances": len(schemes) * 3 * 4 * 4,
                       "stage_summary": len(schemes) * 3 * 3,
                       "pca_summary": 9,
                       "coordinates": len(schemes) * 3 * 4 * 6 * int(config.get("samples_per_kmer", 100))}
    if manifest["rows"] != expected_counts:
        raise ValueError(f"Output row counts disagree: {manifest['rows']} != {expected_counts}")
    if coordinates.duplicated(["scheme", "case", "stage", "corpus", "case_index"]).any():
        raise ValueError("Duplicated coordinate identity")
    for (_, _, _), group in coordinates.groupby(["case", "stage", "corpus"]):
        memberships = [set(zip(part.case_index, part.sample_index, part.kmer)) for _, part in group.groupby("scheme")]
        if len(memberships) != len(schemes) or any(m != memberships[0] for m in memberships):
            raise ValueError("Scheme membership differs within a shared PCA")
    validation = {"status": "passed", "model": config["model"], "strategy": config["strategy"],
                  "schemes": schemes, "cases": list(cases), "rows": manifest["rows"],
                  "figure_count": len(figures), "png_count": sum(p.suffix == ".png" for p in figures),
                  "svg_count": sum(p.suffix == ".svg" for p in figures), "pca_count": len(pca_summary),
                  "shared_pca_verified": True, "paired_membership_verified": True, "inputs_unchanged": True,
                  "data_manifest_sha256": _sha(analysis / "data-manifest.json"),
                  "figures": [{"path": str(p.relative_to(run)), "sha256": _sha(p), "bytes": p.stat().st_size} for p in figures]}
    _json(reports / "plot.validation.json", validation)
    links = []
    for scheme in schemes:
        links.append(f"## {scheme}：{SCHEME_LABELS[scheme]}\n")
        for case, kmers in cases.items():
            rel = f"../figures/{scheme}/{case}"
            links.append(f"- {' / '.join(kmers)}：[PNG]({rel}.png) · [SVG]({rel}.svg)")
        links.append("")
    content = f"""# {config['model']}（{config['strategy']}）跨信号类型空间图

共 {len(schemes) * 3} 张图，各提供 450 dpi PNG 和 SVG。每张沿用 08 的 2×2 布局：前三格为 VQE CNN、BERT context、DLM ODE；第四格为选中 k-mer 在原始 768 维空间中的余弦距离趋势。

颜色固定表示信号类型：Cyclone DNA 绿、ONT R10 HG002 粉、DNA amplicon 黄、Cyclone RNA 橙。点型区分 k-mer；X 为投影样本的二维均值，多类图中旁边数字对应点型图例。椭圆为名义 80% 高斯协方差轮廓，不是均值的置信区间，也不保证经验覆盖率恰好 80%。图中没有训练/测试划分。

同一个模型、同一组 k-mer、同一阶段，所有方案与四域的选中样本共同拟合一次逐样本 L2 归一化后的 full-SVD PCA，方案间共享基与坐标范围。阶段之间、模型之间及单/双/三类图之间分别拟合；这些坐标轴不能直接当成同一空间。PCA 仅用于描述展示，不用于计算距离。

类内距离为每个域×k-mer 组中样本到单位质心的余弦距离均值；同类跨域距离为每类的六对域质心余弦距离；不同类距离为每域内选中 k-mer 的所有质心配对距离。趋势线分别对组或配对等权平均。单类没有类间距离，因此对应 CSV 单元格为空；类间/类内比是均值的比值。散点 X 的二维均值不等同于高维单位质心直接投影的位置。

这些图只使用明确选中的类及其现有固定样本，不代表全 1024 类总体，也不包含旧图中的全类参考曲线。低跨域距离必须结合不同类分离度与类内分散度判断，不能单独等同于更强识别能力或 basecall 准确率。V003 与 V006 的模型和输入标准化策略均不同。短 5-mer 方案若启用，代表本轮固定样本的短窗实验，不声称严格复现 08 的旧样本/旧阶段定义。

坐标与原 sample_index 见 [coordinates.csv](../analysis/coordinates.csv)；PCA 基、解释方差和样本数见 [pca_summary.csv](../analysis/pca_summary.csv)；逐组/逐配对统计及趋势均值在 analysis/。所有输入及绘图数据 SHA-256 记录于 [data-manifest.json](../analysis/data-manifest.json)，图文件校验见 [plot.validation.json](plot.validation.json)。坐标和数据清单先保存，再绘图；各方案使用相同的 case_index 与原始 sample_index。

""" + "\n".join(links) + "\n"
    (reports / "PLOTS.md").write_text(content)
    return validation


if __name__=='__main__':case_main()
