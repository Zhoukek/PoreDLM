#!/usr/bin/env python3
"""Extract and compare three readout strategies without modifying frozen runs."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import time

MODELS = ('V003', 'V006')
SCHEMES = ('chunk_5mer', 'short_5mer')
DOMAINS = ('cyclone-dna', 'ont-r10-hg002', 'dna-amplicon')
RUN_ID = '20260917_three_strategy'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def root_from_config(config):
    return Path(config['project_root'])


def model_run(config, model):
    return Path(config['model_outputs'][model])


def environment(project):
    return {**os.environ, 'PYTHONPATH': str(project/'script/NanoRepProbe/src')+':/tmp/model_eval/deps',
            'PYTHONDONTWRITEBYTECODE':'1', 'OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4',
            'MPLCONFIGDIR':'/tmp/model_eval/mpl_cache','HF_HOME':'/tmp/model_eval/hf_cache',
            'HF_MODULES_CACHE':'/tmp/model_eval/hf_cache/modules','HF_HUB_OFFLINE':'1',
            'TRANSFORMERS_OFFLINE':'1','TORCHDYNAMO_DISABLE':'1'}


def check_config(path):
    path=Path(path).resolve();config=json.loads(path.read_text())
    for file,sha in config['code_hashes'].items():
        if digest(file)!=sha: raise RuntimeError(f'Controller dependency changed: {file}')
    return config


def prepare(config_path):
    config=check_config(config_path);project=root_from_config(config)
    for model in MODELS:
        run=model_run(config,model)
        source=Path(config['models'][model]['source_run'])
        full=json.loads((source/'config/run.json').read_text())
        center=source.parent/'20260917_crossdomain_probe'
        if json.loads((center/'status.json').read_text())['status']!='complete':
            raise RuntimeError('Existing center probe is incomplete')
        base=json.loads((center/'config.resolved.json').read_text())
        for scheme in SCHEMES:
            cfg=json.loads(json.dumps(base))
            cfg['scheme']=scheme
            cfg['source_run']=str(run)
            cfg['phase_confidence']='low; Apple inherits Stone phase'
            cfg['comparison_protocol']='Same frozen samples, no extra signal normalization or clamp, continuous CNN/BERT/DLM'
            for item in cfg['domains']:
                item['representations']={stage:str(run/'representations'/scheme/item['name']/(stage+'.npy')) for stage in cfg['stages']}
            destination=run/'config'/f'probe.{scheme}.json'
            if destination.exists() and json.loads(destination.read_text())!=cfg:
                raise RuntimeError('Existing probe configuration differs')
            dump(destination,cfg)
        dump(run/'config/strategy_protocol.json',{
            'model':model,'model_id':full['model_id'],'normalization':full['strategy'],
            'strategies':['chunk_center',*SCHEMES],'domains':list(DOMAINS),
            'stages':['vqe_cnn','bert','dlm_ode'],'samples_per_domain':98164,
            'center_probe':str(center),'center_probe_snapshot_sha256':digest(center/'SHA256SUMS'),
            'center_reuse':'Reuse complete existing classification results; no refitting or resampling',
            'signal_policy':'Use frozen normalized signal values directly in every strategy, no extra clamp',
            'short_policy':'Crop [signal_start,signal_end), exact-length batches, mean ceil(L/5) content tokens',
            'whole_policy':'Full chunk, mean token centers in [signal_start,signal_end)',
            'center_policy':'Full chunk, mean token centers in [center_signal_start,center_signal_end)',
            'special_tokens':'BOS/EOS excluded from all readouts',
            'short_confounders':'Short input changes context, absolute token positions, convolution boundaries and stride phase; not an isolated attention ablation',
            'inference':full['ode'],'source_config_sha256':digest(source/'config/run.json'),
            'signal_preprocessing_comparison':'V003+Stone versus V006+Apple; changes both model and preprocessing',
            'legacy_short_difference':'V003 legacy selected-four-class short run adds clamp; this controlled full comparison does not',
        })
        (run/'logs').mkdir(parents=True,exist_ok=True)
    print('Prepared full frozen-sample comparison for both models',flush=True)


def run_all(args):
    config=check_config(args.config);project=root_from_config(config)
    env=environment(project);python=config['python']
    shared=Path(config['comparison_root'])
    dump(shared/'status.json',{'status':'extracting','started_unix':time.time()})
    jobs=queue.Queue()
    # Long full-chunk domains first, then the smaller domain and short windows.
    for domain in ['ont-r10-hg002','dna-amplicon','cyclone-dna']:
        for model in MODELS: jobs.put((model,'chunk_5mer',domain))
    for domain in DOMAINS:
        for model in MODELS: jobs.put((model,'short_5mer',domain))
    def gpu_worker(gpu):
        while True:
            try: model,scheme,domain=jobs.get_nowait()
            except queue.Empty: return
            target=model_run(config,model)
            final_folder=target/'representations'/scheme/domain
            scratch_root=Path(config['scratch_root'])/model
            worker_root=target/'representations' if (final_folder/'complete.json').exists() else scratch_root
            log=target/'logs'/f'extract.{scheme}.{domain}.log'
            command=[python,config['worker_path'],'--config',str(Path(args.config).resolve()),
                     '--scheme',scheme,'--model',model,'--dataset',domain,'--device','cuda:0',
                     '--output-root',str(worker_root)]
            print(f'GPU {gpu}: start {model}/{scheme}/{domain}',flush=True)
            with log.open('a') as handle:
                result=subprocess.run(command,env={**env,'CUDA_VISIBLE_DEVICES':str(gpu)},
                                      stdout=handle,stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'Extraction failed: {model}/{scheme}/{domain}; see {log}')
            complete=worker_root/scheme/domain/'complete.json'
            if not complete.is_file(): raise RuntimeError(f'Missing extraction completion: {complete}')
            if worker_root != target/'representations':
                pending=final_folder.with_name(final_folder.name+'.pending')
                if pending.exists() and (pending/'binding.json').exists():
                    if (pending/'binding.json').read_bytes() != (complete.parent/'binding.json').read_bytes():
                        raise RuntimeError('Interrupted output publication belongs to different inputs')
                shutil.copytree(complete.parent,pending,dirs_exist_ok=True)
                metadata=json.loads(complete.read_text())
                for stage,expected in metadata['sha256'].items():
                    if digest(pending/(stage+'.npy'))!=expected:raise RuntimeError('Copied representation fingerprint differs')
                if final_folder.exists():raise RuntimeError('Refuse to overwrite existing published extraction')
                pending.rename(final_folder)
            print(f'GPU {gpu}: complete {model}/{scheme}/{domain}',flush=True)
    try:
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            futures=[pool.submit(gpu_worker,gpu) for gpu in args.gpus]
            for future in futures: future.result()
        dump(shared/'status.json',{'status':'probing','extraction_jobs_complete':12})
        probe_jobs=[(model,scheme) for model in MODELS for scheme in SCHEMES]
        def probe_worker(job):
            model,scheme=job;run=model_run(config,model);output=run/'probes'/scheme
            command=[python,'-m','nanorepprobe','run','--config',str(run/'config'/f'probe.{scheme}.json'),
                     '--output',str(output),'--threads',str(args.cpu_threads)]
            if (output/'identity.json').exists(): command.append('--resume')
            log=run/'logs'/f'probe.{scheme}.log'
            print(f'Probe start {model}/{scheme}',flush=True)
            with log.open('a') as handle:
                result=subprocess.run(command,env=env,stdout=handle,stderr=subprocess.STDOUT)
            if result.returncode: raise RuntimeError(f'Probe failed; see {log}')
            status_path=output/'status.json'
            if not status_path.exists() or json.loads(status_path.read_text()).get('status')!='complete':
                raise RuntimeError(f'Probe did not record completion: {output}')
            print(f'Probe complete {model}/{scheme}',flush=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(probe_worker,probe_jobs))
        dump(shared/'status.json',{'status':'awaiting_independent_validation','extraction_jobs_complete':12,
                                   'new_probe_runs_complete':4,'new_metric_rows':432,'reused_center_metric_rows':216})
    except BaseException as exc:
        dump(shared/'status.json',{'status':'interrupted_or_failed','message':str(exc)})
        raise


def status(config_path):
    config=check_config(config_path)
    rows=[]
    for model in MODELS:
        for scheme in SCHEMES:
            for domain in DOMAINS:
                folder=model_run(config,model)/'representations'/scheme/domain
                completion=folder/'complete.json';progress=folder/'progress.json'
                if not progress.exists():progress=Path(config['scratch_root'])/model/scheme/domain/'progress.json'
                rows.append({'model':model,'scheme':scheme,'domain':domain,
                    'complete':completion.exists(),
                    'progress':json.loads(progress.read_text()) if progress.exists() else None})
    print(json.dumps(rows,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare');sub.add_parser('status')
    run=sub.add_parser('run');run.add_argument('--gpus',nargs='+',type=int,default=[0,1,2,3]);run.add_argument('--cpu-threads',type=int,default=4)
    args=p.parse_args()
    if args.command=='prepare':prepare(args.config)
    elif args.command=='status':status(args.config)
    elif args.command=='run':
        if len(set(args.gpus))!=len(args.gpus):raise ValueError('GPU IDs must be distinct')
        run_all(args)

if __name__=='__main__':main()
