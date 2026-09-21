"""Inference helpers for frozen normalized full chunks, without re-normalization.

V003/V006 PoreVQCodec geometry: 6000 samples -> 1200 tokens, sample center
of token j is 5*j, receptive field [5*j-13, 5*j+14). All source intervals
are half open. CNN uses zero padding at chunk boundaries.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

STRIDE = 5
CENTER_OFFSET = 0
RECEPTIVE_FIELD = 27
CODE_ID_OFFSET = 128


def token_count(signal_length: int) -> int:
    if signal_length <= 0:
        raise ValueError("Signal length must be positive")
    return (signal_length + STRIDE - 1) // STRIDE


def pooling_bounds(starts, ends, signal_length=6000, prefix_tokens=0):
    """Token bounds selecting frame centers inside [start,end), with no fallback.

    Pass center_signal_start/end to pool a center-base representation labelled
    by its local 5-mer; pass signal_start/end only if intentionally pooling the
    entire five-base segment. A DLM BOS prefix shifts both bounds by one.
    """
    starts_raw, ends_raw = np.asarray(starts), np.asarray(ends)
    starts, ends = starts_raw.astype(np.int64), ends_raw.astype(np.int64)
    if not np.array_equal(starts, starts_raw) or not np.array_equal(ends, ends_raw):
        raise ValueError("Source boundaries must be integral")
    if starts.shape != ends.shape or np.any(starts < 0) or np.any(ends > signal_length) or np.any(ends <= starts):
        raise ValueError("Invalid source signal interval")
    if prefix_tokens < 0:
        raise ValueError("Prefix tokens must be nonnegative")
    left = (starts - CENTER_OFFSET + STRIDE - 1) // STRIDE
    right = (ends - CENTER_OFFSET + STRIDE - 1) // STRIDE
    left = np.clip(left, 0, token_count(signal_length))
    right = np.clip(right, 0, token_count(signal_length))
    if np.any(right <= left):
        raise ValueError("An interval contains no CNN token center; do not silently change its pooling rule")
    return left + prefix_tokens, right + prefix_tokens


def load_codec(encoder_dir, device='cuda'):
    """Load local custom class explicitly: checkpoint config has no auto_map.

    No model preprocessing is invoked: caller supplies already normalized
    frozen Stone/Apple signal arrays from the corpus.
    """
    import torch
    encoder_dir = Path(encoder_dir)
    model_source = encoder_dir / 'modeling_pore_vq_codec.py'
    spec = importlib.util.spec_from_file_location('model_eval_pore_vq_codec', model_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    codec = module.PoreVQCodec.from_pretrained(str(encoder_dir), local_files_only=True, torch_dtype=torch.float32)
    codec.eval().requires_grad_(False).to(device)
    if (codec.cnn_stride, codec.RF, codec.codebook_dim) != (5, 27, 768):
        raise ValueError("This adapter only supports the audited stride=5/RF=27/width=768 codec")
    return codec


def cnn_features(codec, signals):
    """B x L normalized signals -> B x ceil(L/5) x 768 float32 features."""
    import torch
    device = next(codec.parameters()).device
    x = torch.as_tensor(signals, dtype=torch.float32, device=device)
    if x.ndim != 2 or not torch.isfinite(x).all():
        raise ValueError("Expected finite normalized signals with shape (batch, samples)")
    if codec.training:
        raise ValueError("Codec must be in eval mode")
    with torch.inference_mode():
        z = codec.cnn_model.encode(x.unsqueeze(1)).transpose(1, 2).contiguous()
    if z.shape[1] != token_count(x.shape[1]):
        raise ValueError("Unexpected CNN token geometry")
    return z


def quantize_features(codec, features, tile_tokens=256):
    """Official VQ eval in independent token tiles; no decoder or EMA updates.

    Return B x T x D quantized vectors and B x T code IDs. Tile size bounds the
    costly token-by-65536-code distance/onehot allocations. This is valid for
    these codecs: a single codebook, identity projection, no affine adaptation.
    """
    import torch
    if codec.training or codec.vq.training:
        raise ValueError("Quantization must run in eval mode")
    if tile_tokens < 1 or features.ndim != 3:
        raise ValueError("Expected B x T x D features and positive tile size")
    book = codec.vq._codebook
    if bool(getattr(book, 'affine_param', False)):
        raise ValueError("Tiled VQ is not supported for affine-adapted codebooks")
    if hasattr(book, 'initted') and not bool(book.initted.all()):
        raise ValueError("Checkpoint codebook is not initialized")
    shape = features.shape
    flat = features.reshape(1, -1, shape[-1])
    quantized, indices = [], []
    with torch.inference_mode(), torch.autocast(device_type=features.device.type, enabled=False):
        for start in range(0, flat.shape[1], tile_tokens):
            values, codes, _, _ = codec.vq(flat[:, start:start+tile_tokens].float(), return_loss_breakdown=True)
            quantized.append(values)
            indices.append(codes)
        zq = torch.cat(quantized, dim=1).reshape(shape)
        ids = torch.cat(indices, dim=1).reshape(shape[:2]).to(dtype=torch.long)
    return zq, ids


def mean_pool_intervals(features, batch_rows, left, right):
    """Pool selected intervals from B x T x D; preserve supplied manifest order.

    Bounds must already include any special-token prefix. Accumulate in fp32
    even if the DLM supplied fp16/bf16 representations.
    """
    import torch
    device = features.device
    batch_rows = torch.as_tensor(batch_rows, dtype=torch.long, device=device)
    left = torch.as_tensor(left, dtype=torch.long, device=device)
    right = torch.as_tensor(right, dtype=torch.long, device=device)
    if batch_rows.ndim != 1 or batch_rows.shape != left.shape or left.shape != right.shape:
        raise ValueError("Pooling indices must be parallel vectors")
    if torch.any((batch_rows < 0) | (batch_rows >= features.shape[0])) or torch.any((left < 0) | (right > features.shape[1]) | (right <= left)):
        raise ValueError("Pooling interval outside the feature sequence")
    with torch.inference_mode():
        sums = torch.nn.functional.pad(features.float().cumsum(dim=1), (0, 0, 1, 0))
        return (sums[batch_rows, right] - sums[batch_rows, left]) / (right-left).unsqueeze(-1)

def mean_pool_intervals_fp64(features, batch_rows, left, right):
    """Return N x D means in supplied manifest order from B x T x D features.

    Select [left,right) for each associated batch row. No prefix copy/padding
    is allocated: use prefix[right-1] minus prefix[left-1] (zero when left=0).
    Peak scratch is approximately B*T*D*8 bytes plus three N*D FP64 gathers.
    No CPU loops or per-interval GPU reduction kernels are used.
    """
    import torch
    if features.ndim != 3 or features.dtype != torch.float32:
        raise ValueError('Expected B x T x D float32 model features')
    device = features.device
    rows = torch.as_tensor(batch_rows, dtype=torch.long, device=device)
    starts = torch.as_tensor(left, dtype=torch.long, device=device)
    stops = torch.as_tensor(right, dtype=torch.long, device=device)
    if rows.ndim != 1 or rows.shape != starts.shape or starts.shape != stops.shape:
        raise ValueError('Pooling indices must be parallel vectors')
    if rows.numel() == 0:
        return features.new_empty((0, features.shape[-1]))
    if torch.any((rows < 0) | (rows >= features.shape[0])) or torch.any((starts < 0) | (stops > features.shape[1]) | (stops <= starts)):
        raise ValueError('Pooling interval outside the feature sequence')
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        prefix = torch.cumsum(features, dim=1, dtype=torch.float64)
        totals = prefix[rows, stops - 1]
        previous = prefix[rows, (starts - 1).clamp_min(0)]
        previous.masked_fill_(starts[:, None] == 0, 0)
        totals.sub_(previous)
        totals.div_((stops - starts).unsqueeze(-1))
        return totals.to(dtype=torch.float32)


import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.metadata import version, PackageNotFoundError

EVAL_ROOT=Path(os.environ.get('NANOEVAL_ROOT',Path(__file__).resolve().parent))
PROJECT=EVAL_ROOT.parent.parent
NAMES=('cyclone-dna','ont-r10-hg002','dna-amplicon','cyclone-rna')
STAGES=('vqe_cnn','bert','dlm_ode')
RUN_NAME='20260916_full5mer'
MODEL_ROOT=Path('/mnt/zzbnew/poregpt/models')

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as handle:
  for block in iter(lambda:handle.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def dump(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temporary=path.with_name(path.name+'.partial')
 temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 os.replace(temporary,path)

def model_paths(model):
 name=f'HF_VQE768C08A001_DNADLLM_{model}'
 strategy={'V003':'stone','V006':'apple'}[model]
 return name,strategy,MODEL_ROOT/name,EVAL_ROOT/name/'runs'/RUN_NAME

def settings(model):
 name,strategy,model_dir,run=model_paths(model)
 corpus=PROJECT/'01.capability/00.corpus'/strategy
 frozen={line.split('  ',1)[1]:line.split('  ',1)[0] for line in (corpus/'SHA256SUMS').read_text().splitlines()}
 protocol={'schema_version':1,'model_id':name,'strategy':strategy,'corpus_root':str(corpus),
 'encoder_dir':str(model_dir/'encoder'),'dlm_dir':str(model_dir/'hf_dlm'),
 'seed':20260916,'stages':list(STAGES),'signal_normalization':'none; frozen corpus values used directly',
 'pooling':{'region':'center base of exact 5-mer','method':'FP64 prefix-sum interval mean of token centers inside [start,end)',
 'token_center':'5*j','left':'ceil(center_signal_start/5)','right':'ceil(center_signal_end/5)',
 'stride':5,'receptive_field':27,'special_tokens_removed':True,'empty_interval_policy':'error','accumulation':'float64; output float32'},
 'precision':'float32','tf32':False,'batch_size':32,'vq_tile_tokens':1024,
 'tokenization':{'code_id_offset':128,'bos':2,'eos':3,'pad':1},
 'ode':{'steps':1,'start_t':0.97,'self_cond_cfg_scale':0.5},
 'corpus_checksums_sha256':digest(corpus/'SHA256SUMS'),
 'runner_sha256':digest(Path(__file__)),
 'checkpoint_fingerprints':{},'datasets':{}}
 for package in ['encoder','hf_dlm']:
  folder=model_dir/package
  files=[folder/'model.safetensors',folder/'config.json',folder/('modeling_pore_vq_codec.py' if package=='encoder' else 'modeling_poredlm.py')]
  if package=='hf_dlm':files+=sorted((folder/'ELF-pytorch-port/src/torch_elf').rglob('*.py'))
  for path in files:protocol['checkpoint_fingerprints'][str(path)]=digest(path)
 for name in NAMES:
  protocol['datasets'][name]={rel:digest(corpus/name/rel) for rel in ['manifest.csv','prep/signals.npy','aligned/chunks.jsonl','aligned/calibration.json']}
  for rel,actual in protocol['datasets'][name].items():
   if frozen.get(name+'/'+rel)!=actual:raise RuntimeError('Input differs from the frozen corpus checksum')
 protocol['nanorepdist_config_sha256']=digest(run/'config/nanorepdist.yaml')
 return protocol

def register():
 entries={}
 for model in ['V003','V006']:
  name,strategy,model_dir,run=model_paths(model);run.mkdir(parents=True,exist_ok=True)
  configuration=settings(model);path=run/'config/run.json'
  if path.exists() and json.loads(path.read_text())!=configuration:
   raise RuntimeError(f'Existing run differs: {path}; use a new run name')
  dump(path,configuration)
  entries[model]={'model_id':name,'strategy':strategy,'model_root':str(model_dir),
   'encoder':str(model_dir/'encoder'),'hf_dlm':str(model_dir/'hf_dlm'),
   'run':str(run.relative_to(EVAL_ROOT)),'status_file':str((run/'reports/status.json').relative_to(EVAL_ROOT))}
  if not (run/'reports/status.json').exists():dump(run/'reports/status.json',{'status':'registered','representations':'pending','nanorepdist':'pending'})
  print(model,'registered:',run,flush=True)
 dump(EVAL_ROOT/'models.json',{'schema_version':1,'models':entries})

def load_run(model):
 name,strategy,model_dir,run=model_paths(model)
 config=json.loads((run/'config/run.json').read_text())
 if config['runner_sha256']!=digest(Path(__file__)):raise RuntimeError('Runner changed since run registration')
 return config,run

def verify_sources(config,dataset=None):
 corpus=Path(config['corpus_root'])
 if digest(corpus/'SHA256SUMS')!=config['corpus_checksums_sha256']:raise RuntimeError('Frozen corpus snapshot changed')
 for name in ([dataset] if dataset else NAMES):
  for relative,expected in config['datasets'][name].items():
   if digest(corpus/name/relative)!=expected:raise RuntimeError('Corpus fingerprint changed')
 for path,expected in config['checkpoint_fingerprints'].items():
  if digest(path)!=expected:raise RuntimeError('Model checkpoint or source changed')


def setup_torch(device):
 import torch
 torch.set_num_threads(2)
 torch.manual_seed(20260916)
 np.random.seed(20260916)
 torch.backends.cuda.matmul.allow_tf32=False
 torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False
 torch.backends.cudnn.deterministic=True
 torch.set_float32_matmul_precision('highest')
 if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; run on GPU-visible host')
 return torch

def load_models(config,device):
 import torch
 from safetensors.torch import load_model
 from transformers import AutoConfig
 from transformers.dynamic_module_utils import get_class_from_dynamic_module
 encoder_dir=Path(config['encoder_dir'])
 spec=importlib.util.spec_from_file_location('eval_pore_codec',encoder_dir/'modeling_pore_vq_codec.py')
 module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 codec=module.PoreVQCodec(module.PoreVQCodecConfig.from_pretrained(str(encoder_dir),local_files_only=True))
 load_model(codec,str(encoder_dir/'model.safetensors'),strict=True,device='cpu')
 codec.to(device).eval().requires_grad_(False)
 if (codec.cnn_stride,codec.RF,codec.codebook_dim)!=(5,27,768):raise RuntimeError('Unsupported codec geometry')
 model_dir=Path(config['dlm_dir'])
 hf=AutoConfig.from_pretrained(str(model_dir),trust_remote_code=True,local_files_only=True)
 hf.elf_src_path=str(model_dir/'ELF-pytorch-port/src')
 cls=get_class_from_dynamic_module('modeling_poredlm.PoreDLMForDiffusion',str(model_dir),local_files_only=True)
 dlm=cls(hf);load_model(dlm,str(model_dir/'model.safetensors'),strict=True,device='cpu')
 dlm.to(device).eval().requires_grad_(False)
 return codec,dlm

def forward_maps(codec,dlm,signals,config):
 import torch
 with torch.inference_mode():
  cnn=cnn_features(codec,signals)
  _,codes=quantize_features(codec,cnn,tile_tokens=config['vq_tile_tokens'])
  if codes.min()<0 or codes.max()>=65536:raise RuntimeError('Invalid codec index')
  ids=torch.cat([codes.new_full((len(codes),1),2),codes+128,codes.new_full((len(codes),1),3)],dim=1)
  if ids.shape[1]>dlm.context_encoder.config.max_position_embeddings:raise RuntimeError('Context length exceeded')
  mask=torch.ones_like(ids)
  bert=dlm.context_encoder(input_ids=ids,attention_mask=mask,return_dict=True).last_hidden_state
  ode=dlm.ode_from_context_hidden(bert,attention_mask=mask,ode_steps=config['ode']['steps'],
   ode_start_t=config['ode']['start_t'],self_cond_cfg_scale=config['ode']['self_cond_cfg_scale'])
  return {'vqe_cnn':cnn,'bert':bert[:,1:-1],'dlm_ode':ode[:,1:-1]},codes

def read_manifest(config,name):
 corpus=Path(config['corpus_root']);signal=corpus/name/'prep/signals.npy'
 groups=defaultdict(list);sample_rows=[]
 with (corpus/name/'manifest.csv').open() as handle:
  for row in csv.DictReader(handle):
   idx=int(row['sample_index']);sample_rows.append(idx)
   if (corpus/row['signal_path']).resolve()!=signal or int(row['signal_ref_start'])!=0 or int(row['signal_length'])!=6000 or int(row['signal_ref_end'])!=6000:
    raise RuntimeError('Expected complete normalized 6000-point chunks')
   left,right=pooling_bounds([int(row['center_signal_start'])],[int(row['center_signal_end'])])
   groups[int(row['signal_row'])].append((idx,int(left[0]),int(right[0])))
 if sample_rows!=list(range(len(sample_rows))) or len(sample_rows)!=98164:raise RuntimeError('Manifest row order or membership changed')
 return signal,groups,len(sample_rows)

def pooled_batch(codec,dlm,signals,chunk_rows,groups,config):
 import torch
 maps,codes=forward_maps(codec,dlm,signals,config)
 samples=[];batch_rows=[];left=[];right=[]
 for batch_idx,row in enumerate(chunk_rows):
  for sample_idx,start,end in groups[row]:
   samples.append(sample_idx);batch_rows.append(batch_idx);left.append(start);right.append(end)
 result={}
 for stage,features in maps.items():
  if features.shape[1:]!=(1200,768):raise RuntimeError('Stage token geometry differs')
  pooled=mean_pool_intervals_fp64(features,batch_rows,left,right)
  values=pooled.cpu().numpy().astype(np.float32,copy=False)
  if not np.isfinite(values).all() or np.any(np.linalg.norm(values,axis=1)==0):raise RuntimeError('Invalid pooled representation')
  result[stage]=values
 return samples,result,codes

def benchmark(args):
 config,run=load_run(args.model);torch=setup_torch(args.device)
 signal,groups,total=read_manifest(config,args.dataset)
 rows=sorted(groups)[:config['batch_size']];array=np.load(signal,mmap_mode='r');values=np.array(array[rows],dtype=np.float32)
 codec,dlm=load_models(config,args.device)
 t=time.monotonic();samples,result,codes=pooled_batch(codec,dlm,values,rows,groups,config)
 torch.cuda.synchronize();elapsed=time.monotonic()-t
 # One real complete chunk: prove repeated maps and tiled VQ preserve identities.
 one=np.array(array[rows[:1]],dtype=np.float32)
 a,ids1=forward_maps(codec,dlm,one,config);b,ids2=forward_maps(codec,dlm,one,config)
 assert torch.equal(ids1,ids2)
 assert torch.equal(codes[:1],ids1), 'Batch vs single code IDs changed'
 diffs={stage:float((a[stage]-b[stage]).abs().max()) for stage in STAGES}
 assert all(x==0 for x in diffs.values()),diffs
 single_samples,single_result,_=pooled_batch(codec,dlm,one,rows[:1],groups,config)
 batch_diff={stage:float(np.max(np.abs(single_result[stage]-result[stage][:len(single_samples)]))) for stage in STAGES}
 assert all(x<=1e-4 for x in batch_diff.values()),batch_diff
 report={'status':'passed','batch_vs_single_code_ids_identical':True,'model':args.model,'dataset':args.dataset,'batch_chunks':len(rows),'batch_samples':len(samples),
 'seconds':elapsed,'repeat_max_abs_diff':diffs,'batch_vs_single_max_abs_diff':batch_diff,
 'gpu':torch.cuda.get_device_name(),'gpu_peak_bytes':torch.cuda.max_memory_allocated(),
 'stages':{k:list(v.shape) for k,v in result.items()}}
 dump(run/'reports/smoke.validation.json',report);print(json.dumps(report,indent=2),flush=True)

def extract(args):
 config,run=load_run(args.model);torch=setup_torch(args.device)
 corpus=Path(config['corpus_root']);target=run/'representations'/args.dataset;target.mkdir(parents=True,exist_ok=True)
 done_path=target/'complete.json'
 verify_sources(config,args.dataset)
 config_sha=digest(run/'config/run.json')
 if done_path.exists():
  done=json.loads(done_path.read_text())
  if done.get('run_config_sha256')!=config_sha:raise RuntimeError('Completed dataset uses different run parameters')
  for stage,expected in done['sha256'].items():
   if digest(target/(stage+'.npy'))!=expected:raise RuntimeError('Completed representation changed')
  print(args.model,args.dataset,'already complete',flush=True);return
 signal,groups,total=read_manifest(config,args.dataset);source=np.load(signal,mmap_mode='r')
 arrays={};paths={}
 for stage in STAGES:
  final=target/(stage+'.npy');path=final if final.exists() else target/(stage+'.npy.partial');paths[stage]=path
  existing=path.exists();arrays[stage]=np.lib.format.open_memmap(path,mode='r+' if existing else 'w+',dtype=np.float32,shape=(total,768))
  if not existing:arrays[stage][:]=np.nan;arrays[stage].flush()
  if arrays[stage].shape!=(total,768) or arrays[stage].dtype!=np.float32:raise RuntimeError('Incompatible output array')
 journal=target/'progress.jsonl';committed=set()
 if journal.exists():
  with journal.open('r+b') as handle:
   while True:
    start=handle.tell();line=handle.readline()
    if not line:break
    if not line.endswith(b'\n'):handle.truncate(start);break
    record=json.loads(line);rows=record['signal_rows'];sample_indices=record['sample_indices']
    if record.get('run_config_sha256')!=config_sha:raise RuntimeError('Resume run configuration changed')
    if committed.intersection(rows) or any(row not in groups for row in rows):raise RuntimeError('Invalid resume journal rows')
    if sample_indices!=[sample[0] for row in rows for sample in groups[row]]:raise RuntimeError('Resume sample membership differs')
    for stage in STAGES:
     if hashlib.sha256(np.asarray(arrays[stage][sample_indices]).tobytes()).hexdigest()!=record['sha256'][stage]:raise RuntimeError('Uncommitted/corrupted representation')
    committed.update(rows)
 remaining=[row for row in sorted(groups) if row not in committed]
 print(args.model,args.dataset,'resume',len(committed),'/',len(groups),'chunks',flush=True)
 codec,dlm=load_models(config,args.device)
 start=time.monotonic();batch_size=config['batch_size']
 with journal.open('a') as handle:
  for offset in range(0,len(remaining),batch_size):
   rows=remaining[offset:offset+batch_size];signals=np.array(source[rows],dtype=np.float32)
   samples,result,_=pooled_batch(codec,dlm,signals,rows,groups,config)
   fingerprints={}
   for stage,values in result.items():
    arrays[stage][samples]=values;arrays[stage].flush()
    fingerprints[stage]=hashlib.sha256(values.tobytes()).hexdigest()
   handle.write(json.dumps({'signal_rows':rows,'sample_indices':samples,'sha256':fingerprints,'run_config_sha256':config_sha},separators=(',',':'))+'\n')
   handle.flush();os.fsync(handle.fileno());committed.update(rows)
   if offset%(batch_size*10)==0 or len(committed)==len(groups):
    elapsed=time.monotonic()-start;rate=(offset+len(rows))/max(elapsed,1e-6)
    print(f'{args.model}/{args.dataset}: {len(committed)}/{len(groups)} chunks, {elapsed:.1f}s, {rate:.2f} chunk/s',flush=True)
 if len(committed)!=len(groups):raise RuntimeError('Incomplete chunk membership')
 for stage in STAGES:
  values=arrays[stage]
  if not np.isfinite(values).all() or np.any(np.linalg.norm(values,axis=1)==0):raise RuntimeError('Missing/invalid sample rows')
  values.flush()
 arrays.clear()
 for stage in STAGES:
  final=target/(stage+'.npy')
  if paths[stage]!=final:os.replace(paths[stage],final)
 output={'status':'passed','samples':total,'chunks':len(groups),'dimensions':768,'dtype':'float32',
  'runner_sha256':config['runner_sha256'],'run_config_sha256':config_sha,'signal_normalization':config['signal_normalization'],
  'sha256':{stage:digest(target/(stage+'.npy')) for stage in STAGES},
  'gpu':torch.cuda.get_device_name(),'seconds':time.monotonic()-start}
 dump(done_path,output);print(json.dumps(output,indent=2),flush=True)

def analyze(args):
 config,run=load_run(args.model)
 verify_sources(config)
 config_sha=digest(run/'config/run.json')
 if digest(run/'config/nanorepdist.yaml')!=config['nanorepdist_config_sha256']:raise RuntimeError('Analysis protocol changed')
 for name in NAMES:
  p=run/'representations'/name/'complete.json'
  if not p.exists() or json.loads(p.read_text())['status']!='passed':raise RuntimeError('Representation extraction incomplete')
  complete=json.loads(p.read_text())
  if complete.get('run_config_sha256')!=config_sha:raise RuntimeError('Representation run configuration differs')
  for stage,expected in complete['sha256'].items():
   if digest(p.parent/(stage+'.npy'))!=expected:raise RuntimeError('Representation output changed')
 sys.path.insert(0,str(PROJECT/'script/NanoRepDist/src'))
 from nanorepdist.config import load_config
 from nanorepdist.analysis import run_analysis
 from nanorepdist.plotting import run_plotting
 from nanorepdist.reporting import run_report
 from nanorepdist.validation import verify_outputs
 import nanorepdist.analysis as analysis_module
 print(args.model,'loading all-1024-kmer configuration',flush=True)
 cfg=load_config(run/'config/nanorepdist.yaml')
 original=analysis_module.compute_stage_metrics
 def measured(*values,**kwargs):
  stage=values[0].stage;t=time.monotonic();print('metrics start',stage,flush=True)
  result=original(*values,**kwargs);print('metrics complete',stage,f'{time.monotonic()-t:.1f}s',flush=True);return result
 analysis_module.compute_stage_metrics=measured
 print('Analyze all 392656 samples / 3 stages',flush=True);run_analysis(cfg)
 print('Generate diagnostic figures',flush=True);run_plotting(cfg)
 print('Generate report',flush=True);run_report(cfg)
 report=verify_outputs(cfg)
 if report['status']!='pass':raise RuntimeError('NanoRepDist output validation failed')
 dump(run/'reports/nanorepdist.validation.json',report)
 environment={'python':sys.version,'executable':sys.executable,'packages':{}}
 for package in ['numpy','pandas','scipy','scikit-learn','matplotlib','PyYAML','torch','transformers','safetensors','vector-quantize-pytorch']:
  try:environment['packages'][package]=version(package)
  except PackageNotFoundError:environment['packages'][package]=None
 dump(run/'provenance/environment.json',environment)
 dump(run/'reports/status.json',{'status':'completed','representations':'passed','nanorepdist':'passed',
  'samples':392656,'kmers':1024,'stages':list(STAGES),'normalization':config['strategy']})
 import tarfile
 with tarfile.open(run/'provenance/evaluation_sources.tar.gz','w:gz') as archive:
  archive.add(Path(__file__),arcname='evaluate.py')
  for path in (PROJECT/'script/NanoRepDist/src/nanorepdist').glob('*.py'):archive.add(path,arcname='nanorepdist/'+path.name)
 paths=sorted(p for p in run.rglob('*') if p.is_file() and p.name!='SHA256SUMS' and p.suffix!='.log' and not p.name.endswith('.partial'))
 (run/'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.relative_to(run)}\n' for p in paths))
 print(args.model,'FULL EVALUATION PASSED',flush=True)

def run_all(args):
 import queue
 jobs=queue.Queue()
 # Long domains first; the smaller Cyclone DNA job fills the fourth GPU.
 for model in ['V003','V006']:
  for name in ['ont-r10-hg002','cyclone-rna','dna-amplicon','cyclone-dna']:jobs.put((model,name))
 def worker(gpu):
  while True:
   try:model,name=jobs.get_nowait()
   except queue.Empty:return
   _,run=load_run(model)
   log=run/'logs'/f'extract.{name}.log'
   env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
   with log.open('a') as handle:
    result=subprocess.run([sys.executable,str(Path(__file__)),'extract','--model',model,'--dataset',name,'--device','cuda:0'],stdout=handle,stderr=subprocess.STDOUT,env=env)
   if result.returncode:raise RuntimeError(f'{model}/{name} extraction failed; {log}')
   print('extraction completed',model,name,'GPU',gpu,flush=True)
 pending_analysis={};finished_models=set()
 with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
  workers=[executor.submit(worker,gpu) for gpu in args.gpus]
  while not all(f.done() for f in workers) or len(pending_analysis)<2:
   for model in ['V003','V006']:
    _,run=load_run(model)
    if model not in pending_analysis and all((run/'representations'/name/'complete.json').exists() for name in NAMES):
     handle=(run/'logs/nanorepdist.log').open('a')
     env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['OMP_NUM_THREADS']='4';env['OPENBLAS_NUM_THREADS']='4';env['MKL_NUM_THREADS']='4'
     process=subprocess.Popen([sys.executable,str(Path(__file__)),'analyze','--model',model],stdout=handle,stderr=subprocess.STDOUT,env=env)
     pending_analysis[model]=(process,handle);print('NanoRepDist started',model,flush=True)
   for f in workers:
    if f.done():f.result()
   for model,(process,handle) in pending_analysis.items():
    if process.poll() is not None and process.returncode:raise RuntimeError(f'{model} NanoRepDist failed')
   time.sleep(5)
  for f in workers:f.result()
 for model,(process,handle) in pending_analysis.items():
  if process.wait():raise RuntimeError(f'{model} NanoRepDist failed')
  handle.close()
 print('Both model evaluations completed',flush=True)

def main():
 parser=argparse.ArgumentParser(description='Paired fixed-corpus model representations and NanoRepDist evaluation')
 subs=parser.add_subparsers(dest='command',required=True)
 subs.add_parser('register')
 for command in ['benchmark','extract']:
  sub=subs.add_parser(command);sub.add_argument('--model',choices=['V003','V006'],required=True)
  sub.add_argument('--dataset',choices=NAMES,default='cyclone-dna');sub.add_argument('--device',default='cuda:0')
 sub=subs.add_parser('analyze');sub.add_argument('--model',choices=['V003','V006'],required=True)
 sub=subs.add_parser('run');sub.add_argument('--gpus',type=int,nargs='+',default=[0,1,2,3])
 args=parser.parse_args()
 if args.command=='register':register()
 elif args.command=='benchmark':benchmark(args)
 elif args.command=='extract':extract(args)
 elif args.command=='analyze':analyze(args)
 else:run_all(args)

if __name__=='__main__':main()
