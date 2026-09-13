"""Short compiled MPS profile; host spans are not exclusive GPU kernel time."""
from pathlib import Path
import sys, os, json, time, argparse, statistics, functools, types
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'outputs/apple-gpu-v2'))
import experiment as exp
import torch, numpy as np
from torch._inductor.codecache import PyCodeCache
from fast_audiovae.gpu import GPUDecoder, _latents, _policy

ACTIVE=[];RECORDS=[]
def wrap(fn,name):
 def run(*args,**kwargs):
  if not ACTIVE:return fn(*args,**kwargs)
  start=time.perf_counter_ns()
  try:return fn(*args,**kwargs)
  finally:
   row=dict(name=name,host_ns=time.perf_counter_ns()-start,packet=ACTIVE[0])
   if name=='convolution':
    row.update(input=list(args[0].shape),weight=list(args[1].shape),groups=kwargs.get('groups'),transposed=kwargs.get('transposed'),dilation=kwargs.get('dilation'))
   RECORDS.append(row)
 return run

def main(output):
 assert not output.exists()
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 assert torch.backends.mps.is_available()
 model=exp.build('before_compile');decoder=GPUDecoder(model,torch,{'experiment':'profile'},'streaming')
 crops=exp.cases();z=crops['English']
 with torch.inference_mode():
  for length in (1,2):
   with decoder.stream() as s:
    for i in range(4):s.decode_chunk(z[...,i*length:(i+1)*length].copy())
 modules=[]
 for mod in PyCodeCache.modules:
  if hasattr(mod,'extern_kernels') and any(k.startswith('generated_kernel_') for k in vars(mod)):
   original=mod.extern_kernels
   mod.extern_kernels=types.SimpleNamespace(convolution=wrap(original.convolution,'convolution'))
   names=[k for k in vars(mod) if k.startswith('generated_kernel_')]
   for name in names:setattr(mod,name,wrap(getattr(mod,name),name))
   modules.append(dict(path=str(mod.__file__),generated=len(names)))
 rows=[]
 with torch.inference_mode(),torch.autocast('mps',enabled=False):
  for profiled in (False,True,False):
   for length in (1,2):
    states={};torch.mps.synchronize()
    for pos in range(0,12,length):
     key=f'{len(rows)}';timer={};begin=time.perf_counter_ns()
     _policy();owned=_latents(z[...,pos:pos+length]);now=time.perf_counter_ns();timer['input_validation_copy_ns']=now-begin
     latent=torch.from_numpy(owned).to('mps',dtype=torch.float32,non_blocking=False);t=time.perf_counter_ns();timer['upload_ns']=t-now
     if profiled:ACTIVE.append(key)
     audio,next_state=model.decode(latent,dict(states))
     if profiled:ACTIVE.clear()
     now=time.perf_counter_ns();timer['model_host_ns']=now-t
     assert audio.dtype==torch.float32 and audio.device.type=='mps' and tuple(audio.shape)==(1,1,length*1920)
     assert set(next_state)==set(model.state_shapes)
     for name,shape in model.state_shapes.items():
      value=next_state[name];assert value.dtype==torch.float32 and value.device.type=='mps' and tuple(value.shape)==shape
     t=time.perf_counter_ns();timer['metadata_ns']=t-now
     checked=torch.cat([v.reshape(-1) for v in (audio,*next_state.values())])
     assert torch.isfinite(checked).all().item()
     now=time.perf_counter_ns();timer['finite_check_and_wait_ns']=now-t
     ready=audio.detach().to('cpu',non_blocking=False);torch.mps.synchronize();result=ready.numpy().copy()
     t=time.perf_counter_ns();timer['output_copy_sync_ns']=t-now;timer['total_ns']=t-begin
     states=next_state
     rows.append(dict(packet=key,packet_ms=40*length,profiled=profiled,position=pos,timer=timer,output_sha=__import__('hashlib').sha256(result.tobytes()).hexdigest()))
 # Stage names follow actual topological conv order; store only shapes and durations.
 for row in rows:
  records=[r for r in RECORDS if r['packet']==row['packet']]
  if row['profiled']:
   conv=[r for r in records if r['name']=='convolution'];assert len(conv)==45
   for index,r in enumerate(conv):r['ordinal']=index
   assert len(records) in (128,129),len(records)
 summary=[]
 for profiled in (False,True):
  for ms in (40,80):
   values=[r for r in rows if r['packet_ms']==ms and r['profiled']==profiled and r['position']>0]
   summary.append(dict(packet_ms=ms,profiled=profiled,packets=len(values),mean_ms={k:statistics.mean(r['timer'][k] for r in values)/1e6 for k in values[0]['timer']}))
 result=dict(status='passed',kind='host dispatch and API span profile',torch=torch.__version__,modules=modules,rows=rows,records=RECORDS,summary=summary,
  caveat='Host convolution spans include enqueue/driver work and any blocking inside calls. They are not exclusive GPU kernel durations. No per-op synchronization was added. Mirror preserves public tensor/finite checks and CPU-ready boundary, but excludes public stream/model locks.',
  flags=dict(fast_math=False,cpu_fallback=False,precision='FP32'),input_sha=__import__('hashlib').sha256(z.tobytes()).hexdigest())
 output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)
