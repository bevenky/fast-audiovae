"""Bounded queue-completion diagnostic, with real80ms arrival spacing."""
from pathlib import Path
import sys,os,time,json,statistics
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'outputs/apple-gpu-v2'))
import experiment as exp
import torch,numpy as np
from fast_audiovae.gpu import GPUDecoder,_latents,_policy

def main():
 out=HERE/'queue-probe-r1.json';assert not out.exists()
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 model=exp.build('before_compile');decoder=GPUDecoder(model,torch,{'experiment':'queue'},'streaming');z=exp.cases()['English']
 with decoder.stream() as s:
  for i in range(4):s.decode_chunk(z[...,2*i:2*i+2].copy())
 rows=[]
 with torch.inference_mode(),torch.autocast('mps',enabled=False):
  for repetition in range(2):
   order=[('public',False),('public',True),('split_wait',False),('split_wait',True)]
   if repetition:order.reverse()
   for route,paced in order:
    states={};stream=decoder.stream();torch.mps.synchronize();arrival=time.perf_counter()
    for i in range(8):
     if paced:time.sleep(max(0,arrival+i*.080-time.perf_counter()))
     packet=z[...,i*2:i*2+2];begin=time.perf_counter_ns();times={}
     if route=='public':
      y=stream.decode_chunk(packet)
     else:
      _policy();owned=_latents(packet);a=time.perf_counter_ns();times['input_check_copy_ms']=(a-begin)/1e6
      x=torch.from_numpy(owned).to('mps',dtype=torch.float32,non_blocking=False);b=time.perf_counter_ns();times['upload_ms']=(b-a)/1e6
      audio,state=model.decode(x,dict(states));a=time.perf_counter_ns();times['model_host_ms']=(a-b)/1e6
      torch.mps.synchronize();b=time.perf_counter_ns();times['remaining_model_gpu_wait_ms']=(b-a)/1e6
      assert audio.dtype==torch.float32 and tuple(audio.shape)==(1,1,3840) and set(state)==set(model.state_shapes)
      for name,shape in model.state_shapes.items():assert state[name].dtype==torch.float32 and tuple(state[name].shape)==shape and state[name].device.type=='mps'
      a=time.perf_counter_ns();times['metadata_ms']=(a-b)/1e6
      checked=torch.cat([v.reshape(-1) for v in (audio,*state.values())]);assert torch.isfinite(checked).all().item()
      b=time.perf_counter_ns();times['finite_check_after_completion_ms']=(b-a)/1e6
      ready=audio.to('cpu',non_blocking=False);torch.mps.synchronize();y=ready.numpy().copy();a=time.perf_counter_ns();times['output_ms']=(a-b)/1e6;states=state
     times['total_ms']=(time.perf_counter_ns()-begin)/1e6
     rows.append(dict(repetition=repetition,route=route,paced=paced,packet=i,times=times,output_sha=__import__('hashlib').sha256(y.tobytes()).hexdigest()))
    stream.close()
 for i in range(8):assert len({r['output_sha'] for r in rows if r['packet']==i})==1
 summaries=[]
 for route in ('public','split_wait'):
  for paced in (False,True):
   rs=[r for r in rows if r['route']==route and r['paced']==paced and r['packet']>0]
   summaries.append(dict(route=route,paced=paced,packets=len(rs),mean_ms={k:statistics.mean(r['times'][k] for r in rs) for k in rs[0]['times']},max_total_ms=max(r['times']['total_ms'] for r in rs)))
 result=dict(status='passed',rows=rows,summary=summaries,exact_outputs_equal=True,packet_ms=80,arrival_ms=80,
  caveat='Split-wait inserts an extra model-completion synchronization, so only the public route is the actual public API latency. GPU wait is host time to complete queued work, not exclusive GPU kernel time.',
  torch=torch.__version__,precision='FP32',cpu_fallback=False,fast_math=False)
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
