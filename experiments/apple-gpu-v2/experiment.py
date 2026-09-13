"""Bounded Apple GPU candidates. Retains per-run inputs, checks and timing evidence."""
from pathlib import Path
import os, sys, json, time, hashlib, argparse, importlib.util, statistics, copy, fcntl
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
os.environ.update(PYTORCH_ENABLE_MPS_FALLBACK='0',PYTORCH_MPS_FAST_MATH='0',ORT_DISABLE_TELEMETRY='1',
    TORCHINDUCTOR_CACHE_DIR=str(HERE/'inductor-cache'),TORCHINDUCTOR_COMPILE_THREADS='1')
sys.path.insert(0,str(ROOT/'work/fast-audiovae-apple-gpu/src'))
sys.path.insert(0,str(ROOT/'outputs/apple-gpu-v1'))
import torch, numpy as np
torch._dynamo.config.suppress_errors=False
torch._dynamo.config.fail_on_recompile_limit_hit=True
from fast_audiovae.mps_decoder import MPSModel, _Snake
from fast_audiovae.gpu import GPUDecoder
from benchmark_mps import cpu_reference, comparison, ORIGINAL, CONFIG, CLIPS, sha

class TensorStep(torch.nn.Module):
    def __init__(self,model):
        super().__init__();self.model=model;self.names=tuple(model.state_shapes)
    def forward(self,z,*state):
        m=self.model; h=dict(zip(self.names,state)); out={}
        x,out['stem.history']=m.stem.decode(z,h['stem.history']);x=m.pointwise(x)
        for i,stage in enumerate(m.stages):x=stage.decode(x,h,out,f'stage{i}')
        x,out['final.history']=m.final_conv.decode(m.final_snake(x),h['final.history'])
        return (torch.tanh(x),*(out[n].contiguous() for n in self.names))

class CompiledModel:
    def __init__(self,model):
        self.model=model;self.state_shapes=model.state_shapes;self.state_bytes=model.state_bytes
        self.sample_rate,self.hop_samples,self.latent_channels=48000,1920,64
        self.names=tuple(model.state_shapes);self.step=TensorStep(model);self.compiled={}
    def decode(self,z,state):
        history=self.model._checked_states(z,state)
        length=z.shape[-1]
        if length==0:return self.model.decode(z,state)
        if length not in self.compiled:
            self.compiled[length]=torch.compile(self.step,backend='inductor',fullgraph=True,dynamic=False)
        result=self.compiled[length](z,*(history[n] for n in self.names))
        # Inductor may return a strided view even when contiguous was requested
        # in the graph. Materialize the public history contract outside it.
        return result[0],{n:v.contiguous() for n,v in zip(self.names,result[1:])}

def legacy():
    name='fast_audiovae._gpu_before_v2'; spec=importlib.util.spec_from_file_location(name,HERE/'mps_decoder_before.py')
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod.MPSModel.from_onnx(ORIGINAL)

def build(name):
    if name=='before':return legacy()
    if name=='before_compile':return CompiledModel(legacy())
    m=MPSModel.from_onnx(ORIGINAL)
    if name=='overlap':return m
    if name=='full_compile':return CompiledModel(m)
    from snake_candidates import apply_snake_candidate
    if name=='snake_compile':return apply_snake_candidate(m,'compile')
    if name=='snake_metal':return apply_snake_candidate(m,'metal')
    raise ValueError(name)

def cases():
    config=json.loads(CONFIG.read_text());crops={}
    with np.load(config['latents'],allow_pickle=False) as a:
        for label,uid in CLIPS:
            z=a[uid+'__z'];start=(z.shape[-1]//2-12)//2*2
            crops[label]=np.ascontiguousarray(z[...,start:start+24])
    return crops

def main(args):
    out=HERE/(args.name+'.json');assert not out.exists(),out
    result=dict(status='running',arms=args.arms,rows=[],checks=[],preparation=[],protocol=dict(
        packet_ms=[40,80],duration_ms=960,precision='FP32',threads=1,cpu_fallback=False,fast_math=False,
        boundary='Identical public GPU API: owned input upload, all 26 histories and audio finite checks, owned CPU-ready output, synchronization, flush.',
        exclude='Model construction and first compilation; ordinary fresh-state initialization remains timed.',
        repetitions=args.repetitions,warmup_sweeps=1,
        order='Reverse arm order on alternating clip plus repetition parity; all repetitions retained',
        short_evidence=True,atol=1e-5,rtol=1e-4),torch=torch.__version__,files={})
    def save():out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    def check(label,y,ref):
        r=comparison(y,ref);result['checks'].append(dict(label=label,**r));save()
        assert r['passed'],(label,r)
    startall=time.perf_counter()
    lock=(HERE/'gpu.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        assert torch.backends.mps.is_available() and torch.__version__.split('+')[0]=='2.14.0'
        for f in (Path(__file__),HERE/'mps_decoder_before.py',ROOT/'work/fast-audiovae-apple-gpu/src/fast_audiovae/mps_decoder.py'):
            result['files'][str(f)]=sha(f)
        crop=cases();reference=cpu_reference(ORIGINAL)
        refs={k:reference.run(None,{'z':z})[0] for k,z in crop.items()}
        result['inputs']=[dict(label=k,shape=list(z.shape),sha256=hashlib.sha256(z.tobytes()).hexdigest()) for k,z in crop.items()]
        decoders={}
        for arm in args.arms:
            before=time.perf_counter();model=build(arm);decoder=GPUDecoder(model,torch,{'experiment':arm},'streaming')
            row=dict(arm=arm,initialization_seconds=time.perf_counter()-before,first_calls=[]);result['preparation'].append(row);save()
            for length in (1,2):
                z=crop['English'][...,:length].copy();ref=reference.run(None,{'z':z})[0]
                with decoder.stream() as stream:
                    t=time.perf_counter();y=stream.decode_chunk(z);elapsed=time.perf_counter()-t
                    row['first_calls'].append(dict(frames=length,seconds=elapsed));save()
                    check(arm+f'/prepared{length}',y,ref)
                print('prepared',arm,length,round(elapsed,3),flush=True)
            decoders[arm]=decoder
        def counters():return {k:dict(v) for k,v in torch._dynamo.utils.counters.items() if k in ('stats','frames','graph_break','unimplemented')}
        result['compile_counters_after_preparation']=counters()
        for phase,repetitions in (('qualification',1),('warmup',1),('measured',args.repetitions)):
            for repetition in range(repetitions):
                for ci,(label,z) in enumerate(crop.items()):
                    for length in (1,2):
                        order=list(args.arms)
                        if (ci+repetition)%2:order.reverse()
                        for arm in order:
                            stream=decoders[arm].stream();torch.mps.synchronize();parts=[];times=[]
                            with stream:
                                for pos in range(0,z.shape[-1],length):
                                    packet=z[...,pos:pos+length]
                                    begin=time.perf_counter();y=stream.decode_chunk(packet);times.append(time.perf_counter()-begin);parts.append(y)
                                begin=time.perf_counter();tail=stream.flush();flush=time.perf_counter()-begin
                                assert tail.shape==(1,1,0) and stream.frames_decoded==24
                            joined=np.concatenate(parts,-1);r=comparison(joined,refs[label]);assert r['passed'],(arm,label,r)
                            result['rows'].append(dict(phase=phase,repetition=repetition,clip=label,arm=arm,packet_ms=40*length,
                                packet_seconds=times,flush_seconds=flush,rtf=(sum(times)+flush)/.96,comparison=r))
                            # Excludes compile and preparation. Fixed small audio and no auto repeats.
                            assert sum(sum(x['packet_seconds']) for x in result['rows'])<35,'Decode budget exhausted'
                save();print(phase,repetition+1,'rows',len(result['rows']),flush=True)
        summaries=[]
        for arm in args.arms:
            for ms in (40,80):
                rows=[r for r in result['rows'] if r['phase']=='measured' and r['arm']==arm and r['packet_ms']==ms]
                times=[t for r in rows for t in r['packet_seconds']]
                summaries.append(dict(arm=arm,packet_ms=ms,rtf=sum(sum(r['packet_seconds'])+r['flush_seconds'] for r in rows)/(.96*len(rows)),
                    mean_packet_ms=1000*statistics.mean(times),median_packet_ms=1000*statistics.median(times),
                    stream_rtfs=[r['rtf'] for r in rows]))
        result.update(status='passed',summary=summaries,compile_counters_after_timing=counters())
        assert result['compile_counters_after_preparation']==result['compile_counters_after_timing'],'Compilation occurred during timed sweeps'
    except BaseException as e:result.update(status='failed',error=repr(e));raise
    finally:result['wall_seconds']=time.perf_counter()-startall;save()
    print(json.dumps(result['summary'],indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--arms',nargs='+',required=True);p.add_argument('--repetitions',type=int,default=2)
    main(p.parse_args())
