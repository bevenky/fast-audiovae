"""Direct upstream AudioVAE2 MPS reference, short causal streaming comparison."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys
import time

HERE=Path(__file__).resolve().parent
for key in ('PYTORCH_ENABLE_MPS_FALLBACK','PYTORCH_MPS_FAST_MATH','TORCHINDUCTOR_USE_FAST_MATH'):
    assert os.environ.get(key,'0')=='0'
    os.environ[key]='0'
os.environ.pop('PYTORCH_MPS_PREFER_METAL',None)
sys.path[:0]=[str(HERE.parent/'apple-gpu-v4'),str(HERE.parent/'apple-gpu-v2')]
import screen_repair as repair
import experiment as exp
import numpy as np
import torch
from fast_audiovae.gpu import GPUDecoder,_latents,_policy


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main():
    out=HERE/'upstream-gpu-r1.json';assert not out.exists()
    result=dict(status='running',checks=[],rows=[],preparation=[],ordinary_gpu_seconds=0.)
    def save():out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    def compare(label,a,b):
        assert a.dtype==b.dtype==np.float32 and a.shape==b.shape
        assert np.isfinite(a).all() and np.isfinite(b).all()
        d=np.abs(a.astype(np.float64)-b)
        passed=bool(np.allclose(a,b,atol=1e-5,rtol=1e-4))
        row=dict(label=label,passed=passed,max_abs=float(d.max()) if d.size else 0.,
                 rms=float(np.sqrt(np.mean(d*d))) if d.size else 0.)
        result['checks'].append(row)
        assert passed,row
    def original_call(stream,z):
        _policy();owned=_latents(z)
        x=torch.from_numpy(owned).to('mps',dtype=torch.float32,non_blocking=False)
        y=stream.decode_chunk(x)
        assert y.shape==(1,1,z.shape[-1]*1920) and y.device.type=='mps' and y.dtype==torch.float32
        assert len(stream._states)==26
        assert all(v.device.type=='mps' and v.dtype==torch.float32 for v in stream._states.values())
        assert torch.isfinite(torch.cat([y.reshape(-1),*(v.reshape(-1) for v in stream._states.values())])).all().item()
        ready=y.to('cpu',non_blocking=False);torch.mps.synchronize()
        return ready.numpy().copy()
    def call(route,stream,z,prepare=False):
        start=time.perf_counter()
        y=original_call(stream,z) if route=='upstream_gpu' else stream.decode_chunk(z)
        elapsed=time.perf_counter()-start
        if not prepare:
            result['ordinary_gpu_seconds']+=elapsed
            assert result['ordinary_gpu_seconds']<35,'Short ordinary GPU-call budget exceeded'
        return y,elapsed
    try:
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        assert torch.backends.mps.is_available() and torch.__version__.split('+')[0]=='2.14.0'
        provenance=json.loads(Path('/Users/venky/tech/pockettts/benchmarks/results/fleurs-en-pilot/audiovae2.json').read_text())['provenance']
        source=Path(provenance['source_path']);weights=Path(provenance['checkpoint'])
        assert sha(source)==provenance['source_sha256'] and sha(weights)==provenance['checkpoint_sha256']
        spec=importlib.util.spec_from_file_location('_upstream_audiovae2_gpu_reference',source)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        state=torch.load(weights,map_location='cpu',weights_only=True)
        upstream=module.AudioVAE(module.AudioVAEConfig())
        upstream.load_state_dict(state.get('state_dict',state),strict=True)
        upstream.eval().requires_grad_(False).to(device='mps',dtype=torch.float32)
        assert all(p.device.type=='mps' and p.dtype==torch.float32 for p in upstream.parameters())
        weight_norm_modules=sum(hasattr(m,'weight_g') for m in upstream.decoder.modules())
        assert weight_norm_modules==45,weight_norm_modules
        candidate=GPUDecoder(repair.build('hybrid_pointwise'),torch,{'experiment':'qualified hybrid'},'streaming')
        result['provenance']=dict(upstream=provenance,upstream_execution_device='mps',
            upstream_class=spec.name+'.AudioVAE',weight_norm_retained_modules=weight_norm_modules,
            candidate='hybrid_pointwise',torch=torch.__version__)
        result['protocol']=dict(precision='FP32',host_threads=1,output_rate=48000,latent_rate=25,
            causal_streaming=True,cpu_fallback=False,fast_math=False,atol=1e-5,rtol=1e-4,
            reference='Unmodified pinned upstream AudioVAE class and official streaming_decode on MPS, original checkpoint and retained weight normalization.',
            timing='Owned CPU input through complete MPS decode to owned CPU output. Audio and all26 histories finite-checked on both paths. Model construction and compilation excluded.',
            timing_clips='Three960ms speech crops, one warmup and two measured sweeps, alternating arm order.',
            limit='35s ordinary measured/validation GPU API work; small fixtures only; no perceptual metric rerun.')
        files=[Path(__file__).resolve(),source,exp.CONFIG,exp.ROOT/'outputs/apple-gpu-v4/candidates.py',
               exp.ROOT/'outputs/apple-gpu-v4/repairs.py',exp.ROOT/'outputs/apple-gpu-v4/screen_repair.py']
        result['files']={str(p):sha(p) for p in files}
        config=json.loads(exp.CONFIG.read_text());fixtures={}
        for key in ('latents','extra_latents'):
            assert sha(config[key])==config['inputs_sha256'][key]
        with np.load(config['latents'],allow_pickle=False) as archive:
            for label,uid in exp.CLIPS:fixtures[label]=np.ascontiguousarray(archive[uid+'__z'][...,:7])
        with np.load(config['extra_latents'],allow_pickle=False) as archive:
            for i,name in enumerate([n for n in archive.files if n.endswith('__z')][:2]):fixtures[f'Expressive{i}']=np.ascontiguousarray(archive[name][...,:7])
        fixtures['Zero latent']=np.zeros((1,64,7),np.float32)
        fixtures['Quiet latent']=fixtures['English']*np.float32(.001)
        result['fixtures']=[dict(label=k,sha256=hashlib.sha256(v.tobytes()).hexdigest(),frames=v.shape[-1]) for k,v in fixtures.items()]
        with torch.inference_mode(),torch.autocast('mps',enabled=False):
            for length in (1,2):
                z=fixtures['English'][...,:length].copy()
                with upstream.streaming_decode() as s:
                    a,old_time=call('upstream_gpu',s,z,prepare=True)
                with candidate.stream() as s:
                    b,new_time=call('repaired_gpu',s,z,prepare=True)
                compare(f'prepare{length}/candidate-upstream',b,a)
                result['preparation'].append(dict(frames=length,upstream_s=old_time,candidate_s=new_time))
            counter=lambda:{k:dict(v) for k,v in torch._dynamo.utils.counters.items() if k in ('stats','frames','graph_break','unimplemented')}
            result['compile_counters_prepared']=counter()
            for label,z in fixtures.items():
                full=upstream.decode(torch.from_numpy(z).to('mps')).to('cpu').numpy().copy()
                torch.mps.synchronize();assert np.isfinite(full).all()
                old_parts=[];new_parts=[];pos=0
                with upstream.streaming_decode() as old,candidate.stream() as new:
                    for length in (2,1,2,2):
                        packet=z[...,pos:pos+length].copy()
                        a,_=call('upstream_gpu',old,packet);b,_=call('repaired_gpu',new,packet)
                        compare(f'{label}/{pos}/candidate-upstream-stream',b,a)
                        old_parts.append(a);new_parts.append(b);pos+=length
                    assert new.frames_decoded==7 and new.flush().shape==(1,1,0)
                compare(label+'/upstream-stream-full',np.concatenate(old_parts,-1),full)
                compare(label+'/candidate-upstream-full',np.concatenate(new_parts,-1),full)
            save();print('upstream quality passed',len(result['checks']),flush=True)
            crops=exp.cases()
            refs={label:upstream.decode(torch.from_numpy(z).to('mps')).to('cpu').numpy().copy() for label,z in crops.items()}
            torch.mps.synchronize()
            for phase,repetitions in (('warmup',1),('measured',2)):
                for repeat in range(repetitions):
                    for ci,(label,z) in enumerate(crops.items()):
                        for length in (1,2):
                            routes=['upstream_gpu','repaired_gpu']
                            if (ci+repeat)%2:routes.reverse()
                            for route in routes:
                                context=upstream.streaming_decode() if route=='upstream_gpu' else candidate.stream()
                                parts=[];times=[]
                                with context as stream:
                                    for pos in range(0,24,length):
                                        y,elapsed=call(route,stream,z[...,pos:pos+length]);parts.append(y);times.append(elapsed)
                                    if route=='repaired_gpu':assert stream.frames_decoded==24 and stream.flush().shape==(1,1,0)
                                joined=np.concatenate(parts,-1)
                                compare(f'{phase}/{repeat}/{label}/{length}/{route}',joined,refs[label])
                                result['rows'].append(dict(phase=phase,repetition=repeat,clip=label,packet_ms=40*length,route=route,
                                    packet_seconds=times,rtf=sum(times)/.96,samples=joined.shape[-1]))
                    save();print(phase,repeat,'complete',flush=True)
            result['compile_counters_final']=counter();assert result['compile_counters_final']==result['compile_counters_prepared']
        result['summary']=[]
        for route in ('upstream_gpu','repaired_gpu'):
            for ms in (40,80):
                rows=[r for r in result['rows'] if r['phase']=='measured' and r['route']==route and r['packet_ms']==ms]
                samples=[t for r in rows for t in r['packet_seconds']]
                result['summary'].append(dict(route=route,packet_ms=ms,rtf=sum(sum(r['packet_seconds']) for r in rows)/(.96*len(rows)),
                    mean_packet_ms=1000*sum(samples)/len(samples),streams=len(rows)))
        assert all(sha(p)==result['files'][str(p)] for p in files)
        result.update(status='passed',artifacts_unchanged=True)
        print(json.dumps(result['summary'],indent=2),flush=True)
    except BaseException as error:
        result.update(status='failed',error=repr(error));raise
    finally:save()


if __name__=='__main__':main()
