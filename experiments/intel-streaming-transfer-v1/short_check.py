"""Bounded CPU-only Intel streaming timing and attribution; no audio export."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','OMP_THREAD_LIMIT','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
os.environ.update(CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1',OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE')
import argparse, collections, hashlib, json, time
from pathlib import Path
import numpy as np
import onnxruntime as ort
UIDS=('bn_in_00151_1818','en_us_00103_1779','es_419_00060_1994')
ROOT=Path('/dev/shm/fast-audiovae-projection-candidate-20260908-r1')
ASSETS=Path('/var/tmp/fast-audiovae-20260907/assets/multilingual')
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def digest(v):
    assert np.isfinite(v).all()
    return dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.tobytes()).hexdigest())
def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['intel','stock','mimi'],default='intel');p.add_argument('--model');p.add_argument('--library',action='append',default=[]);p.add_argument('--worker',action='store_true');p.add_argument('--profile',action='store_true');p.add_argument('--gates',action='store_true');p.add_argument('--packet',type=int,default=80);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert a.packet in (40,80) and not(a.kind=='mimi' and a.packet==40)
    os.sched_setaffinity(0,{0}); assert ort.__version__=='1.29.0'
    libs=[]
    if a.kind=='mimi':
        root=Path('/dev/shm/fast-audiovae-streaming-comparison-mimi-20260908-r1/mimi_onnx');m=json.loads((root/'manifest.json').read_text());graph=root/m['stream_model'];latent='latent';fps=12.5;rate=24000;archive=ASSETS/'mimi_vae.npz'
        states=[dict(input=s['name'],output=None,shape=s['shape'],dtype=s['dtype']) for s in m['states']]
        assert sha(graph)==m['model_sha256'][m['stream_model']]
    else:
        m=json.loads((ROOT/'bundle.json').read_text());native=m['native']['Linux/x86_64'];key=native['model'] if a.kind=='intel' else m['fallback'];entry=m['streaming']['models'][key];graph=ROOT/entry['model'];latent=entry['latent_input'];fps=25;rate=48000;archive=ASSETS/'fast_audiovae2.npz';states=entry['states']
        assert sha(graph)==entry['model_sha256']
        if a.kind=='intel':
            assert sha(graph)=='59a937ee76d494896561b994c111b708603fb2f3c8e0fdf53df1380f2df41516'
            libs=[ROOT/native['library']]+[ROOT/x['library'] for x in entry.get('additional_libraries',native['additional_libraries'])]
        if a.model:
            graph=Path(a.model);pin=json.loads((graph.parent/'manifest.json').read_text());assert sha(graph)==next(r['sha256'] for r in [pin['baseline'],*pin['variants'].values()] if r['model']==graph.name)
            assert sha(graph.parent/pin['shared_weights']['file'])==pin['shared_weights']['sha256']
    libs.extend(map(Path,a.library)); artifacts=[graph,archive,*libs]
    if a.model:artifacts.append(graph.parent/pin['shared_weights']['file'])
    hashes={str(q):sha(q) for q in artifacts}
    opts=ort.SessionOptions();opts.intra_op_num_threads=opts.inter_op_num_threads=1;opts.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL;opts.log_severity_level=3
    for k in ('session.intra_op.allow_spinning','session.inter_op.allow_spinning'):opts.add_session_config_entry(k,'0')
    for lib in libs:opts.register_custom_ops_library(str(lib))
    if a.profile:opts.enable_profiling=True;opts.profile_file_prefix=str(a.output.with_suffix(''))
    session=ort.InferenceSession(str(graph),opts,providers=['CPUExecutionProvider']);session.disable_fallback();assert session.get_providers()==['CPUExecutionProvider']
    names=[o.name for o in session.get_outputs()];outstate=[s.get('output') or names[i+1] for i,s in enumerate(states)]
    with np.load(archive,allow_pickle=False) as ar:
        zs={}
        for uid in UIDS:
            keys=[k for k in (uid,uid+'_z',uid+'__z') if k in ar];assert len(keys)==1
            zs[uid]=np.ascontiguousarray(ar[keys[0]][...,:int(fps*1.6)])
    budget=0; calls=0
    def run(z,chunks,trace=False):
        nonlocal budget,calls
        state={s['input']:np.zeros(s['shape'],dtype=s['dtype']) for s in states};ys=[];packet=[];elapsed=0;pos=0
        for size in chunks:
            x=np.ascontiguousarray(z[...,pos:pos+size]);assert x.shape[-1]==size
            t=time.perf_counter();v=session.run(None,{latent:x,**state});dt=time.perf_counter()-t
            elapsed+=dt;budget+=dt;calls+=1;assert budget<25,'Short-run CPU budget exhausted'
            byname=dict(zip(names,v));y=v[0];assert y.shape==(1,1,round(size*rate/fps));assert np.isfinite(y).all()
            state={s['input']:byname[o] for s,o in zip(states,outstate)}
            assert all(list(state[s['input']].shape)==s['shape'] and np.isfinite(state[s['input']]).all() for s in states)
            ys.append(y);pos+=size
            if trace:packet.append({'audio':digest(y),'states':{k:digest(v) for k,v in state.items()}})
        assert pos==z.shape[-1]
        return {'seconds':elapsed,'duration':pos/fps,'rtf':elapsed/(pos/fps),'audio':digest(np.concatenate(ys,axis=-1)),'state':{k:digest(v) for k,v in state.items()},'packets':packet}
    report={'ort':ort.__version__,'kind':a.kind,'packet_ms':a.packet,'cpu':0,'threads':1,'cpu_only':True,'hashes':hashes,'timing_scope':'completed Session.run only; setup/checking/hashing excluded','results':[]}
    if a.worker:
        import sys
        print(json.dumps({'ready':True,'kind':a.kind,'hashes':hashes}),flush=True)
        for line in sys.stdin:
            request=json.loads(line)
            if request.get('quit'):break
            z=zs[request['uid']];chunk=int(request.get('packet_ms',80)*fps/1000)
            r=run(z,[chunk]*(z.shape[-1]//chunk));r.update(uid=request['uid'],packet_ms=request.get('packet_ms',80))
            print(json.dumps(r),flush=True)
        assert all(sha(q)==h for q,h in hashes.items())
        return
    if a.gates:
        assert a.kind=='intel'
        cases={**{k:z[...,:5] for k,z in zs.items()},'zero':np.zeros((1,64,5),np.float32),'tiny':zs[UIDS[0]][...,:5]*np.float32(1e-5)}
        for uid,z in cases.items():
            for chunks in ([1,2,2],[2,1,2],[1,1,1,1,1],[5]):
                r=run(z,chunks,True);r.update(uid=uid,chunks=chunks);report['results'].append(r)
        import sys
        sys.path.insert(0,'/var/tmp/intel-streaming-transfer-v1/src')
        from fast_audiovae.streaming import StreamingDecoder
        class Counter:
            def __init__(self,inner):self.inner=inner;self.calls=0
            def __getattr__(self,name):return getattr(self.inner,name)
            def run(self,*args,**kwargs):
                nonlocal budget,calls
                self.calls+=1;calls+=1;t=time.perf_counter()
                try:return self.inner.run(*args,**kwargs)
                finally:budget+=time.perf_counter()-t
        counter=Counter(session); public=StreamingDecoder(counter,entry);stream=public.streaming_decode()
        empty=np.empty((1,64,0),np.float32);assert stream.decode_chunk(empty).shape==(1,1,0) and counter.calls==0
        first=stream.decode_chunk(zs[UIDS[0]][...,:1]);previous={k:digest(v) for k,v in stream._history.items()};nc=counter.calls
        assert stream.decode_chunk(empty).shape==(1,1,0) and counter.calls==nc and previous=={k:digest(v) for k,v in stream._history.items()}
        stream.reset();assert stream.frames_decoded==0 and all(not np.any(v) for v in stream._history.values())
        assert digest(stream.decode_chunk(zs[UIDS[0]][...,:1]))==digest(first)
        stream.close();report['public_api']={'empty_before_and_after_no_native_call':True,'empty_retains_all_state':True,'reset_zeros_all_state':True,'reset_replay_exact':True}
    else:
        chunk=int(a.packet*fps/1000)
        if a.profile:
            z=zs[UIDS[0]][...,:chunk*10];r0=run(z,[chunk]*10);r=run(z,[chunk]*10);assert r0['audio']==r['audio'] and r0['state']==r['state']
            raw=Path(session.end_profiling());ev=json.loads(raw.read_text());runs=sorted([e for e in ev if e.get('name')=='model_run'],key=lambda x:x['ts']);assert len(runs)==20
            windows=[(e['ts'],e['ts']+e['dur']) for e in runs[10:]];nodes={};ops=collections.Counter()
            for e in ev:
                if e.get('cat')!='Node' or not e['name'].endswith('_kernel_time') or not any(t<=e['ts']<end for t,end in windows):continue
                arg=e.get('args',{});op=arg.get('op_name');row=nodes.setdefault(e['name'],dict(op=op,us=0,calls=0,inputs=arg.get('input_type_shape')));row['us']+=e['dur'];row['calls']+=1;ops[op]+=e['dur']
            report.update(profile=True,raw_profile=str(raw),ops_us=dict(ops.most_common()),nodes=dict(sorted(nodes.items(),key=lambda x:-x[1]['us'])),total_kernel_us=sum(ops.values()),model_run_us=sum(e['dur'] for e in runs[10:]))
        else:
            for uid,z in zs.items():run(z,[chunk]*(z.shape[-1]//chunk))
            for rep in range(3):
                for uid,z in zs.items():
                    r=run(z,[chunk]*(z.shape[-1]//chunk));r.update(uid=uid,rep=rep);report['results'].append(r)
            report['pooled_rtf']=sum(r['seconds'] for r in report['results'])/sum(r['duration'] for r in report['results'])
    assert all(sha(q)==h for q,h in hashes.items())
    maps=sorted({x.split()[-1] for x in Path('/proc/self/maps').read_text().splitlines() if '/' in x and '.so' in x and any(s in x.lower() for s in ('mkl','precision','stage','rawhistory','phase','fast_audiovae','paired'))})
    report.update(native_seconds=budget,native_calls=calls,libraries_mapped=maps,state_bytes=sum(np.zeros(s['shape'],s['dtype']).nbytes for s in states))
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k in ('kind','pooled_rtf','ops_us','total_kernel_us','native_seconds','native_calls')}))
if __name__=='__main__':main()
