"""Exact phase/state gates followed by optional short, balanced regional timing."""
import os
for key in ('OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
for key in ('CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES'):
    os.environ[key]='-1'
from pathlib import Path
import argparse, hashlib, json, random, statistics, time, traceback
import numpy as np
import onnx
from onnx import numpy_helper as N
import onnxruntime as ort
from prepare import sha, dump, require

def equal(a,b):
    require(a.dtype==b.dtype==np.float32 and a.shape==b.shape,'Output dtype/shape mismatch')
    require(np.isfinite(a).all() and np.isfinite(b).all(),'Nonfinite output')
    require(np.array_equal(a.view(np.uint32),b.view(np.uint32)),'Bitwise phase/state mismatch')
    return float(np.max(np.abs(a-b))) if a.size else 0.0

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--baseline-library',type=Path,required=True);p.add_argument('--baseline-sha256',required=True)
    p.add_argument('--candidate-library',type=Path,required=True);p.add_argument('--candidate-sha256',required=True)
    p.add_argument('--threads',type=int,choices=(1,4),default=1);p.add_argument('--ort-version',default='1.30.0')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--time',action='store_true');a=p.parse_args()
    require(not a.output.exists(),'Output exists');require(ort.__version__==a.ort_version,'Wrong ORT version')
    plan=json.loads(a.plan.read_text());require(plan['version']=='amd_phase_state_plan_v1','Wrong plan')
    require(plan['initializers_byte_identical'] and plan['projection_nodes_unchanged'],'Unverified source derivation')
    expected={a.baseline_library:a.baseline_sha256,a.candidate_library:a.candidate_sha256,
              Path(plan['source']):plan['source_sha256'],a.plan.parent/plan['overlay']:plan['overlay_sha256']}
    for row in plan['regions']:
        for arm in ('baseline','candidate'):expected[a.plan.parent/row[arm+'_file']]=row[arm+'_sha256']
    for path,digest in expected.items():require(sha(path)==digest,'Artifact changed: '+str(path))
    expected.update({a.plan:sha(a.plan),Path(__file__):sha(__file__),Path(__file__).with_name('prepare.py'):sha(Path(__file__).with_name('prepare.py'))})
    a.output.mkdir(parents=True)
    result={'version':'amd_phase_state_screen_v1','status':'running','cpu_only':True,'onnxruntime':ort.__version__,
            'threads':a.threads,'native_threads':1,'precision':'FP32, ordered (current+previous)+bias',
            'timing_requested':a.time,'warmups':2,'pairs':6,'calls_per_interval':16,
            'native_call_cap_seconds':2.0,'measured_cap_seconds':1.0,'native_calls':0,
            'native_seconds':0.0,'measured_seconds':0.0,'regions':[],
            'artifacts_before':{str(k.resolve()):v for k,v in expected.items()},
            'scope':'Phase assembly only; supplied projections and exact trained bias. Matrix work unchanged.',
            'timing_boundary':'Complete session.run, including operator dispatch and output/history allocations. Preparation and comparisons excluded.'}
    def call(session,feed,measured=False,samples=None):
        require(result['native_seconds']<2 and result['measured_seconds']<1,'Native budget exceeded')
        start=time.perf_counter_ns()
        try:return session.run(None,feed)
        finally:
            elapsed=(time.perf_counter_ns()-start)/1e9
            result['native_seconds']+=elapsed;result['native_calls']+=1
            if measured:result['measured_seconds']+=elapsed
            if samples is not None:samples.append(elapsed)
            require(result['native_seconds']<2 and result['measured_seconds']<1,'Native budget exceeded')
    def session(path,lib):
        options=ort.SessionOptions();options.intra_op_num_threads=a.threads;options.inter_op_num_threads=1
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        for k in ('session.intra_op.allow_spinning','session.inter_op.allow_spinning'):options.add_session_config_entry(k,'0')
        options.register_custom_ops_library(str(lib.resolve()))
        s=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
        require(s.get_providers()==['CPUExecutionProvider'],'Unexpected provider');return s
    rng=np.random.default_rng(84014);order_rng=random.Random(84014)
    prepared=[]
    try:
        for row in plan['regions']:
            c,s=row['channels'],row['stride'];width=c*s
            sessions={arm:session(a.plan.parent/row[arm+'_file'],getattr(a,arm+'_library')) for arm in ('baseline','candidate')}
            cm=onnx.load(a.plan.parent/row['candidate_file']);bias=N.to_array(next(w for w in cm.graph.initializer if w.name==row['bias_name']))
            require(hashlib.sha256(bias.tobytes()).hexdigest()==row['bias_sha256'],'Bias mismatch')
            rec={'node':row['node'],'channels':c,'stride':s,'checks':[],'timings':[]};result['regions'].append(rec)
            def feed(t,scale=1.0):
                return {'current':(rng.standard_normal((1,width,t))*scale).astype(np.float32),
                        'previous':(rng.standard_normal((1,width,t))*scale).astype(np.float32),
                        'history':rng.standard_normal((1,width,1),dtype=np.float32)}
            def oracle(f):
                t=f['current'].shape[-1]
                shifted=np.concatenate((f['history'],f['previous']),axis=2)[...,:t]
                summed=np.add(f['current'],shifted,dtype=np.float32).reshape(1,c,s,t)
                out=np.add(summed,bias.reshape(1,c,1,1),dtype=np.float32).transpose(0,1,3,2).reshape(1,c,t*s)
                history=f['previous'][...,-1:].copy() if t else f['history'].copy()
                return out,history
            def check(label,feeds):
                got={}
                for arm in sessions:
                    f=feeds[arm];snap={k:v.copy() for k,v in f.items()}
                    got[arm]=call(sessions[arm],f)
                    require(len(got[arm])==2,'Both output slots required')
                    for x,y in zip(got[arm],oracle(f)):equal(x,y)
                    for k in snap:equal(f[k],snap[k])
                for x,y in zip(got['baseline'],got['candidate']):equal(x,y)
                rec['checks'].append({'case':label,'outputs_and_history_bitwise':True,'oracle_exact':True,'input_unchanged':True})
                return got
            for t in sorted({0,1,row['T40'],row['T80'],row['T80']+1,row['T80']+3}):
                for scale in (0.0,1e-5,1.0):
                    f=feed(t,scale);check(f'T{t}/scale{scale}',dict(baseline=f,candidate=f))
            f=feed(row['T80']);f['previous']=f['current']
            check('read_input_alias',dict(baseline=f,candidate=f))
            f=feed(row['T80']);f['current'].fill(2**20);f['previous'].fill(-(2**20));f['history'].fill(-(2**20))
            check('large_finite_cancellation',dict(baseline=f,candidate=f))
            # Own-history interleaving, empty input between packets, and resets.
            histories={arm:[np.zeros((1,width,1),np.float32),feed(0)['history']] for arm in sessions}
            histories['candidate']=[h.copy() for h in histories['baseline']]
            for index,(stream,t) in enumerate(((0,1),(1,2),(0,0),(0,2),(1,1))):
                f=feed(t);fs={arm:dict(f,history=histories[arm][stream]) for arm in sessions}
                got=check(f'independent_stream{stream}/step{index}',fs)
                for arm in sessions:histories[arm][stream]=got[arm][1]
            long=feed(5);prefix={k:(v[...,:2].copy() if k!='history' else v.copy()) for k,v in long.items()}
            a0=check('prefix',dict(baseline=prefix,candidate=prefix));a1=check('extended',dict(baseline=long,candidate=long))
            for arm in sessions:equal(a0[arm][0],a1[arm][0][...,:2*s])
            # Full/partition parity uses each arm's returned state, not its peer's.
            whole=feed(5);full=check('partition_full',dict(baseline=whole,candidate=whole))
            hs={arm:whole['history'].copy() for arm in sessions};chunks={arm:[] for arm in sessions};start=0
            for t in (1,2,2):
                f={k:v[...,start:start+t] for k,v in whole.items() if k!='history'}
                got=check(f'partition/{start}',{arm:dict(f,history=hs[arm]) for arm in sessions})
                for arm in sessions:hs[arm]=got[arm][1];chunks[arm].append(got[arm][0])
                start+=t
            for arm in sessions:equal(np.concatenate(chunks[arm],axis=2),full[arm][0]);equal(hs[arm],full[arm][1])
            reset=feed(1);reset['history'].fill(0)
            r0=check('resetA',dict(baseline=reset,candidate=reset));r1=check('resetB',dict(baseline=reset,candidate=reset))
            for arm in sessions:
                for x,y in zip(r0[arm],r1[arm]):equal(x,y)
            prepared.append((row,rec,sessions))
        # All regions must clear numerical/state gates before any measured call.
        if a.time:
            for row,rec,sessions in prepared:
                width=row['channels']*row['stride']
                for ms in (40,80):
                    t=row[f'T{ms}'];f={k:rng.standard_normal((1,width,t if k!='history' else 1),dtype=np.float32)
                                      for k in ('current','previous','history')}
                    for _ in range(2):
                        for ss in sessions.values():call(ss,f)
                    orders=[['baseline','candidate'],['candidate','baseline']]*3;order_rng.shuffle(orders)
                    pairs=[]
                    for index,order in enumerate(orders):
                        times={};last={};raw_samples={}
                        for arm in order:
                            raw_samples[arm]=[]
                            for _ in range(16):last[arm]=call(sessions[arm],f,True,raw_samples[arm])
                            times[arm]=sum(raw_samples[arm])/16
                        for x,y in zip(last['baseline'],last['candidate']):equal(x,y)
                        pairs.append({'pair':index,'order':order,'seconds_per_call':times,'raw_call_seconds':raw_samples,
                                      'reduction':1-times['candidate']/times['baseline']})
                    reductions=[r['reduction'] for r in pairs];median=statistics.median(reductions)
                    rec['timings'].append({'packet_ms':ms,'T':t,'pairs':pairs,'median_reduction':median,
                                          'faster_pairs':sum(x>0 for x in reductions),
                                          'qualifies':median>=0.10 and all(x>0 for x in reductions)})
        require(all(sha(k)==v for k,v in expected.items()),'Artifacts changed during execution')
        result.update(status='passed',artifacts_unchanged=True)
    except Exception as exc:
        result.update(status='failed',error=str(exc),traceback=traceback.format_exc());raise
    finally:
        dump(a.output/'results.json',result)
        print(json.dumps({k:result[k] for k in ('status','native_calls','native_seconds','measured_seconds')}))
if __name__=='__main__':main()
