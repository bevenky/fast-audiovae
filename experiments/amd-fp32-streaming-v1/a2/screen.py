"""Actual-weight A2 math, then optional bounded regional timing; no codec run.

Synthetic region inputs/history are not audio or encoded silence. Source model
and all runtime files are explicitly pinned. This script runs only when called.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='1'
os.environ.update(ORT_DISABLE_TELEMETRY='1',CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1')
import argparse
import copy
import ctypes
import faulthandler
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import onnx
from onnx import helper,numpy_helper
import onnxruntime as ort
import rewrite as rw
HERE=Path(__file__).resolve().parent
ATOL=1e-5
RTOL=1e-4
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def same(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def check(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype==np.float32
    assert np.isfinite(a).all() and np.isfinite(b).all()
    np.testing.assert_allclose(a,b,atol=ATOL,rtol=RTOL)
    return {'max_abs':float(np.max(np.abs(a-b),initial=0)), 'bitwise':same(a,b),'elements':int(a.size)}
def models(source,r):
    wanted=set(r['remove']);nodes=[copy.deepcopy(n) for n in source.graph.node if n.name in wanted]
    used={x for n in nodes for x in n.input}
    initializers=[numpy_helper.from_array(np.array(numpy_helper.to_array(t),copy=True),t.name)
                  for t in source.graph.initializer if t.name in used]
    c,h=r['channels'],r['halo']
    vi=lambda name,d:helper.make_tensor_value_info(name,onnx.TensorProto.FLOAT,d)
    # Retain the full streaming graph's intermediate shape declarations. Its
    # rewriter deliberately gives T+halo intermediates distinct named symbols.
    # With these dropped, ORT's native InferOutputShape can propagate an
    # unnamed unknown dimension as integer -1 into Slice shape inference.
    # Do not infer T+halo as T or disable optimization to hide this extraction
    # defect. The nodes/constants and their original ValueInfo stay unchanged.
    live={x for n in nodes for x in (*n.input,*n.output)}
    interface={r['x'],r['history'],r['y'],r['next']}
    metadata=[copy.deepcopy(v) for v in source.graph.value_info
              if v.name in live and v.name not in interface]
    by_name={v.name:v for v in metadata}
    for name in (r['node'].input[0],r['node'].output[0]):
        assert name in by_name,('Missing original streaming shape metadata',name)
        value=by_name[name];shape=value.type.tensor_type.shape.dim
        assert value.type.tensor_type.elem_type==onnx.TensorProto.FLOAT and len(shape)==3
        assert shape[0].dim_value==1 and shape[1].dim_value==c
        assert shape[2].HasField('dim_param') and shape[2].dim_param and shape[2].dim_param!='T'
    graph=helper.make_graph(nodes,'a2_existing_region',[vi(r['x'],[1,c,'T']),vi(r['history'],[1,c,h])],
                            [vi(r['y'],[1,c,'T']),vi(r['next'],[1,c,h])],initializers,value_info=metadata)
    model=helper.make_model(graph,opset_imports=[copy.deepcopy(o) for o in source.opset_import]);model.ir_version=source.ir_version
    onnx.checker.check_model(model)
    candidate,proof=rw.rewrite(copy.deepcopy(model),r['node'].name)
    proof['preserved_source_value_info']={v.name:hashlib.sha256(v.SerializeToString()).hexdigest() for v in metadata}
    return model,candidate,proof
def session(model,library,threads):
    o=ort.SessionOptions();o.intra_op_num_threads=threads;o.inter_op_num_threads=1
    o.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    o.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    o.register_custom_ops_library(str(library))
    s=ort.InferenceSession(model.SerializeToString(),sess_options=o,providers=['CPUExecutionProvider'])
    assert s.get_providers()==['CPUExecutionProvider']
    assert s.get_session_options().intra_op_num_threads==threads
    return s
def core_info(library):
    lib=ctypes.CDLL(str(library));out={}
    for name in ('ncc_abi_version','ncc_compiled_tile'):
        fn=getattr(lib,name);fn.argtypes=[];fn.restype=ctypes.c_uint32;out[name]=int(fn())
    for name in ('ncc_backend_available','ncc_vector_sine_available','ncc_streaming_math_version'):
        fn=getattr(lib,name);fn.argtypes=[ctypes.c_int32];fn.restype=ctypes.c_int32;out[name]=int(fn(5))
    lib.ncc_snake_math_name.argtypes=[ctypes.c_int32];lib.ncc_snake_math_name.restype=ctypes.c_char_p
    out['math']=lib.ncc_snake_math_name(5).decode()
    assert out=={'ncc_abi_version':1,'ncc_compiled_tile':256,'ncc_backend_available':1,
                'ncc_vector_sine_available':1,'ncc_streaming_math_version':1,'math':'sleef-u10-avx512f-fma'},out
    return out
def main():
    p=argparse.ArgumentParser()
    for k in ('source','build','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--source-sha',required=True);p.add_argument('--build-sha',required=True)
    p.add_argument('--node',required=True,help='Exact C1024/d9 node used for the initial timing region')
    p.add_argument('--ort-version',default='1.30.0');p.add_argument('--threads',type=int,choices=(1,4),default=1)
    p.add_argument('--timing',action='store_true');p.add_argument('--cap-seconds',type=float,default=2.0)
    p.add_argument('--empty-control',choices=('native','facade'),default='native',
                   help='native preserves the original diagnostic; facade uses production empty-bypass contract')
    p.add_argument('--probe-empty-control',action='store_true',help='Stop after one native empty-control diagnostic; no candidate or timing')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    report={'complete':False,'scope':'isolated actual-weight regions, synthetic x/raw history, no decoder',
            'atol':ATOL,'rtol':RTOL,'threads':a.threads,'inter_op_threads':1,'providers':['CPUExecutionProvider'],
            'native_seconds':0.0,'native_calls':0,'math':[],'state_checks':[],'timings':[],
            'empty_control_policy':a.empty_control,
            'empty_control_note':'facade means zero Session.run calls and unchanged raw history, matching StreamingDecoder.decode_chunk T0 bypass; candidate T0 still executes',
            'probe_empty_control_only':a.probe_empty_control}
    progress_file=(out/'progress.jsonl').open('x')
    fault_file=(out/'fault.log').open('x');faulthandler.enable(file=fault_file,all_threads=True)
    routes={};case={}
    def progress(event,**fields):
        row={'event':event,**case,**fields,'completed_native_calls':report['native_calls'],
             'counted_native_seconds':report['native_seconds']}
        progress_file.write(json.dumps(row)+'\n');progress_file.flush();os.fsync(progress_file.fileno())
    def call(s,r,x,h):
        assert report['native_seconds']<a.cap_seconds,'Native cap reached before call'
        progress('before_session_run',node=r['node'].name,arm=routes[id(s)],T=int(x.shape[2]))
        start=time.perf_counter_ns()
        y,n=s.run([r['y'],r['next']],{r['x']:x,r['history']:h})
        elapsed=(time.perf_counter_ns()-start)*1e-9
        report['native_seconds']+=elapsed;report['native_calls']+=1
        progress('after_session_run',node=r['node'].name,arm=routes[id(s)],T=int(x.shape[2]),seconds=elapsed)
        assert report['native_seconds']<=a.cap_seconds,'Native cap exceeded after call; preserve failure'
        return y,n,elapsed
    try:
        assert platform.system()=='Linux' and platform.machine() in ('x86_64','amd64')
        assert ort.__version__==a.ort_version and 0<a.cap_seconds<=2.0
        assert not a.probe_empty_control or (a.empty_control=='native' and not a.timing)
        assert sha(a.source)==a.source_sha and sha(a.build)==a.build_sha
        build=json.loads(a.build.read_text());assert build['complete']
        control_lib=Path(build['baseline_library']);candidate_lib=Path(build['library'])
        assert sha(control_lib)==build['baseline_library_sha256'] and sha(candidate_lib)==build['library_sha256']
        assert all(sha(k)==v for k,v in build['inputs'].items()),'Build sources changed'
        files={str(q.resolve()):sha(q) for q in [a.source,a.build,control_lib,candidate_lib,HERE/'screen.py',HERE/'rewrite.py']}
        report.update(artifact_hashes_before=files,onnxruntime=ort.__version__,control_core=core_info(control_lib),candidate_core=core_info(candidate_lib),
            timing_boundary='Session.run including ORT output allocation, dispatch, original channel scheduling and next-history copy; checks/setup excluded',
            budget_boundary='All regional Session.run calls; load, JIT-free metadata queries and session construction excluded')
        source=onnx.load(a.source)
        if any(t.data_location==onnx.TensorProto.EXTERNAL for t in source.graph.initializer):
            raise ValueError('Use the embedded authenticated streaming graph; no unhashed external weights')
        names=[n.name for n in source.graph.node if n.domain=='venky.audio.cpu.portable' and n.op_type=='SnakeDW7SnakeF32'
               and rw.attrs(n).get('channels') in (512,1024)]
        regions=[rw.inspect(source,n) for n in names]
        assert {(r['channels'],r['dilation']) for r in regions}=={(c,d) for c in (512,1024) for d in (1,3,9)} and len(regions)==6
        target=next(r for r in regions if r['node'].name==a.node);assert target['channels']==1024 and target['dilation']==9
        sessions={}
        for r in regions:
            name=r['node'].name;c,h=r['channels'],r['halo'];rng=np.random.default_rng(6200+c+r['dilation'])
            old,new,proof=models(source,r)
            case.update(phase='setup',node=name)
            progress('before_session_create',arm='control')
            left_session=session(old,control_lib,a.threads);routes[id(left_session)]='control'
            progress('after_session_create',arm='control')
            if a.probe_empty_control:
                case.update(phase='empty_control_probe',pattern='empty')
                call(left_session,r,np.empty((1,c,0),np.float32),np.zeros((1,c,h),np.float32))
                report.update(complete=True,scope='One native empty-control diagnostic only; no candidate qualification')
                return
            progress('before_session_create',arm='candidate')
            right_session=session(new,candidate_lib,a.threads);routes[id(right_session)]='candidate'
            progress('after_session_create',arm='candidate')
            ss=(left_session,right_session);sessions[name]=ss
            m80=16 if c==1024 else 96
            patterns=[('empty',0),('single',1),('tail3',3),('short40',m80//2),('normal80',m80),
                      ('below_halo',h-1),('at_halo',h),('above_halo',h+1),('tile_tail',257),
                      ('zero',m80),('quiet',m80),('alternating_large',m80)]
            for pattern,t in patterns:
                case.update(phase='math',node=name,pattern=pattern)
                x=(rng.normal(0,.15,(1,c,t))).astype(np.float32);history=(rng.normal(0,.15,(1,c,h))).astype(np.float32)
                if pattern=='zero':x.fill(0);history.fill(0)
                if pattern=='single':history.fill(0)
                if pattern=='quiet':x*=np.float32(1e-6);history*=np.float32(1e-6)
                if pattern=='alternating_large':
                    x[:]=np.resize(np.array([-32768.,.000001,32768.,-.000001],np.float32),x.shape)
                    history[:]=np.resize(np.array([32768.,-.000001,-32768.,.000001],np.float32),history.shape)
                xb,hb=x.tobytes(),history.tobytes()
                if t==0 and a.empty_control=='facade':
                    # The public streaming API returns before Session.run on
                    # an empty packet; do not claim native Slice T0 support.
                    progress('control_empty_facade_oracle',arm='control',T=0)
                    left,lh=np.empty_like(x),history.copy()
                else:left,lh,_=call(ss[0],r,x,history)
                right,rh,_=call(ss[1],r,x,history)
                comparison=check(right,left);expected=np.concatenate([history,x],axis=2)[:,:,-h:]
                assert same(lh,expected) and same(rh,expected) and x.tobytes()==xb and history.tobytes()==hb
                report['math'].append({'node':name,'channels':c,'dilation':r['dilation'],'pattern':pattern,'T':t,
                                       'waveform':comparison,'raw_next_history_exact':True,'inputs_unchanged':True,
                                       'control_session_executed':bool(t or a.empty_control=='native')})
            # Separate A/B streams, tiny uneven packets and future perturbation.
            initial=rng.normal(0,.15,(1,c,h)).astype(np.float32);initial_bytes=initial.tobytes()
            chunk=rng.normal(0,.15,(1,c,7)).astype(np.float32)
            for arm,s in zip(('control','candidate'),ss):
                case.update(phase='state_checks',node=name,pattern='partition_interleave_future')
                full,fullh,_=call(s,r,chunk,initial)
                first,h1,_=call(s,r,np.ascontiguousarray(chunk[:,:,:2]),initial)
                # Interleaved unrelated stream must not mutate h1 or initial.
                h1bytes=h1.tobytes();call(s,r,np.zeros((1,c,1),np.float32),np.zeros_like(initial))
                tail,h2,_=call(s,r,np.ascontiguousarray(chunk[:,:,2:]),h1)
                partition=check(np.concatenate([first,tail],axis=2),full)
                future=chunk.copy();future[:,:,2:]+=np.float32(8)
                altered,_,_=call(s,r,future,initial);prefix=check(altered[:,:,:2],full[:,:,:2])
                assert same(h2,fullh) and h1.tobytes()==h1bytes and initial.tobytes()==initial_bytes
                report['state_checks'].append({'node':name,'arm':arm,'partition':partition,'future_prefix':prefix,
                    'raw_history_exact':True,'old_states_unchanged':True,'independent_interleaving':True})
            print(json.dumps({'node':name,'math_patterns':12,'state_checks':2,'native_seconds':report['native_seconds']}),flush=True)
        assert len(report['math'])==72 and len(report['state_checks'])==12
        report['math_passed']=True
        if a.timing:
            ss=sessions[a.node];r=target;c,h=r['channels'],r['halo'];rng=np.random.default_rng(6299)
            for t in (8,16):
                case.update(phase='warmup',node=a.node,pattern='timing_input')
                x=rng.normal(0,.15,(1,c,t)).astype(np.float32);history=rng.normal(0,.15,(1,c,h)).astype(np.float32)
                for warm in range(2):
                    for i in (warm%2,1-warm%2):call(ss[i],r,x,history)
                pairs=[]
                for i,order in enumerate(((0,1),(1,0),(0,1))):
                    case.update(phase='timing',pair=i)
                    values={};outputs={}
                    for arm in order:
                        y,n,seconds=call(ss[arm],r,x,history);values[arm]=seconds;outputs[arm]=(y,n)
                    assert values[0]>0 and values[1]>0
                    parity=check(outputs[1][0],outputs[0][0]);assert same(outputs[1][1],outputs[0][1])
                    pairs.append({'pair':i,'order':['control' if z==0 else 'candidate' for z in order],
                        'control_seconds':values[0],'candidate_seconds':values[1],
                        'reduction_percent':100*(1-values[1]/values[0]),'waveform':parity,'raw_history_exact':True})
                report['timings'].append({'node':a.node,'T':t,'geometry':[1,c,t],'warmups_per_arm':2,
                    'pairs':pairs,'median_paired_reduction_percent':float(np.median([p['reduction_percent'] for p in pairs])),
                    'wins':sum(p['candidate_seconds']<p['control_seconds'] for p in pairs)})
        report['artifact_hashes_after']={k:sha(k) for k in files};assert files==report['artifact_hashes_after']
        assert all(sha(k)==v for k,v in build['inputs'].items())
        report['complete']=True
    except BaseException as e:report['error']=repr(e);raise
    finally:
        (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        progress_file.close();faulthandler.disable();fault_file.close()
if __name__=='__main__':main()
