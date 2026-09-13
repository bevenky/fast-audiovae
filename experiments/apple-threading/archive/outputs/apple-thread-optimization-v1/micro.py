"""Bounded paired operator screen, CPU only, original trained constants."""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import copy, gc, hashlib, json, random, statistics, time
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
import onnxruntime as ort
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
cfg=json.loads((ROOT/'outputs/apple-streaming-promotion-v1/config-r3.json').read_text())
bundle=Path(cfg['bundle']); manifest=json.loads((bundle/'bundle.json').read_text())
native=manifest['native']['Darwin/arm64']; spec=manifest['streaming']['models'][native['model']]
REPORT={'status':'running','cpu_only':True,'ort':ort.__version__,'checks':0,'pairs':[],
        'timing_calls':0,'call_seconds':0.,'protocol':'Two warmups, six interleaved paired repetitions; three calls per repetition; 1/2/4 ORT workers; fixed original constants; no spin; 20s call cap'}
def write(): (HERE/'micro-results.json').write_text(json.dumps(REPORT,indent=2)+'\n')
def library(domain, candidate):
    if candidate:
        return HERE/'build'/('libthread_state.dylib' if '.state.' in domain else 'libthread_xsmm_ort.dylib')
    return bundle/next(x['library'] for x in spec['additional_libraries'] if x['domain']==domain)
def fixture(node, constants, grain=0):
    node=copy.deepcopy(node)
    if grain: node.attribute.append(h.make_attribute('parallel_channels' if '.state.' in node.domain else 'parallel_panels',grain))
    matrix='Libxsmm' in node.op_type
    attrs={a.name:h.get_attribute_value(a) for a in node.attribute}
    c=attrs['k'] if matrix else attrs['channels']; n=attrs['n'] if matrix else c
    vi=lambda name,s:h.make_tensor_value_info(name,T.FLOAT,s)
    inputs=[vi(node.input[0],[1,c,'T'])]
    outputs=[vi(node.output[0],[1,n,'T'])]
    if not matrix:
        inputs.append(vi(node.input[1],[1,c,6*attrs['dilation']]))
        outputs.append(vi(node.output[1],[1,c,6*attrs['dilation']]))
    arrays=[copy.deepcopy(constants[name]) for name in node.input if name in constants]
    model=h.make_model(h.make_graph([node],'isolated_native',inputs,outputs,arrays),
        opset_imports=[h.make_opsetid('',20),h.make_opsetid(node.domain,1)],ir_version=10)
    return model
def session(model, domain, threads, candidate=True):
    o=ort.SessionOptions(); o.intra_op_num_threads=threads; o.inter_op_num_threads=1
    o.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    o.log_severity_level=3
    o.add_session_config_entry('session.intra_op.allow_spinning','0')
    o.add_session_config_entry('session.inter_op.allow_spinning','0')
    o.register_custom_ops_library(str(library(domain,candidate)))
    s=ort.InferenceSession(model.SerializeToString(),o,providers=['CPUExecutionProvider'])
    s.disable_fallback(); assert s.get_providers()==['CPUExecutionProvider']; return s
def run(s,feed):
    assert REPORT['call_seconds']<20
    start=time.perf_counter()
    try:y=s.run(None,feed)
    finally:REPORT['call_seconds']+=time.perf_counter()-start
    assert REPORT['call_seconds']<20
    return y
def equal(a,b):
    assert len(a)==len(b)
    for x,y in zip(a,b):
        assert x.dtype==y.dtype==np.float32 and np.isfinite(x).all() and np.isfinite(y).all()
        assert x.shape==y.shape and np.array_equal(x.view('u4'),y.view('u4')), 'Bitwise parity failure'
    REPORT['checks']+=1
def paired(label,sessions,feed,grain,threads):
    references=run(sessions[0],feed)
    for _ in range(2):
        for s in sessions: equal(references,run(s,feed))
    rows=[]
    for repeat in range(6):
        order=[0,1];random.Random(label+str(repeat)).shuffle(order);samples={}
        for i in order:
            start=time.perf_counter();cpu=time.process_time()
            for _ in range(3): y=run(sessions[i],feed)
            samples[i]={'seconds':(time.perf_counter()-start)/3,'cpu':(time.process_time()-cpu)/3}
            REPORT['timing_calls']+=3; equal(references,y)
        rows.append({'baseline':samples[0],'candidate':samples[1],
                     'reduction':100*(1-samples[1]['seconds']/samples[0]['seconds'])})
    REPORT['pairs'].append(dict(label=label,threads=threads,grain=grain,rows=rows,
        median_reduction_percent=statistics.median(r['reduction'] for r in rows),
        wins=sum(r['reduction']>0 for r in rows),
        baseline_us=statistics.median(r['baseline']['seconds'] for r in rows)*1e6,
        candidate_us=statistics.median(r['candidate']['seconds'] for r in rows)*1e6))
    write()
def main():
    graph=bundle/spec['model']; assert hashlib.sha256(graph.read_bytes()).hexdigest()==cfg['stream_graph_sha256']
    m=onnx.load(graph); constants={x.name:x for x in m.graph.initializer}
    matrix=next(n for n in m.graph.node if n.op_type=='LibxsmmPanelSmeWeightLeftF32')
    states=[n for n in m.graph.node if n.op_type=='StatefulDW7SnakeF32']
    chosen=[]
    for channels in (256,32):
        chosen.extend(n for n in states if {a.name:h.get_attribute_value(a) for a in n.attribute}['channels']==channels and
                      {a.name:h.get_attribute_value(a) for a in n.attribute}['dilation'] in (1,9))
    rng=np.random.default_rng(20260913)
    for node in [matrix,*chosen]:
        is_matrix=node is matrix;attrs={a.name:h.get_attribute_value(a) for a in node.attribute}
        c=attrs['k'] if is_matrix else attrs['channels'];domain=node.domain
        base=fixture(node,constants); old=session(base,domain,1,False); serial=session(base,domain,1)
        times=(0,1,2,4) if is_matrix else (0,1,5,6*attrs['dilation']-1,6*attrs['dilation'],240 if c==256 else 1920,480 if c==256 else 3840)
        feeds=[]
        for t in times:
            feed={node.input[0]:rng.normal(0,.5,(1,c,t)).astype('f')}
            if not is_matrix:feed[node.input[1]]=rng.normal(0,.3,(1,c,6*attrs['dilation'])).astype('f')
            equal(run(old,feed),run(serial,feed));feeds.append(feed)
        del old,serial;gc.collect()
        grains=(16,32) if is_matrix else (max(1,c//8),max(1,c//4))
        for threads in (1,2,4):
            serial=session(base,domain,threads)
            for grain in grains:
                candidate=session(fixture(node,constants,grain),domain,threads)
                for feed in feeds: equal(run(serial,feed),run(candidate,feed))
                for feed in feeds:
                    t=feed[node.input[0]].shape[-1]
                    if t not in ((1,2) if is_matrix else ((240,480) if c==256 else (1920,3840))):continue
                    label=f'{node.name}/T{t}/threads{threads}/grain{grain}'
                    paired(label,[serial,candidate],feed,grain,threads)
                del candidate;gc.collect()
            del serial;gc.collect()
    REPORT['status']='passed';write()
    print(json.dumps({k:REPORT[k] for k in ('status','checks','call_seconds','timing_calls')}))
    for p in REPORT['pairs']:print(p['label'],round(p['median_reduction_percent'],2),p['wins'])
if __name__=='__main__':
    try:main()
    except BaseException as e:REPORT.update(status='failed',error=repr(e));write();raise
