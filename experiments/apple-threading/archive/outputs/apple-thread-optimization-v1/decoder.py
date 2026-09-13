"""Short actual-codec screen for operator winners. Never changes released bundles."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import copy, hashlib, json, random, statistics, sys, time
import numpy as np
import onnx
from onnx import helper
import onnxruntime as ort
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'work/fast-audiovae-streaming-baseline/src'))
from fast_audiovae.streaming import StreamingDecoder
CFG=json.loads((ROOT/'outputs/apple-streaming-promotion-v1/config-r3.json').read_text())
BUNDLE=Path(CFG['bundle']);MAN=json.loads((BUNDLE/'bundle.json').read_text())
NATIVE=MAN['native']['Darwin/arm64'];SPEC=MAN['streaming']['models'][NATIVE['model']]
RESULT={'status':'running','cpu_only':True,'checks':0,'max_abs':0.,'call_seconds':0.,'runs':[]}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(): (HERE/'decoder-results.json').write_text(json.dumps(RESULT,indent=2)+'\n')
def make(model,kind,threads,grain,blocks):
    m=copy.deepcopy(model);changed=[]
    for n in m.graph.node:
        attr={a.name:helper.get_attribute_value(a) for a in n.attribute}
        if kind in ('matrix','combined') and n.op_type=='LibxsmmPanelSmeWeightLeftF32':
            n.attribute.append(helper.make_attribute('parallel_panels',grain));changed.append(n.name)
        if kind in ('dw','combined') and n.op_type=='StatefulDW7SnakeF32' and attr['channels']<=256:
            n.attribute.append(helper.make_attribute('parallel_channels',max(1,attr['channels']//blocks)));changed.append(n.name)
    o=ort.SessionOptions();o.intra_op_num_threads=threads;o.inter_op_num_threads=1
    o.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;o.log_severity_level=3
    o.add_session_config_entry('session.intra_op.allow_spinning','0');o.add_session_config_entry('session.inter_op.allow_spinning','0')
    libs=[BUNDLE/NATIVE['library']]
    replacements={'libapple_state_v2.dylib':'libthread_state.dylib',
        'libapple_libxsmm_panel_core.dylib':'libthread_xsmm_core.dylib','libapple_libxsmm_panel_ort.dylib':'libthread_xsmm_ort.dylib'}
    for r in SPEC['additional_libraries']:
        p=BUNDLE/r['library']
        if kind!='frozen' and p.name in replacements: p=HERE/'build'/replacements[p.name]
        libs.append(p)
    for p in libs:o.register_custom_ops_library(str(p))
    s=ort.InferenceSession(m.SerializeToString(),o,providers=['CPUExecutionProvider']);s.disable_fallback()
    assert s.get_providers()==['CPUExecutionProvider']
    return StreamingDecoder(s,SPEC),dict(kind=kind,threads=threads,changed_nodes=changed,
        model_sha256=hashlib.sha256(m.SerializeToString()).hexdigest(),libraries={str(p):sha(p) for p in libs})
def equal(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype==np.float32 and np.isfinite(a).all() and np.isfinite(b).all()
    delta=float(np.max(np.abs(a-b))) if a.size else 0.
    RESULT['max_abs']=max(RESULT['max_abs'],delta)
    assert np.array_equal(a.view('u4'),b.view('u4')),f'Decoder/state parity {delta}'
    RESULT['checks']+=1
def one(d,z,packet,reference=None):
    outputs=[];states=[];dt=0.;cpu=0.
    with d.streaming_decode() as s:
        for i in range(0,z.shape[-1],packet):
            assert RESULT['call_seconds']<25
            part=np.ascontiguousarray(z[...,i:i+packet]);start=time.perf_counter();c=time.process_time()
            y=s.decode_chunk(part)
            dt+=time.perf_counter()-start;cpu+=time.process_time()-c
            assert RESULT['call_seconds']+dt<25
            outputs.append(y); states.append({k:v.copy() for k,v in s._history.items()})
        assert s.frames_decoded==z.shape[-1]
        assert s.flush().shape[-1]==0
    RESULT['call_seconds']+=dt
    if reference:
        assert len(outputs)==len(reference[0])==len(states)==len(reference[1])
        for y,r,st,rst in zip(outputs,reference[0],states,reference[1]):
            equal(y,r)
            assert st.keys()==rst.keys()
            for k in st:equal(st[k],rst[k])
    return outputs,states,dt,cpu
def main():
    micro=json.loads((HERE/'micro-results.json').read_text());assert micro['status']=='passed'
    ps=[p for p in micro['pairs'] if p['threads']==4]
    matrices=[p for p in ps if 'up_1' in p['label']]
    grain=max({p['grain'] for p in matrices},key=lambda g:statistics.median(p['median_reduction_percent'] for p in matrices if p['grain']==g))
    # Identify four/eight channel ranges across both representative channel widths.
    dw=[p for p in ps if 'direct_state' in p['label']]
    def count(p):return (256 if '666' in p['label'] or '542' in p['label'] else 32)//p['grain']
    blocks=max((4,8),key=lambda b:statistics.median(p['median_reduction_percent'] for p in dw if count(p)==b))
    RESULT.update(matrix_grain=grain,channel_blocks=blocks,protocol='Three960ms multilingual prefixes,40/80ms; one warmup and three paired repetitions; exact waveform and all26states; short edge prefixes;25s decoder-call cap')
    graph=BUNDLE/SPEC['model'];assert sha(graph)==CFG['stream_graph_sha256'];model=onnx.load(graph)
    models={};meta={}
    for kind in ('frozen','serial','matrix','dw','combined'):
        models[kind],meta[kind]=make(model,kind,4,grain,blocks)
    RESULT['models']=meta
    ids=['bn_in_00151_1818','en_us_00103_1779','es_419_00060_1994']
    with np.load(CFG['latents'],allow_pickle=False) as a:
        inputs={uid:np.ascontiguousarray(a[uid+'__z'][...,:24]) for uid in ids}
    # Independent zero/nonverbal/quiet prefixes and an odd tail, plus longer-packet BLAS fallback.
    with np.load(CFG['extra_latents'],allow_pickle=False) as a:
        inputs_edges=[np.ascontiguousarray(a[k+'__z'][...,:3]) for k in ('panel_000','digital_zero_1s','quiet_speech_6s')]
    inputs_edges += [np.zeros((1,64,3),dtype='f'),inputs[ids[1]][...,:3],inputs[ids[1]][...,:4]]
    for z in inputs_edges:
        for packet in (1,2,4):
            ref=one(models['frozen'],z,packet)
            for kind in ('serial','matrix','dw','combined'):one(models[kind],z,packet,ref)
    for packet in (1,2):
        for uid,z in inputs.items():
            ref=one(models['frozen'],z,packet)
            for rep in range(-1,3):
                kinds=['frozen','serial','matrix','dw','combined'];random.Random(f'{uid}/{packet}/{rep}').shuffle(kinds)
                for kind in kinds:
                    _,_,dt,cpu=one(models[kind],z,packet,ref)
                    RESULT['runs'].append(dict(uid=uid,packet_ms=packet*40,repeat=rep,kind=kind,
                        seconds=dt,cpu_seconds=cpu,rtf=dt/(z.shape[-1]*.04)))
                write()
    RESULT['pairs']=[]
    for packet in (40,80):
        for kind in ('serial','matrix','dw','combined'):
            pairs=[]
            for uid in ids:
                for rep in range(3):
                    rows={r['kind']:r for r in RESULT['runs'] if r['uid']==uid and r['packet_ms']==packet and r['repeat']==rep}
                    pairs.append(100*(1-rows[kind]['seconds']/rows['frozen']['seconds']))
            RESULT['pairs'].append(dict(kind=kind,packet_ms=packet,median_reduction=statistics.median(pairs),wins=sum(v>0 for v in pairs),reductions=pairs))
    RESULT['status']='passed';write();print(json.dumps({k:RESULT[k] for k in ('status','checks','max_abs','call_seconds','pairs')},indent=2))
if __name__=='__main__':
    try:main()
    except BaseException as e:RESULT.update(status='failed',error=repr(e));write();raise
