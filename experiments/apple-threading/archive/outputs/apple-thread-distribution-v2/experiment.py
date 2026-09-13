"""Common setup for bounded CPU-only four-worker packet distribution checks."""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import copy,ctypes,hashlib,json,sys,time,statistics
import numpy as np
import onnx
from onnx import helper,numpy_helper,TensorProto
import onnxruntime as ort
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;BUILD=HERE/'build'
sys.path.insert(0,str(ROOT/'work/fast-audiovae-streaming-baseline/src'))
from fast_audiovae.streaming import StreamingDecoder
CFG=json.loads((ROOT/'outputs/apple-streaming-promotion-v1/config-r3.json').read_text())
BUNDLE=Path(CFG['bundle']);MAN=json.loads((BUNDLE/'bundle.json').read_text());NATIVE=MAN['native']['Darwin/arm64'];SPEC=MAN['streaming']['models'][NATIVE['model']]
DOMAIN='fast.audiovae.apple.paired.first.v1'
CURRENT='ncc_up_1_split_matmul_current_node_fixed_right';PREVIOUS='ncc_up_1_split_matmul_previous_unshifted_node_fixed_right'
RESULT={'status':'running','cpu_only':True,'threads':4,'checks':0,'max_abs':0.,'native_seconds':0.}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def graph():
    path=BUNDLE/SPEC['model'];assert sha(path)==CFG['stream_graph_sha256']
    return onnx.load(path)
def pair_nodes(model):
    nodes={n.name:n for n in model.graph.node};a,b=nodes[CURRENT],nodes[PREVIOUS]
    assert a.input[0]==b.input[0]
    return a,b
def paired(a,b,grain=32,serial=0,trace=0):
    return helper.make_node('PairedFirstProjectionsF32',[a.input[0],a.input[1],b.input[1]],list(a.output)+list(b.output),
        name='paired_first_projections',domain=DOMAIN,matrix_abi=1,threads=1,k=2048,n=8192,max_m=2,
        task_grain=grain,serial_mode=serial,trace=trace)
def options(threads=4,old=False):
    opts=ort.SessionOptions();opts.intra_op_num_threads=threads;opts.inter_op_num_threads=1
    opts.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;opts.log_severity_level=3
    opts.add_session_config_entry('session.intra_op.allow_spinning','0');opts.add_session_config_entry('session.inter_op.allow_spinning','0')
    replacements={'libapple_state_v2.dylib':ROOT/'outputs/apple-thread-optimization-v1/build/libthread_state.dylib',
        'libapple_libxsmm_panel_core.dylib':BUILD/'libdist_xsmm_core.dylib','libapple_libxsmm_panel_ort.dylib':BUILD/'libdist_xsmm_ort.dylib',
        'libapple_multitile_core.dylib':BUILD/'libdist_multitile_core.dylib','libapple_multitile_ort.dylib':BUILD/'libdist_multitile_ort.dylib'}
    libs=[BUNDLE/NATIVE['library']]+[BUNDLE/r['library'] for r in SPEC['additional_libraries']]
    for p in libs:opts.register_custom_ops_library(str(p if old else replacements.get(p.name,p)))
    if not old:opts.register_custom_ops_library(str(BUILD/'libdist_paired_ort.dylib'))
    return opts
def session(model,threads=4,old=False):
    s=ort.InferenceSession(model.SerializeToString(),options(threads,old),providers=['CPUExecutionProvider']);s.disable_fallback()
    assert s.get_providers()==['CPUExecutionProvider'];return s
def transform(model,kind,grain=32,trace=0):
    model=copy.deepcopy(model);a,b=pair_nodes(model)
    nodes=[]
    for n in model.graph.node:
        if kind=='paired' and n.name==a.name:nodes.append(paired(a,b,grain,trace=trace));continue
        if kind=='paired' and n.name==b.name:continue
        attr={x.name:helper.get_attribute_value(x) for x in n.attribute}
        if kind!='frozen' and n.op_type=='StatefulDW7SnakeF32' and attr['channels']<=256:
            n.attribute.append(helper.make_attribute('parallel_channels',max(1,attr['channels']//8)))
        if kind=='separate' and n.name==a.name:n.attribute.append(helper.make_attribute('parallel_panels',32))
        nodes.append(n)
    del model.graph.node[:];model.graph.node.extend(nodes)
    if kind=='paired':model.opset_import.append(helper.make_opsetid(DOMAIN,1))
    return model
def equal(x,y):
    assert x.shape==y.shape and x.dtype==y.dtype==np.float32 and np.isfinite(x).all() and np.isfinite(y).all()
    delta=float(np.max(np.abs(x-y))) if x.size else 0.;RESULT['max_abs']=max(RESULT['max_abs'],delta)
    assert np.array_equal(x.view('u4'),y.view('u4')),f'Bitwise parity failed: {delta}'
    RESULT['checks']+=1
def timed(fn):
    assert RESULT['native_seconds']<25,'Bounded experiment time cap'
    wall=time.perf_counter();cpu=time.process_time()
    try:out=fn()
    finally:elapsed=time.perf_counter()-wall;RESULT['native_seconds']+=elapsed
    assert RESULT['native_seconds']<25,'Bounded experiment time cap'
    return out,elapsed,time.process_time()-cpu
def packet_run(decoder,z,packet,capture=False):
    output=[];states=[]
    with decoder.streaming_decode() as stream:
        for t in range(0,z.shape[-1],packet):
            output.append(stream.decode_chunk(np.ascontiguousarray(z[...,t:t+packet])))
            if capture:states.append({k:v.copy() for k,v in stream._history.items()})
        assert stream.frames_decoded==z.shape[-1] and stream.flush().shape[-1]==0
    assert sum(x.shape[-1] for x in output)==z.shape[-1]*1920
    return output,states
def compare_run(a,b):
    assert len(a[0])==len(b[0]) and len(a[1])==len(b[1])
    for x,y in zip(a[0],b[0]):equal(x,y)
    for x,y in zip(a[1],b[1]):
        assert x.keys()==y.keys()
        for k in x:equal(x[k],y[k])
