"""Four-worker second-pair experiment; retains the previous first-pair baseline."""
from pathlib import Path
import importlib.util,copy,json,hashlib,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];V2=HERE.parent/'apple-thread-distribution-v2';BUILD=HERE/'build'
spec=importlib.util.spec_from_file_location('thread_v2_common',V2/'experiment.py');e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
np=e.np;onnx=e.onnx;helper=e.helper;ort=e.ort
DOMAIN='fast.audiovae.apple.paired.second.v1'
CURRENT='ncc_up_2_split_matmul_current_node';PREVIOUS='ncc_up_2_split_matmul_previous_unshifted_node'
RESULT={'status':'running','cpu_only':True,'threads':4,'checks':0,'max_abs':0.,'call_seconds':0.}
def pair_nodes(model):
    n={n.name:n for n in model.graph.node};a,b=n[CURRENT],n[PREVIOUS]
    assert a.input[0]==b.input[0] and a.domain==b.domain=='fast.audiovae.apple.matrix.sweep.v1'
    return a,b
def node(a,b,grain,serial=0,trace=0):
    return helper.make_node('PairedSecondProjectionsF32',[a.input[0],a.input[1],b.input[1]],list(a.output)+list(b.output),
        name='paired_second_projections',domain=DOMAIN,matrix_abi=1,threads=1,k=1024,n=3072,max_m=16,
        task_channels=grain,serial_mode=serial,trace=trace)
def options(threads=4,old=False):
    if old:return e.options(threads,old=True)
    opts=ort.SessionOptions();opts.intra_op_num_threads=threads;opts.inter_op_num_threads=1
    opts.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;opts.log_severity_level=3
    opts.add_session_config_entry('session.intra_op.allow_spinning','0');opts.add_session_config_entry('session.inter_op.allow_spinning','0')
    replacements={'libapple_state_v2.dylib':ROOT/'outputs/apple-thread-optimization-v1/build/libthread_state.dylib',
        'libapple_libxsmm_panel_core.dylib':V2/'build/libdist_xsmm_core.dylib','libapple_libxsmm_panel_ort.dylib':V2/'build/libdist_xsmm_ort.dylib',
        'libapple_multitile_core.dylib':V2/'build/libdist_multitile_core.dylib','libapple_multitile_ort.dylib':V2/'build/libdist_multitile_ort.dylib',
        'libapple_matrix_sweep_core.dylib':BUILD/'libdist3_sweep_core.dylib','libapple_matrix_sweep_ort.dylib':BUILD/'libdist3_sweep_ort.dylib'}
    libs=[e.BUNDLE/e.NATIVE['library']]+[e.BUNDLE/r['library'] for r in e.SPEC['additional_libraries']]
    for p in libs:opts.register_custom_ops_library(str(replacements.get(p.name,p)))
    opts.register_custom_ops_library(str(V2/'build/libdist_paired_ort.dylib'))
    opts.register_custom_ops_library(str(BUILD/'libdist3_paired_ort.dylib'))
    return opts
def session(model,threads=4,old=False):
    s=ort.InferenceSession(model.SerializeToString(),options(threads,old),providers=['CPUExecutionProvider']);s.disable_fallback()
    assert s.get_providers()==['CPUExecutionProvider'];return s
def transform(full,kind,grain=768):
    if kind=='frozen':return copy.deepcopy(full)
    model=e.transform(full,'paired',16);a,b=pair_nodes(model)
    if kind=='baseline':return model
    assert kind=='paired';nodes=[]
    for n in model.graph.node:
        if n.name==a.name:nodes.append(node(a,b,grain));continue
        if n.name==b.name:continue
        nodes.append(n)
    del model.graph.node[:];model.graph.node.extend(nodes);model.opset_import.append(helper.make_opsetid(DOMAIN,1))
    return model
def equal(x,y):
    e.equal(x,y);RESULT['checks']=e.RESULT['checks'];RESULT['max_abs']=e.RESULT['max_abs']
def compare_run(a,b):
    e.compare_run(a,b);RESULT['checks']=e.RESULT['checks'];RESULT['max_abs']=e.RESULT['max_abs']
def timed(fn):
    assert RESULT['call_seconds']<25,'Bounded call-time cap'
    t=time.perf_counter();c=time.process_time()
    try:out=fn()
    finally:wall=time.perf_counter()-t;RESULT['call_seconds']+=wall
    assert RESULT['call_seconds']<25,'Bounded call-time cap'
    return out,wall,time.process_time()-c
def artifact_receipt():
    paths=[Path(__file__),V2/'experiment.py',BUILD/'receipt.json',V2/'build/receipt.json',e.BUNDLE/'bundle.json']
    for p in (BUILD/'receipt.json',V2/'build/receipt.json'):
        r=json.loads(p.read_text())
        for f,h in r['files'].items():assert e.sha(f)==h;paths.append(Path(f))
    paths += [ROOT/'outputs/apple-thread-optimization-v1/build/libthread_state.dylib']
    return {str(p):e.sha(p) for p in paths}
