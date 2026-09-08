"""Focused CPU correctness for Apple r4 stage/upsample operators, without timing."""
import argparse,concurrent.futures,copy,hashlib,json,os
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='-1',NVIDIA_VISIBLE_DEVICES='void',ROCR_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
import numpy as np
import onnx
from onnx import TensorProto,helper,numpy_helper
import onnxruntime as ort
NATIVE='venky.audio.cpu';MATRIX='fast.audiovae.precision.apple.r4.experimental'
STAGE='fast.audiovae.precision.apple.r4.fused.stage.experimental';UP='fast.audiovae.precision.apple.r4.fused.upsample.experimental'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def fixture(c,up=False):
    rng=np.random.default_rng(731+c);initializers=[];nodes=[];constants=[];outputs=[]
    def put(name,a):initializers.append(numpy_helper.from_array(np.asarray(a,np.float32),name));return name
    def mat(name,w,x,m,k):
        if min(m,k)>=128:return helper.make_node('PrecisionMatMulF32',[w,x],[name],name=name,domain=MATRIX,native_abi=1,M=m,K=k,precision_mode=8,backend=3,shards=4)
        return helper.make_node('MatMul',[w,x],[name],name=name)
    value='x';inputs=[]
    if up:
        wc=put('wc',rng.normal(0,.10,(256,256)));wp=put('wp',rng.normal(0,.10,(256,256)));pb=put('phase_bias',rng.normal(0,.05,128))
        nodes += [mat('current',wc,'x',256,256),mat('previous',wp,'x',256,256)]
        nodes.append(helper.make_node('PhaseSumBiasInterleaveF32',['current','previous',pb],['phase'],name='phase',domain=NATIVE,native_abi=1,channels=128,stride=2,previous_shift=1,row_batches=0))
        value='phase';inputs=[wc,wp,pb];outputs=['current','previous','phase']
    for u,d in enumerate((1,3,9)):
        prefix=f'u{u}_';w=put(prefix+'dw',rng.normal(0,.1,(c,1,7)));b=put(prefix+'db',rng.normal(0,.025,c))
        ap=put(prefix+'ap',rng.uniform(.4,2,c));rp=put(prefix+'rp',rng.uniform(.2,1,c))
        aq=put(prefix+'aq',rng.uniform(.4,2,c));rq=put(prefix+'rq',rng.uniform(.2,1,c))
        pw=put(prefix+'pw',rng.normal(0,.15/np.sqrt(c),(c,c)));bias=rng.normal(0,.025,c);pb=put(prefix+'pb',bias);pb3=put(prefix+'pb3',bias.reshape(1,c,1))
        constants += [w,b,ap,rp,aq,rq,pw,pb]
        policy=dict(native_abi=1,channels=c,backend=2,row_batches=0,require_vforce=1)
        nodes.append(helper.make_node('SnakeF32',[value,ap,rp],[prefix+'pre'],name=prefix+'pre',domain=NATIVE,**policy))
        nodes.append(helper.make_node('CausalDW7SnakeF32',[prefix+'pre',w,b,aq,rq],[prefix+'post'],name=prefix+'post',domain=NATIVE,dilation=d,**policy))
        nodes.append(mat(prefix+'product',pw,prefix+'post',c,c))
        nodes.append(helper.make_node('Add',[prefix+'product',pb3],[prefix+'biased'],name=prefix+'biased'))
        nodes.append(helper.make_node('Add',[value,prefix+'biased'],[prefix+'out'],name=prefix+'out'));value=prefix+'out';outputs.append(value)
    ins=['B',256 if up else c,'T'];high=['B',c,'H' if up else 'T']
    model=helper.make_model(helper.make_graph(nodes,'unfused precision region',[helper.make_tensor_value_info('x',TensorProto.FLOAT,ins)],
        [helper.make_tensor_value_info(v,TensorProto.FLOAT,ins if up and i<2 else high) for i,v in enumerate(outputs)],initializers),
        opset_imports=[helper.make_opsetid('',20),helper.make_opsetid(NATIVE,1),helper.make_opsetid(MATRIX,1)],ir_version=10)
    return model,inputs+constants

def candidate(reference,constants,c,up,segments,debug=True):
    model=copy.deepcopy(reference);del model.graph.node[:]
    outputs=[v.name for v in model.graph.output]
    if not debug:
        final=copy.deepcopy(model.graph.output[-1]);del model.graph.output[:];model.graph.output.append(final);outputs=outputs[-1:]
    attrs=dict(native_abi=1,channels=c,tile_time=512,segments=segments,backend=2,matrix_mode=0,matrix_isa=128,precision_mode=8 if c>=128 else 0)
    if up:attrs.update(stride=2,projection_mode=8,projection_isa=0)
    domain=UP if up else STAGE;name=('UpsampleStage' if up else 'StageStack')+('DebugF32' if debug else 'F32')
    model.graph.node.append(helper.make_node(name,['x',*constants],outputs,name='fused',domain=domain,**attrs));model.opset_import.append(helper.make_opsetid(domain,1));return model

def session(model,a):
    if ort.__version__!='1.29.0':raise RuntimeError('ORT1.29 required')
    so=ort.SessionOptions();so.intra_op_num_threads=4;so.inter_op_num_threads=1;so.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL;so.log_severity_level=3
    for p in (a.native_library,a.core_ops,a.stage_library,a.upsample_library):so.register_custom_ops_library(str(p.resolve()))
    s=ort.InferenceSession(model.SerializeToString(),so,providers=['CPUExecutionProvider'])
    if s.get_providers()!=['CPUExecutionProvider']:raise RuntimeError('Unexpected provider')
    return s

def run_reference(s,x):
    if x.shape[0]==0:return None
    batches=[s.run(None,{'x':b[None]}) for b in x]
    return [np.concatenate([v[i] for v in batches],axis=0) for i in range(len(batches[0]))]

def compare(x,y,label,exact=False):
    if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():raise AssertionError(label+' shape/nonfinite')
    bits=bool(np.array_equal(x.view(np.uint32),y.view(np.uint32)))
    delta=np.abs(x.astype(np.float64)-y.astype(np.float64));maximum=float(delta.max(initial=0))
    if exact and not bits:raise AssertionError(label+' exact repeat failed')
    if not np.allclose(x,y,atol=1e-5,rtol=1e-4):raise AssertionError(f'{label}: maxabs {maximum}')
    return {'max_abs':maximum,'rmse':float(np.sqrt(np.mean(delta*delta))) if delta.size else 0.,'bitwise_equal':bits}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('native-library','core-ops','stage-library','upsample-library','output'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();report={'scope':'Focused correctness only, no performance measurements','gpu_used':False,'ORT':ort.__version__,'records':[],'extra':[],'libraries':{str(v):sha(v) for v in (a.native_library,a.core_ops,a.stage_library,a.upsample_library)}}
    lengths=(1,2,3,31,32,33,64,77,78,79,80,81,121,255,256,257,511,512,513)
    for c,up in ((256,False),(128,True),(64,False),(32,False)):
        ref,constants=fixture(c,up);rs=session(ref,a);cs={n:session(candidate(ref,constants,c,up,n),a) for n in (1,2,4)};ps=session(candidate(ref,constants,c,up,4,False),a)
        rng=np.random.default_rng(988+c)
        for t in lengths:
            x=rng.normal(0,.3,(1,256 if up else c,t)).astype(np.float32);expected=run_reference(rs,x);one=cs[1].run(None,{'x':x})
            for parts,s in cs.items():
                values=s.run(None,{'x':x});checks=[compare(v,w,f'C{c}/T{t}/seg{parts}/out{i}') for i,(v,w) in enumerate(zip(values,expected))]
                checks += [compare(v,w,'segmentation') for v,w in zip(values,one)]
                report['records'].append({'channels':c,'upsample':up,'time':t,'segments':parts,'checks':checks})
            compare(ps.run(None,{'x':x})[0],one[-1],'production')
        x=rng.normal(0,.3,(2,256 if up else c,121)).astype(np.float32);values=cs[4].run(None,{'x':x});expected=run_reference(rs,x)
        report['extra'].append({'channels':c,'kind':'batch2','checks':[compare(v,w,'batch2') for v,w in zip(values,expected)]})
        x=x[:1].copy();baseline=cs[4].run(None,{'x':x});future=x.copy();future[:,:,65:]+=1
        changed=cs[4].run(None,{'x':future})
        for i,(v,w) in enumerate(zip(baseline,changed)):
            cut=65 if up and i<2 else 130 if up else 65;compare(v[:,:,:cut],w[:,:,:cut],'future prefix',exact=True)
        compare(cs[4].run(None,{'x':x})[-1],baseline[-1],'repeat',exact=True)
        ys=[rng.normal(0,.3,(1,256 if up else c,t)).astype(np.float32) for t in (79,257,513)]
        serial=[cs[4].run(None,{'x':y})[-1] for y in ys]
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:parallel=list(pool.map(lambda y:cs[4].run(None,{'x':y})[-1],ys))
        report['extra'].append({'channels':c,'kind':'concurrency','checks':[compare(v,w,'concurrent',exact=True) for v,w in zip(serial,parallel)]})
        for b,t in ((0,0),(0,13),(1,0),(2,0)):
            z=np.empty((b,256 if up else c,t),np.float32);values=cs[4].run(None,{'x':z})
            if any(v.size for v in values):raise AssertionError('Empty output')
    report.update(status='passed',shape_segment_records=len(report['records']),failures=0)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('records','extra','libraries')},indent=2))
if __name__=='__main__':main()
