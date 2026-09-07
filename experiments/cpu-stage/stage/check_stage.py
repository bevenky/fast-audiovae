"""CPU-only correctness checks for the isolated stage pipeline. No timing loop.

Execute on the target host, after building the candidate. Reports each of the
three intermediate residual outputs against original native operators + ORT
MatMul, and segmented execution against one uninterrupted pipeline. All quality
gates are correctness screens; full decoder clips still need separate checks.
"""
import argparse
import concurrent.futures
import copy
import ctypes
import json
import os
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='-1',NVIDIA_VISIBLE_DEVICES='void',
                  HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1')
import numpy as np
import onnx
from onnx import TensorProto,helper,numpy_helper

DOMAIN='fast.audiovae.stage.experimental'
NATIVE='venky.audio.cpu.portable'

def fixture(c,backend=4,seed=501):
    rng=np.random.default_rng(seed+c);initializers=[];nodes=[];value='x';constants=[];outputs=[]
    def put(name,a):
        initializers.append(numpy_helper.from_array(np.asarray(a,np.float32),name));return name
    for u,d in enumerate((1,3,9)):
        prefix=f'u{u}_';w=put(prefix+'dw',rng.normal(0,.15,(c,1,7)));b=put(prefix+'db',rng.normal(0,.05,c))
        ap=put(prefix+'ap',rng.uniform(.4,2,c));rp=put(prefix+'rp',rng.uniform(.2,1,c))
        aq=put(prefix+'aq',rng.uniform(.4,2,c));rq=put(prefix+'rq',rng.uniform(.2,1,c))
        pw=put(prefix+'pw',rng.normal(0,.2/np.sqrt(c),(c,c)));pb=put(prefix+'pb',rng.normal(0,.05,c))
        constants.extend([w,b,ap,rp,aq,rq,pw,pb])
        policy=dict(channels=c,native_abi=1,row_batches=0,backend=backend,require_vector_sine=1)
        nodes.append(helper.make_node('SnakeF32',[value,ap,rp],[prefix+'pre'],name=prefix+'presnake',domain=NATIVE,**policy))
        nodes.append(helper.make_node('CausalDW7SnakeF32',[prefix+'pre',w,b,aq,rq],[prefix+'post'],name=prefix+'dwpost',domain=NATIVE,dilation=d,**policy))
        nodes.append(helper.make_node('MatMul',[pw,prefix+'post'],[prefix+'product'],name=prefix+'mm'))
        # Broadcast shape is explicit for the graph rewrite proof.
        bias3=put(prefix+'pb3',np.asarray(numpy_helper.to_array(initializers[-1])).reshape(1,c,1))
        nodes.append(helper.make_node('Add',[prefix+'product',bias3],[prefix+'biased'],name=prefix+'bias'))
        nodes.append(helper.make_node('Add',[value,prefix+'biased'],[prefix+'output'],name=prefix+'residual'))
        value=prefix+'output';outputs.append(value)
    dims=['B',c,'T']
    model=helper.make_model(helper.make_graph(nodes,'stage reference',[helper.make_tensor_value_info('x',TensorProto.FLOAT,dims)],
                    [helper.make_tensor_value_info(v,TensorProto.FLOAT,dims) for v in outputs],initializers),
                    opset_imports=[helper.make_opsetid('',20),helper.make_opsetid(NATIVE,1)],ir_version=10)
    return model,constants

def candidate(reference,constants,c,q,segments,backend,mode,isa,debug=True):
    m=copy.deepcopy(reference);del m.graph.node[:]
    outs=[v.name for v in m.graph.output]
    if not debug:
        keep=copy.deepcopy(m.graph.output[-1]);del m.graph.output[:];m.graph.output.append(keep);outs=outs[-1:]
    m.graph.node.append(helper.make_node('StageStackDebugF32' if debug else 'StageStackF32',['x',*constants],outs,name='stage',domain=DOMAIN,
                  channels=c,tile_time=q,segments=segments,backend=backend,matrix_mode=mode,matrix_isa=isa,native_abi=1))
    m.opset_import.append(helper.make_opsetid(DOMAIN,1));return m

def session(model,native,stage):
    import onnxruntime as ort
    if ort.__version__!='1.29.0':raise RuntimeError('This test requires ORT1.29.0')
    so=ort.SessionOptions();so.intra_op_num_threads=2;so.inter_op_num_threads=1
    so.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;so.log_severity_level=3
    so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.register_custom_ops_library(str(native));so.register_custom_ops_library(str(stage))
    s=ort.InferenceSession(model.SerializeToString(),so,providers=['CPUExecutionProvider'])
    if s.get_providers()!=['CPUExecutionProvider']:raise RuntimeError('Unexpected execution provider')
    return s

def compare(a,b,label,atol=2e-5,rtol=2e-5):
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise AssertionError(label+' shape/nonfinite')
    delta=np.abs(a.astype(np.float64)-b.astype(np.float64));maximum=float(delta.max(initial=0))
    rmse=float(np.sqrt(np.mean(delta*delta))) if delta.size else 0.
    if not np.allclose(a,b,atol=atol,rtol=rtol):raise AssertionError(f'{label}: maxabs={maximum} rmse={rmse}')
    return {'max_abs':maximum,'rmse':rmse,'bitwise_equal':bool(np.array_equal(a.view(np.uint32),b.view(np.uint32)))}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native-library',required=True,type=Path)
    p.add_argument('--stage-library',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--channels',default='32,256');p.add_argument('--tiles',default='64,128,256')
    p.add_argument('--backend',type=int,choices=(4,5),default=4);p.add_argument('--matrix-mode',type=int,default=0)
    p.add_argument('--matrix-isa',type=int,default=256)
    a=p.parse_args();native=a.native_library.resolve();stage=a.stage_library.resolve()
    lib=ctypes.CDLL(str(native));lib.ncc_backend_available.argtypes=[ctypes.c_int32]
    if not lib.ncc_backend_available(a.backend):raise RuntimeError('Requested CPU backend unavailable')
    records=[];rng=np.random.default_rng(7123);safety=0
    for c in [int(v) for v in a.channels.split(',')]:
        ref,const=fixture(c,a.backend);base=session(ref,native,stage)
        for q in [int(v) for v in a.tiles.split(',')]:
            def make(parts,debug=True):return candidate(ref,const,c,q,parts,a.backend,a.matrix_mode,a.matrix_isa,debug)
            whole=session(make(1),native,stage);split=session(make(2),native,stage);prod=session(make(2,False),native,stage)
            lengths=sorted({0,1,53,54,55,77,78,79,q-1,q,q+1,2*q-1,2*q,2*q+1,777})
            for n in lengths:
                x=rng.normal(0,.35,(2,c,n)).astype(np.float32);original=x.copy()
                expected=base.run(None,{'x':x});all_at_once=whole.run(None,{'x':x});segmented=split.run(None,{'x':x})
                for u in range(3):
                    records.append({'c':c,'q':q,'time':n,'unit':u,'reference':compare(segmented[u],expected[u],'ORT reference'),
                                    'segmentation':compare(segmented[u],all_at_once[u],'segmentation',2e-6,2e-6)})
                compare(prod.run(None,{'x':x})[0],segmented[-1],'production vs debug',0,0)
                if not np.array_equal(x,original):raise AssertionError('Input mutation')
            # Prefix independence, repeated calls and interleaved sessions.
            x=rng.normal(0,.3,(1,c,513)).astype(np.float32);changed=x.copy();changed[:,:,259:]+=rng.normal(0,.7,changed[:,:,259:].shape).astype(np.float32)
            y=split.run(None,{'x':x});z=split.run(None,{'x':changed})
            for u in range(3):compare(y[u][:,:,:259],z[u][:,:,:259],'causality',0,0)
            compare(split.run(None,{'x':x})[-1],y[-1],'reentrant repeat',0,0)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                parallel_outputs=list(pool.map(lambda _:split.run(None,{'x':x})[-1],range(4)))
            for value in parallel_outputs:compare(value,y[-1],'concurrent inference',0,0)
            empty=prod.run(None,{'x':np.empty((0,c,55),np.float32)})[0]
            if empty.shape!=(0,c,55):raise AssertionError('Empty batch')
            for attr,value in [('tile_time',65),('segments',0),('backend',3),('matrix_mode',4)]:
                invalid=make(2)
                for item in invalid.graph.node[0].attribute:
                    if item.name==attr:item.i=value
                try:session(invalid,native,stage)
                except Exception:safety+=1
                else:raise AssertionError('Malformed attributes accepted: '+attr)
    report={'records':records,'record_count':len(records),'malformed_cases_rejected':safety,
            'backend':a.backend,'matrix_mode':a.matrix_mode,'matrix_isa':a.matrix_isa,
            'runtime':'1.29.0','execution_providers':['CPUExecutionProvider'],'gpu_used':False,
            'timing_benchmark':False,'full_decoder_quality_validated':False}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))

if __name__=='__main__':main()
