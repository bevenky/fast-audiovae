"""Check the paired projection arithmetic against the accepted CPU matrix operator."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):os.environ[key]='1'
os.environ.update(CUDA_VISIBLE_DEVICES='-1',NVIDIA_VISIBLE_DEVICES='void',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1')
import argparse,json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper as h,numpy_helper as nh

def exact(a,b):return a.shape==b.shape and np.array_equal(a.view(np.uint32),b.view(np.uint32))

def session(model,libraries):
    opt=ort.SessionOptions();opt.intra_op_num_threads=1;opt.inter_op_num_threads=1;opt.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    for lib in libraries:opt.register_custom_ops_library(str(lib))
    result=ort.InferenceSession(model.SerializeToString(),opt,providers=['CPUExecutionProvider']);result.disable_fallback()
    assert result.get_providers()==['CPUExecutionProvider'];return result

def micro(base,candidate):
    rng=np.random.default_rng(71923);m,k=8192,2048
    # Exercise all weight ranges/rounding ties while giving two distinct outputs.
    wa=rng.standard_normal((m,k),dtype=np.float32)*np.float32(.3)
    wb=rng.standard_normal((m,k),dtype=np.float32)*np.float32(.1)
    wa[0]=0;wb[0]=np.nextafter(np.float32(0),np.float32(1))
    wa[1]=np.resize(np.array([-127,-126.5,-2.5,-1.5,-.5,0,.5,1.5,2.5,126.5,127],np.float32),k)
    wb[1]=-wa[1]
    init=[nh.from_array(wa,'wa'),nh.from_array(wb,'wb')]
    x=h.make_tensor_value_info('x',onnx.TensorProto.FLOAT,[1,k,'T']);outputs=[h.make_tensor_value_info(n,onnx.TensorProto.FLOAT,[1,m,'T']) for n in ('ya','yb')]
    attrs=dict(native_abi=1,M=m,K=k,precision_mode=8,backend=1,shards=1)
    oldnodes=[h.make_node('PrecisionMatMulF32',[w,'x'],[y],domain='fast.audiovae.precision.matrix.experimental',**attrs) for w,y in [('wa','ya'),('wb','yb')]]
    newnodes=[h.make_node('PackedProjectionPairF32',['wa','wb','x'],['ya','yb'],domain='fast.audiovae.streaming.matrix.experimental',native_abi=1,M=m,K=k)]
    def model(nodes,domain):
        result=h.make_model(h.make_graph(nodes,'micro',[x],outputs,init),opset_imports=[h.make_opsetid('',18),h.make_opsetid(domain,1)]);result.ir_version=10;return result
    old=session(model(oldnodes,'fast.audiovae.precision.matrix.experimental'),[base/'libs/precision_ops.so'])
    new=session(model(newnodes,'fast.audiovae.streaming.matrix.experimental'),[candidate/'libs/libpaired_projection.so'])
    cases=[]
    for t in [0,1,2,3,4,5,17]:
        for pattern in ['random','zero','ties','subnormal','extrema']:
            v=rng.standard_normal((1,k,t),dtype=np.float32)
            if pattern=='zero':v.fill(0)
            elif pattern=='ties':v[:]=np.resize(wa[1],v.shape)
            elif pattern=='subnormal':v[:]=np.resize(np.array([0,1,-1,127,-127],np.float32)*np.nextafter(np.float32(0),np.float32(1)),v.shape)
            elif pattern=='extrema':v[:]=np.resize(np.array([-127,127],np.float32),v.shape)
            expected=old.run(None,{'x':v});actual=new.run(None,{'x':v})
            ok=all(exact(a,b) for a,b in zip(expected,actual));assert ok,(t,pattern)
            cases.append({'frames':t,'pattern':pattern,'passed':True,'bitwise_exact':True})
    return cases

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise RuntimeError('Fresh evidence path required')
    os.sched_setaffinity(0,{0});assert ort.__version__=='1.29.0'
    result={'scope':'micro','threads':1,'providers':['CPUExecutionProvider'],'gpu_used':False,'status':'running'}
    try:
        result['checks']=micro(a.baseline,a.candidate)
        result.update(status='passed',check_count=len(result['checks']))
    except Exception as e:result.update(status='failed',error=repr(e));raise
    finally:a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
if __name__=='__main__':main()
