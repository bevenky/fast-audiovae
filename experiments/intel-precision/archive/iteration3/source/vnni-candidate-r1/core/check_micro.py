"""Focused CPU numerical/safety checks. No timing or model/weight downloads."""
import argparse
import concurrent.futures
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import platform

for key in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'):
    os.environ[key]='-1'
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import numpy as np

P=ct.POINTER(ct.c_float)

def pointer(a):return a.ctypes.data_as(P)
def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

class API:
    def __init__(self,path):
        self.lib=ct.CDLL(str(Path(path).resolve()))
        for name,args,result in [
            ('ip_abi',[],ct.c_int),('ip_capabilities',[],ct.c_int),('ip_last_error',[],ct.c_char_p),
            ('ip_create',[ct.c_int,ct.c_int,P,ct.c_int,ct.c_int],ct.c_void_p),
            ('ip_prepare',[ct.c_void_p,P,ct.c_int],ct.c_void_p),
            ('ip_run_rows',[ct.c_void_p,ct.c_void_p,P,ct.c_int,ct.c_int],ct.c_int),
            ('ip_destroy_plan',[ct.c_void_p],None),('ip_destroy_input',[ct.c_void_p],None),
            ('ip_plan_bytes',[ct.c_void_p],ct.c_size_t),('ip_input_bytes',[ct.c_void_p],ct.c_size_t)]:
            fn=getattr(self.lib,name);fn.argtypes=args;fn.restype=result
        assert self.lib.ip_abi()==1
    def error(self):return self.lib.ip_last_error().decode()
    def run(self,w,x,mode,backend,shards=1):
        w=np.ascontiguousarray(w,dtype=np.float32);x=np.ascontiguousarray(x,dtype=np.float32)
        m,k=w.shape;t=x.shape[1];assert x.shape[0]==k
        plan=self.lib.ip_create(m,k,pointer(w),mode,backend)
        if not plan:raise RuntimeError(self.error())
        inp=None
        try:
            inp=self.lib.ip_prepare(plan,pointer(x),t)
            if not inp:raise RuntimeError(self.error())
            y=np.full((m,t),np.nan,dtype=np.float32)
            def run_rows(j):
                status=self.lib.ip_run_rows(plan,inp,pointer(y),m*j//shards,m*(j+1)//shards)
                if status:raise RuntimeError(self.error())
            if shards==1:run_rows(0)
            else:
                with concurrent.futures.ThreadPoolExecutor(shards) as pool:list(pool.map(run_rows,range(shards)))
            return y,{'plan_bytes':self.lib.ip_plan_bytes(plan),'input_bytes':self.lib.ip_input_bytes(inp)}
        finally:
            if inp:self.lib.ip_destroy_input(inp)
            self.lib.ip_destroy_plan(plan)

def scales(a,axis):
    mx=np.max(np.abs(a),axis=axis,keepdims=True)
    s=(mx/np.float32(127)).astype(np.float32)
    return np.where(mx==0,np.float32(1),np.where(s>0,s,mx)).astype(np.float32)

def reference(w,x,mode):
    if x.shape[1]==0:return np.zeros((w.shape[0],0),np.float32)
    if mode==16:
        return np.einsum('mk,kt->mt',w.astype(np.float16).astype(np.float64),
                         x.astype(np.float16).astype(np.float64),optimize=False).astype(np.float32)
    sw=scales(w,1);sx=scales(x,0)
    qw=np.rint(np.clip(w/sw,-127,127)).astype(np.int64)
    qx=np.rint(np.clip(x/sx,-127,127)).astype(np.int64)
    return ((qw@qx).astype(np.float32)*(sw*sx)).astype(np.float32)

def compare(got,want,mode):
    assert got.shape==want.shape and np.isfinite(got).all()
    if mode==8:
        if not np.array_equal(got,want):
            raise AssertionError('INT8 formula differs, max='+str(float(np.max(np.abs(got-want)))))
    else:np.testing.assert_allclose(got,want,rtol=2e-5,atol=2e-5)

def run_checks(api,backend):
    rng=np.random.default_rng(20260908);rows=[]
    shapes=[(1,1,0),(1,1,1),(3,5,7),(17,31,16),(32,32,33),(64,64,65),
            (128,128,127),(129,129,129),(129,257,513),(3,2048,17),(257,64,1)]
    for mode in (8,16):
        for m,k,t in shapes:
            w=(rng.normal(size=(m,k))*.2).astype(np.float32)
            x=(rng.normal(size=(k,t))*.2).astype(np.float32)
            # Unequal row/column ranges expose accidental scale axes.
            w*=np.linspace(.2,3,m,dtype=np.float32)[:,None]
            if t:x*=np.linspace(.1,5,t,dtype=np.float32)[None,:]
            before_w=w.copy();before_x=x.copy();wanted=reference(w,x,mode)
            y,bytes_=api.run(w,x,mode,backend,2)
            compare(y,wanted,mode);assert np.array_equal(w,before_w) and np.array_equal(x,before_x)
            repeat,_=api.run(w,x,mode,backend,1);compare(repeat,y,mode)
            if t>1:
                cut=t//2;future=x.copy();future[:,cut:]*=-23
                # Hold the output-row partition fixed when perturbing future
                # samples. SGEMM can use a different FP32 reduction for M1+M2
                # versus M3; that is tested numerically by compare above.
                for shards,baseline in ((1,repeat),(2,y)):
                    changed,_=api.run(w,future,mode,backend,shards)
                    np.testing.assert_array_equal(changed[:,:cut],baseline[:,:cut],
                                                  err_msg=f'future dependency mode={mode},shape={(m,k,t)},shards={shards}')
                prefix,_=api.run(w,x[:,:cut],mode,backend)
                compare(prefix,repeat[:,:cut],mode)
            delta=np.abs(repeat-y)
            rows.append({'mode':mode,'M':m,'K':k,'T':t,**bytes_,'passed':True,
                         'cross_shard_mismatches':int(np.count_nonzero(repeat!=y)),
                         'cross_shard_max_abs':float(np.max(delta,initial=0)),
                         'strict_future_test_shards':[1,2] if t>1 else []})
    # VNNI full-range cases expose U8/S8 sign reversal and INT16 saturation.
    for sign_w,sign_x in ((1,1),(1,-1),(-1,1),(-1,-1)):
        w=np.full((16,128),sign_w*127,dtype=np.float32);x=np.full((128,17),sign_x*127,dtype=np.float32)
        y,_=api.run(w,x,8,backend);compare(y,reference(w,x,8),8)
    # Exact zeros, including a zero weight row and zero activation column.
    w=np.zeros((3,7),np.float32);x=np.zeros((7,5),np.float32)
    w[1]=[-3,-2,-1,0,1,2,3];x[:,2]=[-3,-2,-1,0,1,2,3]
    y,_=api.run(w,x,8,backend);compare(y,reference(w,x,8),8)
    # Half ties, subnormals, signed zeros and largest finite values. Identity
    # multiplication makes operand rounding directly observable without a
    # floating-point reduction tolerance masking a conversion error.
    edge=np.array([0.,-0.,2**-25,3*2**-25,2**-24,2**-14,1+2**-11,1+3*2**-11,
                   -1-2**-11,-1-3*2**-11,65504.,-65504.,65519.,-65519.,.33333334,-.33333334],np.float32)
    w=np.eye(8,dtype=np.float32);x=np.tile(edge,(8,1))
    y,_=api.run(w,x,16,backend)
    np.testing.assert_array_equal(y,x.astype(np.float16).astype(np.float32))
    # Shared quantized input across different weight matrices AND output sizes.
    w=rng.normal(size=(5,11)).astype(np.float32);w2=rng.normal(size=(7,11)).astype(np.float32)
    x=rng.normal(size=(11,19)).astype(np.float32)
    for mode in (8,16):
        p=api.lib.ip_create(5,11,pointer(w),mode,backend);p2=api.lib.ip_create(7,11,pointer(w2),mode,backend)
        assert p and p2,api.error();q=api.lib.ip_prepare(p,pointer(x),19);assert q,api.error()
        try:
            y=np.empty((7,19),np.float32)
            assert api.lib.ip_run_rows(p2,q,pointer(y),0,7)==0,api.error()
            compare(y,reference(w2,x,mode),mode)
        finally:api.lib.ip_destroy_input(q);api.lib.ip_destroy_plan(p);api.lib.ip_destroy_plan(p2)
    # Multiple concurrent prepared inputs share one immutable plan.
    w=rng.normal(size=(17,31)).astype(np.float32)
    for mode in (8,16):
        p=api.lib.ip_create(17,31,pointer(w),mode,backend);assert p,api.error()
        def invocation(t):
            x=np.arange(31*t,dtype=np.float32).reshape(31,t)/1234
            q=api.lib.ip_prepare(p,pointer(x),t);assert q,api.error()
            try:
                y=np.empty((17,t),np.float32);assert api.lib.ip_run_rows(p,q,pointer(y),0,17)==0,api.error()
                compare(y,reference(w,x,mode),mode)
            finally:api.lib.ip_destroy_input(q)
        try:
            with concurrent.futures.ThreadPoolExecutor(4) as pool:list(pool.map(invocation,[1,15,33,65,3,129]))
        finally:api.lib.ip_destroy_plan(p)
    invalid=0;w=np.ones((2,2),np.float32);x=np.ones((2,4),np.float32)
    for m,k,mode,b in [(0,2,8,backend),(2,0,8,backend),(2,16385,8,backend),(2,2,7,backend),(2,2,8,99)]:
        assert not api.lib.ip_create(m,k,pointer(w),mode,b);invalid+=1
    for mode,bad in [(8,np.nan),(8,np.inf),(16,np.nan),(16,65520.)]:
        wrong=np.full((2,2),bad,np.float32);assert not api.lib.ip_create(2,2,pointer(wrong),mode,backend);invalid+=1
    p=api.lib.ip_create(2,2,pointer(w),8,backend);assert p
    try:
        assert not api.lib.ip_prepare(p,pointer(x),-1);invalid+=1
        bad=x.copy();bad[0,0]=np.nan;assert not api.lib.ip_prepare(p,pointer(bad),4);invalid+=1
        q=api.lib.ip_prepare(p,pointer(x),4);assert q
        try:
            assert api.lib.ip_run_rows(p,q,pointer(x),0,2)<0;invalid+=1
            y=np.empty((2,4),np.float32)
            for first,last in [(-1,2),(0,3),(2,1)]:
                assert api.lib.ip_run_rows(p,q,pointer(y),first,last)<0;invalid+=1
        finally:api.lib.ip_destroy_input(q)
        storage=np.ones(24,np.float32);partial=storage[:8].reshape(2,4)
        q=api.lib.ip_prepare(p,pointer(partial),4);assert q
        try:
            assert api.lib.ip_run_rows(p,q,pointer(storage[1:9].reshape(2,4)),0,2)<0;invalid+=1
        finally:api.lib.ip_destroy_input(q)
    finally:api.lib.ip_destroy_plan(p)
    return {'shapes':rows,'invalid_cases':invalid,'full_range_signedness_cases':4,
            'causal_future_perturbation':True,'prefix_shapes':True,'shared_projection_input':True,
            'concurrent_different_inputs':True,'half_conversion_edges':16,'GPU_used':False,'timings_collected':False}

def ort_checks(api,ops,backend):
    import onnx
    import onnxruntime as ort
    assert ort.__version__=='1.29.0'
    from onnx import helper as h,numpy_helper as n
    rng=np.random.default_rng(6);checked=[]
    for mode in (8,16):
        w=rng.normal(size=(17,31)).astype(np.float32)
        node=h.make_node('PrecisionMatMulF32',['w','x'],['y'],domain='fast.audiovae.precision.matrix.experimental',
                         native_abi=1,M=17,K=31,precision_mode=mode,backend=backend,shards=2)
        graph=h.make_graph([node],'micro',[h.make_tensor_value_info('x',1,[1,31,'T'])],[h.make_tensor_value_info('y',1,[1,17,'T'])],[n.from_array(w,'w')])
        model=h.make_model(graph,opset_imports=[h.make_opsetid('',18),h.make_opsetid(node.domain,1)]);model.ir_version=10
        onnx.checker.check_model(model);options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        options.register_custom_ops_library(str(Path(ops).resolve()))
        session=ort.InferenceSession(model.SerializeToString(),options,providers=['CPUExecutionProvider'])
        assert session.get_providers()==['CPUExecutionProvider']
        for t in (0,1,17,65,129,3):
            x=rng.normal(size=(1,31,t)).astype(np.float32)
            y=session.run(None,{'x':x})[0][0]
            want,_=api.run(w,x[0],mode,backend);compare(y,want,mode);checked.append({'mode':mode,'T':t})
    return {'onnxruntime':ort.__version__,'onnx':onnx.__version__,'shapes':checked,'CPU_only':True}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library',type=Path,required=True)
    p.add_argument('--backend',type=int,choices=(0,1),required=True);p.add_argument('--ops',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Output exists')
    api=API(a.library);result=run_checks(api,a.backend)
    if a.ops:result['ORT_micro']=ort_checks(api,a.ops,a.backend)
    result.update(status='passed',backend=a.backend,capabilities=api.lib.ip_capabilities(),
                  library_sha256=sha(a.library),script_sha256=sha(__file__),platform=platform.platform(),numpy=np.__version__)
    result['MKL_dispatch_environment']={key:os.environ.get(key) for key in ('MKL_ENABLE_INSTRUCTIONS','MKL_CBWR')}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'passed','shapes':len(result['shapes']),'invalid_cases':result['invalid_cases'],
                      'backend':a.backend,'capabilities':result['capabilities'],'output':str(a.output)},indent=2))

if __name__=='__main__':main()
