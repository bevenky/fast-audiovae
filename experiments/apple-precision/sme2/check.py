"""CPU-only exact arithmetic, lifetime, partition and ORT checks. No timing."""
import argparse,concurrent.futures,ctypes as ct,hashlib,json,os,platform
from pathlib import Path
for key in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'):os.environ[key]='-1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import numpy as np
P=ct.POINTER(ct.c_float);B=ct.POINTER(ct.c_int8);I=ct.POINTER(ct.c_int32)
def ptr(x):return x.ctypes.data_as(P)
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def exact(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype
    np.testing.assert_array_equal(a.view(np.uint8),b.view(np.uint8))
def scales(x,axis):
    mx=np.max(np.abs(x),axis=axis,keepdims=True);s=(mx/np.float32(127)).astype(np.float32)
    return np.where(mx==0,np.float32(1),np.where(s>0,s,mx)).astype(np.float32)
def reference(w,x):
    if not x.shape[1]:return np.zeros((len(w),0),np.float32)
    sw=scales(w,1);sx=scales(x,0)
    qw=np.rint(np.clip(w/sw,-127,127)).astype(np.int64)
    qx=np.rint(np.clip(x/sx,-127,127)).astype(np.int64)
    return ((qw@qx).astype(np.float32)*(sw*sx)).astype(np.float32)
class API:
    def __init__(self,path):
        self.lib=ct.CDLL(str(Path(path).resolve()))
        for name,args,result in [('abi',[],ct.c_int),('capabilities',[],ct.c_int),('last_error',[],ct.c_char_p),
          ('create',[ct.c_int,ct.c_int,P,ct.c_int,ct.c_int],ct.c_void_p),('prepare',[ct.c_void_p,P,ct.c_int],ct.c_void_p),
          ('run_rows',[ct.c_void_p,ct.c_void_p,P,ct.c_int,ct.c_int],ct.c_int),
          ('allocate_input',[ct.c_void_p,P,ct.c_int,ct.c_int],ct.c_void_p),('prepare_jobs',[ct.c_void_p],ct.c_int),('prepare_job',[ct.c_void_p,ct.c_int],ct.c_int),('check_packed_input',[ct.c_void_p],ct.c_int),('row_step',[ct.c_void_p],ct.c_int),
          ('destroy_plan',[ct.c_void_p],None),('destroy_input',[ct.c_void_p],None),
          ('inspect_input',[ct.c_void_p,ct.POINTER(B),ct.POINTER(P),ct.POINTER(I)],ct.c_int)]:
            fn=getattr(self.lib,'ipb_'+name);fn.argtypes=args;fn.restype=result;setattr(self,name,fn)
    def require(self,ok):
        if not ok:raise RuntimeError(self.last_error().decode())
    def run(self,w,x,backend,shards=1):
        w=np.ascontiguousarray(w,np.float32);x=np.ascontiguousarray(x,np.float32)
        m,k=w.shape;t=x.shape[1];plan=self.create(m,k,ptr(w),8,backend);self.require(plan);inp=None
        try:
            inp=self.allocate_input(plan,ptr(x),t,shards) if backend==3 else self.prepare(plan,ptr(x),t);self.require(inp)
            if backend==3:
                jobs=self.prepare_jobs(inp);self.require(jobs>0)
                with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
                    for status in pool.map(lambda j:self.prepare_job(inp,j),range(jobs)):self.require(status==0)
                self.require(self.check_packed_input(inp)==0)
            q=B();s=P();sums=I();self.require(self.inspect_input(inp,ct.byref(q),ct.byref(s),ct.byref(sums))==0)
            prep=[np.ctypeslib.as_array(q,shape=(k*t,)).copy() if t else np.empty(0,np.int8),
                  np.ctypeslib.as_array(s,shape=(t,)).copy() if t else np.empty(0,np.float32),
                  np.ctypeslib.as_array(sums,shape=(t,)).copy() if t else np.empty(0,np.int32)]
            y=np.full((m,t),np.nan,np.float32)
            def job(j):self.require(self.run_rows(plan,inp,ptr(y),m*j//shards,m*(j+1)//shards)==0)
            if shards==1:job(0)
            else:
                with concurrent.futures.ThreadPoolExecutor(shards) as pool:list(pool.map(job,range(shards)))
            return y,prep
        finally:
            if inp:self.destroy_input(inp)
            self.destroy_plan(plan)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh result required')
    build=json.loads(a.build.read_text());api=API(build['core_path']);assert api.abi()==1 and api.capabilities()&16
    rng=np.random.default_rng(20260908);rows=[]
    shapes=[(1,1,0),(1,1,1),(3,5,7),(4,4,8),(5,7,9),(17,31,17),(33,32,33),
            (64,65,65),(128,128,127),(129,129,129),(256,256,65),(17,257,257),(3,16384,9)]
    cases=[]
    for m,k,t in shapes:
        w=rng.normal(0,.2,(m,k)).astype(np.float32);x=rng.normal(0,.2,(k,t)).astype(np.float32)
        w*=np.linspace(.2,3,m,dtype=np.float32)[:,None]
        if t:x*=np.linspace(.1,5,t,dtype=np.float32)[None,:]
        cases.append(('random',w,x))
    for sw,sx in ((1,1),(1,-1),(-1,1),(-1,-1)):
        cases.append(('full_range',np.full((5,16384),sw*127,np.float32),np.full((16384,9),sx*127,np.float32)))
    edge=np.array([0.,-0.,.5,1.5,2.5,-.5,-1.5,-2.5,126.5,-126.5,127,-127],np.float32)
    cases.append(('half_ties',np.tile(edge,(5,1)),np.tile(edge,(12,1))))
    tiny=np.array([0,1,2,3,0x007fffff,0x00800000,0x80800000,0x80000001],np.uint32).view(np.float32)
    cases.append(('subnormal',np.ones((5,8),np.float32),np.tile(tiny,(8,1))))
    cases.append(('zero',np.zeros((5,7),np.float32),np.zeros((7,9),np.float32)))
    for name,w,x in cases:
        before_w=w.copy();before_x=x.copy();base,bprep=api.run(w,x,0);wanted=reference(w,x)
        exact(base,wanted)
        for shards in (1,2,3):
            got,prep=api.run(w,x,3,shards);exact(got,base)
            for actual,expected in zip(prep,bprep):exact(actual,expected)
        exact(w,before_w);exact(x,before_x)
        if x.shape[1]>1:
            cut=x.shape[1]//2;future=x.copy();future[:,cut:]*=-23
            changed,_=api.run(w,future,3,3);exact(changed[:,:cut].copy(),base[:,:cut].copy())
            prefix,_=api.run(w,x[:,:cut],3,2);exact(prefix,base[:,:cut].copy())
        rows.append({'case':name,'M':w.shape[0],'K':w.shape[1],'T':x.shape[1],
                     'exact_bytes_scales_sums_output':True,'shards':[1,2,3]})
    # Concurrent different lengths and contents share the same immutable plan.
    w=rng.normal(size=(17,31)).astype(np.float32);plan=api.create(17,31,ptr(w),8,3);api.require(plan)
    def task(t):
        x=(np.arange(31*t,dtype=np.float32).reshape(31,t)-t)/np.float32(97)
        inp=api.prepare(plan,ptr(x),t);api.require(inp)
        try:
            y=np.empty((17,t),np.float32);api.require(api.run_rows(plan,inp,ptr(y),0,17)==0);exact(y,reference(w,x))
        finally:api.destroy_input(inp)
    try:
        with concurrent.futures.ThreadPoolExecutor(4) as pool:list(pool.map(task,[1,3,8,17,33,129]))
    finally:api.destroy_plan(plan)
    invalid=0;w=np.ones((2,2),np.float32);x=np.ones((2,9),np.float32)
    for m,k,mode,backend in [(0,2,8,3),(2,0,8,3),(2,16385,8,3),(2,2,7,3),(2,2,16,3),(2,2,8,1),(2,2,8,99)]:
        assert not api.create(m,k,ptr(w),mode,backend);invalid+=1
    for bad in (np.nan,np.inf,-np.inf):
        wrong=np.full((2,2),bad,np.float32);assert not api.create(2,2,ptr(wrong),8,3);invalid+=1
    plan=api.create(2,2,ptr(w),8,3);api.require(plan)
    try:
        assert not api.prepare(plan,ptr(x),-1);invalid+=1
        for bad in (np.nan,np.inf):
            wrong=x.copy();wrong[0,0]=bad;assert not api.prepare(plan,ptr(wrong),9);invalid+=1
        inp=api.prepare(plan,ptr(x),9);api.require(inp)
        try:
            y=np.empty((2,9),np.float32)
            for lo,hi in [(-1,2),(0,3),(2,1)]:assert api.run_rows(plan,inp,ptr(y),lo,hi)==-1;invalid+=1
            assert api.run_rows(plan,inp,ptr(x),0,2)==-1;invalid+=1
        finally:api.destroy_input(inp)
    finally:api.destroy_plan(plan)
    import onnx,onnxruntime as ort
    from onnx import helper as h,numpy_helper as n
    assert ort.__version__=='1.29.0' and onnx.__version__=='1.22.0'
    w=rng.normal(size=(17,31)).astype(np.float32)
    node=h.make_node('PrecisionMatMulF32',['w','x'],['y'],domain=build['domain'],native_abi=1,M=17,K=31,precision_mode=8,backend=3,shards=3)
    graph=h.make_graph([node],'micro',[h.make_tensor_value_info('x',1,[1,31,'T'])],
        [h.make_tensor_value_info('y',1,[1,17,'T'])],[n.from_array(w,'w')])
    model=h.make_model(graph,opset_imports=[h.make_opsetid('',18),h.make_opsetid(build['domain'],1)]);model.ir_version=10
    options=ort.SessionOptions();options.intra_op_num_threads=3;options.inter_op_num_threads=1
    options.add_session_config_entry('session.intra_op.allow_spinning','0');options.register_custom_ops_library(build['ops_path'])
    session=ort.InferenceSession(model.SerializeToString(),options,providers=['CPUExecutionProvider'])
    assert session.get_providers()==['CPUExecutionProvider']
    for t in (0,1,3,7,8,9,17,129):
        x=rng.normal(size=(1,31,t)).astype(np.float32);exact(session.run(None,{'x':x})[0][0],reference(w,x[0]))
    result={'status':'passed','GPU_used':False,'timings_collected':False,'backend':3,'capabilities':api.capabilities(),
            'native_shapes':rows,'invalid_cases':invalid,'concurrent_shared_plan_cases':6,'ORT_micro_cases':8,
            'ORT_version':ort.__version__,'providers':session.get_providers(),'numpy':np.__version__,
            'platform':platform.platform(),'build_sha256':sha(a.build),'script_sha256':sha(__file__),
            'library_sha256':{Path(build[k]).name:sha(build[k]) for k in ('core_path','ops_path')}}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='native_shapes'},indent=2))
if __name__=='__main__':main()
