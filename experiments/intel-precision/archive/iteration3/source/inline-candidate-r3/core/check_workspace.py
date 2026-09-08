"""CPU-only INT8 workspace/parity checks. No timing or model execution."""
import argparse
import concurrent.futures
import ctypes as ct
import json
from pathlib import Path
import numpy as np
from check_micro import API as BaseAPI,P,pointer,sha

class API(BaseAPI):
    def __init__(self,path,prefix='ip_',workspace=False):
        self.lib=ct.CDLL(str(Path(path).resolve()))
        signatures=[
            ('abi',[],ct.c_int),('capabilities',[],ct.c_int),('last_error',[],ct.c_char_p),
            ('create',[ct.c_int,ct.c_int,P,ct.c_int,ct.c_int],ct.c_void_p),
            ('prepare',[ct.c_void_p,P,ct.c_int],ct.c_void_p),
            ('run_rows',[ct.c_void_p,ct.c_void_p,P,ct.c_int,ct.c_int],ct.c_int),
            ('destroy_plan',[ct.c_void_p],None),('destroy_input',[ct.c_void_p],None),
            ('plan_bytes',[ct.c_void_p],ct.c_size_t),('input_bytes',[ct.c_void_p],ct.c_size_t)]
        if workspace:signatures += [
            ('workspace_create',[ct.c_int,ct.c_int,ct.c_int],ct.c_void_p),
            ('workspace_prepare',[ct.c_void_p,ct.c_void_p,P,ct.c_int],ct.c_int),
            ('workspace_run_rows',[ct.c_void_p,ct.c_void_p,P,ct.c_int,ct.c_int,P,P],ct.c_int),
            ('destroy_workspace',[ct.c_void_p],None),('workspace_bytes',[ct.c_void_p],ct.c_size_t)]
        for name,args,result in signatures:
            fn=getattr(self.lib,prefix+name);fn.argtypes=args;fn.restype=result
            setattr(self.lib,'ip_'+name,fn)
        assert self.lib.ip_abi()==1

def bits(got,wanted):
    assert got.shape==wanted.shape and got.dtype==wanted.dtype==np.float32
    np.testing.assert_array_equal(got.view(np.uint32),wanted.view(np.uint32))

def ordered(raw,bias,skip):
    return np.add(skip,np.add(raw,bias[:,None],dtype=np.float32),dtype=np.float32)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--reference-library',type=Path,required=True)
    p.add_argument('--symbol-prefix',choices=['ip_','ip3_'],default='ip3_')
    p.add_argument('--backend',type=int,choices=[0,1],default=1)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Output exists')
    api=API(a.library,a.symbol_prefix,True);ref=API(a.reference_library)
    rng=np.random.default_rng(20260910);records=[]
    workspace=api.lib.ip_workspace_create(257,513,257);assert workspace,api.error()
    allocation=api.lib.ip_workspace_bytes(workspace)
    shapes=[(1,1,0),(1,1,1),(3,5,7),(17,31,17),(32,32,256),(64,64,255),
            (128,128,128),(256,256,64),(256,256,256),(129,257,513),(513,128,129),
            (256,256,1),(128,128,3),(17,31,17)]
    try:
        for m,k,t in shapes:
            w=rng.normal(0,.2,(m,k)).astype(np.float32);x=rng.normal(0,.2,(k,t)).astype(np.float32)
            bias=rng.normal(0,.2,m).astype(np.float32);skip=rng.normal(0,.2,(m,t)).astype(np.float32)
            original=[v.copy() for v in (w,x,bias,skip)]
            baseline,_=ref.run(w,x,8,a.backend,2)
            unchanged,_=api.run(w,x,8,a.backend,2);bits(unchanged,baseline)
            plan=api.lib.ip_create(m,k,pointer(w),8,a.backend);assert plan,api.error()
            try:
                assert api.lib.ip_workspace_prepare(workspace,plan,pointer(x),t)==0,api.error()
                raw=np.full((m,t),np.nan,np.float32)
                edges=sorted(set([0,m]+[r for r in (1,3,17,64,127,256) if r<m]))
                for first,last in zip(edges,edges[1:]):
                    assert api.lib.ip_workspace_run_rows(workspace,plan,pointer(raw),first,last,None,None)==0,api.error()
                bits(raw,baseline)
                out=np.empty_like(raw)
                assert api.lib.ip_workspace_run_rows(workspace,plan,pointer(out),0,m,pointer(bias),pointer(skip))==0,api.error()
                bits(out,ordered(baseline,bias,skip))
                if t>1:
                    cut=t//2;future=x.copy();future[:,cut:]*=-17
                    assert api.lib.ip_workspace_prepare(workspace,plan,pointer(future),t)==0,api.error()
                    assert api.lib.ip_workspace_run_rows(workspace,plan,pointer(out),0,m,pointer(bias),pointer(skip))==0,api.error()
                    bits(out[:,:cut],ordered(baseline,bias,skip)[:,:cut])
                    prefix=np.ascontiguousarray(x[:,:cut]);prefix_skip=np.ascontiguousarray(skip[:,:cut])
                    prefix_out=np.empty((m,cut),np.float32)
                    assert api.lib.ip_workspace_prepare(workspace,plan,pointer(prefix),cut)==0,api.error()
                    assert api.lib.ip_workspace_run_rows(workspace,plan,pointer(prefix_out),0,m,pointer(bias),pointer(prefix_skip))==0,api.error()
                    bits(prefix_out,ordered(baseline,bias,skip)[:,:cut])
                assert api.lib.ip_workspace_bytes(workspace)==allocation
                for value,before in zip((w,x,bias,skip),original):bits(value,before)
                records.append({'M':m,'K':k,'T':t,'raw_uint32_equal':True,'ordered_epilogue_uint32_equal':True,
                                'future_and_prefix_uint32_equal':t>1,'capacity_unchanged':True})
            finally:api.lib.ip_destroy_plan(plan)
    finally:api.lib.ip_destroy_workspace(workspace)

    # One prepared input, two different projection matrices and output sizes.
    k,t=256,65;x=rng.normal(size=(k,t)).astype(np.float32)
    weights=[rng.normal(size=(m,k)).astype(np.float32) for m in (256,128)]
    plans=[api.lib.ip_create(w.shape[0],k,pointer(w),8,a.backend) for w in weights]
    assert all(plans),api.error();workspace=api.lib.ip_workspace_create(k,t,256);assert workspace
    try:
        assert api.lib.ip_workspace_prepare(workspace,plans[0],pointer(x),t)==0,api.error()
        for plan,w in zip(plans,weights):
            out=np.empty((w.shape[0],t),np.float32)
            assert api.lib.ip_workspace_run_rows(workspace,plan,pointer(out),0,w.shape[0],None,None)==0,api.error()
            wanted,_=ref.run(w,x,8,a.backend);bits(out,wanted)
    finally:
        api.lib.ip_destroy_workspace(workspace)
        for plan in plans:api.lib.ip_destroy_plan(plan)

    # Scale multiplication underflows to zero. Negative products, negative-zero
    # bias and negative-zero skip exercise actual IEEE sign bits in the epilogue.
    w=np.full((17,31),np.float32(1e-22));w[::2]*=-1
    x=np.full((31,17),np.float32(1e-22));bias=np.full(17,-0.,np.float32)
    skip=np.full((17,17),-0.,np.float32);raw,_=ref.run(w,x,8,a.backend)
    workspace=api.lib.ip_workspace_create(31,17,17);plan=api.lib.ip_create(17,31,pointer(w),8,a.backend)
    try:
        out=np.empty_like(skip)
        assert api.lib.ip_workspace_prepare(workspace,plan,pointer(x),17)==0,api.error()
        assert api.lib.ip_workspace_run_rows(workspace,plan,pointer(out),0,17,pointer(bias),pointer(skip))==0,api.error()
        bits(out,ordered(raw,bias,skip));assert np.signbit(out[0]).all() and not np.signbit(out[1]).any()
    finally:api.lib.ip_destroy_workspace(workspace);api.lib.ip_destroy_plan(plan)

    # Cancellation distinguishes skip+(product+bias) from (skip+product)+bias.
    w=np.full((17,1),np.float32(2**26));x=np.ones((1,17),np.float32)
    raw,_=ref.run(w,x,8,a.backend);bias=np.ones(17,np.float32);skip=-raw
    wanted=ordered(raw,bias,skip);assert np.all(wanted==0)
    workspace=api.lib.ip_workspace_create(1,17,17);plan=api.lib.ip_create(17,1,pointer(w),8,a.backend)
    assert workspace and plan
    try:
        out=np.empty_like(raw)
        assert api.lib.ip_workspace_prepare(workspace,plan,pointer(x),17)==0,api.error()
        assert api.lib.ip_workspace_run_rows(workspace,plan,pointer(out),0,17,pointer(bias),pointer(skip))==0,api.error()
        bits(out,wanted)
    finally:api.lib.ip_destroy_workspace(workspace);api.lib.ip_destroy_plan(plan)

    # Concurrent independent calls share immutable weights, never scratch.
    w=rng.normal(size=(128,128)).astype(np.float32);plan=api.lib.ip_create(128,128,pointer(w),8,a.backend);assert plan
    def worker(t):
        x=np.cos(np.arange(128*t,dtype=np.float32)/71).reshape(128,t)
        workspace=api.lib.ip_workspace_create(128,t,128);assert workspace
        try:
            assert api.lib.ip_workspace_prepare(workspace,plan,pointer(x),t)==0,api.error()
            out=np.empty((128,t),np.float32)
            assert api.lib.ip_workspace_run_rows(workspace,plan,pointer(out),0,128,None,None)==0,api.error()
            wanted,_=ref.run(w,x,8,a.backend);bits(out,wanted)
        finally:api.lib.ip_destroy_workspace(workspace)
    try:
        with concurrent.futures.ThreadPoolExecutor(4) as pool:list(pool.map(worker,[1,3,17,65,128,255,256,513]))
    finally:api.lib.ip_destroy_plan(plan)

    invalid=0
    for bounds in [(0,1,1),(16385,1,1),(2,-1,1),(2,1,0)]:
        assert not api.lib.ip_workspace_create(*bounds);invalid+=1
    w=np.ones((2,2),np.float32);x=np.ones((2,4),np.float32)
    bias=np.zeros(2,np.float32);skip=np.zeros((2,4),np.float32);out=np.empty_like(skip)
    workspace=api.lib.ip_workspace_create(2,4,1);plan=api.lib.ip_create(2,2,pointer(w),8,a.backend)
    half=api.lib.ip_create(2,2,pointer(w),16,a.backend);wide_w=np.ones((2,3),np.float32)
    wide=api.lib.ip_create(2,3,pointer(wide_w),8,a.backend)
    try:
        assert api.lib.ip_workspace_run_rows(workspace,plan,pointer(out),0,2,None,None)!=0;invalid+=1
        for target,t in [(plan,-1),(plan,5),(half,4),(wide,4)]:
            assert api.lib.ip_workspace_prepare(workspace,target,pointer(x),t)!=0;invalid+=1
        assert api.lib.ip_workspace_prepare(workspace,plan,pointer(x),4)==0
        for target in (half,wide):
            assert api.lib.ip_workspace_run_rows(workspace,target,pointer(out),0,2,None,None)!=0;invalid+=1
        for first,last in [(-1,2),(0,3),(2,1)]:
            assert api.lib.ip_workspace_run_rows(workspace,plan,pointer(out),first,last,None,None)!=0;invalid+=1
        for target in (x,w,bias,skip):
            assert api.lib.ip_workspace_run_rows(workspace,plan,pointer(target),0,2,pointer(bias),pointer(skip))!=0;invalid+=1
        for target in (x,skip):
            partial=ct.cast(target.ctypes.data+4,P)
            assert api.lib.ip_workspace_run_rows(workspace,plan,partial,0,2,pointer(bias),pointer(skip))!=0;invalid+=1
        for b,s in [(pointer(bias),None),(None,pointer(skip))]:
            assert api.lib.ip_workspace_run_rows(workspace,plan,pointer(out),0,2,b,s)!=0;invalid+=1
        for bad in (np.nan,np.inf):
            wrong=x.copy();wrong[0,0]=bad
            assert api.lib.ip_workspace_prepare(workspace,plan,pointer(wrong),4)!=0;invalid+=1
            assert api.lib.ip_workspace_run_rows(workspace,plan,pointer(out),0,2,None,None)!=0;invalid+=1
        assert api.lib.ip_workspace_prepare(workspace,plan,pointer(x),4)==0
        bad_bias=bias.copy();bad_bias[0]=np.inf
        assert api.lib.ip_workspace_run_rows(workspace,plan,pointer(out),0,2,pointer(bad_bias),pointer(skip))!=0;invalid+=1
        bad_skip=skip.copy();bad_skip[0,0]=np.nan
        assert api.lib.ip_workspace_run_rows(workspace,plan,pointer(out),0,2,pointer(bias),pointer(bad_skip))!=0;invalid+=1
        huge_bias=np.full(2,np.finfo(np.float32).max,np.float32);huge_skip=np.full((2,4),np.finfo(np.float32).max,np.float32)
        assert api.lib.ip_workspace_run_rows(workspace,plan,pointer(out),0,2,pointer(huge_bias),pointer(huge_skip))!=0;invalid+=1
        assert api.lib.ip_workspace_run_rows(workspace,plan,pointer(out),0,2,None,None)==0,api.error()
        wanted,_=ref.run(w,x,8,a.backend);bits(out,wanted)
    finally:
        api.lib.ip_destroy_workspace(workspace)
        for target in (plan,half,wide):api.lib.ip_destroy_plan(target)

    result={'status':'passed','GPU_used':False,'timings_collected':False,'models_executed':False,
            'library_sha256':sha(a.library),'reference_library_sha256':sha(a.reference_library),
            'script_sha256':sha(__file__),'symbol_prefix':a.symbol_prefix,'backend':a.backend,
            'shape_cases':records,'shared_projection_input':True,'signed_zero_uint32':True,
            'independent_concurrent_calls':8,'cancellation_order_fixture':True,
            'shared_workspace_misuse_tested':False,'invalid_cases':invalid,'workspace_bytes_constant':allocation}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'passed','shape_cases':len(records),'invalid_cases':invalid,'output':str(a.output)}))

if __name__=='__main__':main()
