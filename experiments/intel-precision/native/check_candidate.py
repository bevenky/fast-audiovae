"""Validate larger ordinary INT8 GEMM and fixed-geometry half GEMM without timing.

No timing. C API handles are loaded locally by ctypes; no ORT plugin from
either core is registered together. Full-decoder comparisons use processes.
"""
import argparse
import concurrent.futures
import json
from pathlib import Path
import numpy as np
from check_micro import API,pointer,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--reference-library',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Output exists')
    packed=API(a.library);plain=API(a.reference_library);rng=np.random.default_rng(20260909)
    rows=[]
    for m,k,t in [(1,1,1),(17,31,17),(129,257,513),(256,256,65),
                  (128,128,129),(64,64,127),(32,32,257),(3,2048,17),(2049,31,65),(4097,17,17),(3072,128,513)]:
        w=rng.normal(size=(m,k)).astype(np.float32);x=rng.normal(size=(k,t)).astype(np.float32)
        for shards in (1,2):
            ref,_=plain.run(w,x,8,1,shards);got,usage=packed.run(w,x,8,1,shards)
            np.testing.assert_array_equal(got,ref)
            rows.append({'M':m,'K':k,'T':t,'shards':shards,'bitwise_equal':True,**usage})

    # Mode16 has a deliberately different, fixed SGEMM reduction geometry from
    # r2. It must retain strict causal prefixes across clip lengths and row
    # partition boundaries; no numerical tolerance is used for these checks.
    half_rows=[]
    for m,k in [(1,1),(3,2048),(65,257),(128,128),(129,31)]:
        t=170;w=rng.normal(0,.1,size=(m,k)).astype(np.float32)
        x=rng.normal(0,.1,size=(k,t)).astype(np.float32)
        full,_=packed.run(w,x,16,1,1)
        sharded,_=packed.run(w,x,16,1,2)
        np.testing.assert_array_equal(sharded,full)
        scalar,_=plain.run(w,x,16,0,1)
        np.testing.assert_allclose(full,scalar,rtol=2e-5,atol=2e-5)
        for length in (1,2,17,31,32,63,64,65,127,128,129,169):
            prefix=np.ascontiguousarray(x[:,:length])
            for shards in (1,2):
                got,_=packed.run(w,prefix,16,1,shards)
                np.testing.assert_array_equal(got,full[:,:length])
                half_rows.append({'M':m,'K':k,'T':length,'shards':shards,'bitwise_prefix':True})
        plan=packed.lib.ip_create(m,k,pointer(w),16,1);assert plan,packed.error()
        inp=packed.lib.ip_prepare(plan,pointer(x),t);assert inp,packed.error()
        try:
            got=np.full((m,t),np.nan,np.float32)
            edges=sorted(set([0,m]+[r for r in (1,3,17,33,63,64,65,127) if r<m]))
            for first,last in zip(edges,edges[1:]):
                assert packed.lib.ip_run_rows(plan,inp,pointer(got),first,last)==0,packed.error()
            np.testing.assert_array_equal(got,full)
        finally:packed.lib.ip_destroy_input(inp);packed.lib.ip_destroy_plan(plan)

    result={'status':'passed','GPU_used':False,'timings_collected':False,
            'library_sha256':sha(a.library),'reference_library_sha256':sha(a.reference_library),
            'script_sha256':sha(__file__),'bitwise_cases':rows,
            'half_fixed_geometry':[64,64],'half_prefix_cases':half_rows,
            'half_irregular_row_partition_cases':5,
            'simultaneous_ORT_plugins':False}
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'passed','bitwise_cases':len(rows),
                      'half_prefix_cases':len(half_rows),'output':str(a.output)}))

if __name__=='__main__':main()
