"""Focused CPU-only XSMM tiny/full/tail dispatch and numerical regression.
No timing. Run this and the matrix probe's --tiny-only gate before benchmarks.
"""
import argparse,ctypes,hashlib,json,os
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='-1',NVIDIA_VISIBLE_DEVICES='void',
                  ROCR_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    lib=ctypes.CDLL(str(a.library.resolve()));ptr=ctypes.POINTER(ctypes.c_float)
    lib.fx_create_ordered.argtypes=[ctypes.c_int]*8;lib.fx_create_ordered.restype=ctypes.c_void_p
    lib.fx_destroy.argtypes=[ctypes.c_void_p];lib.fx_blocks.argtypes=[ctypes.c_void_p]
    lib.fx_run_range.argtypes=[ctypes.c_void_p,*([ptr]*5),ctypes.c_int,ctypes.c_int]
    if lib.fx_init()<0 or lib.fx_has_xsmm()!=1:raise RuntimeError('Real guarded XSMM build required')
    pointer=lambda x:x.ctypes.data_as(ptr)
    rng=np.random.default_rng(9134);records=[]
    report={'library_sha256':hashlib.sha256(a.library.read_bytes()).hexdigest(),'records':records,
            'gpu_used':False,'timing':False,'scope':'Real XSMM tiny/full/tail category regression, both modes and epilogue orders'}
    def save():a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    try:
        for c in (32,64):
            for t in (1,17,63,64,65,127,128,129,255,256,257):
                w=rng.normal(0,.1,(c,c)).astype(np.float32);x=rng.normal(0,.1,(c,t)).astype(np.float32)
                b=rng.normal(0,.1,c).astype(np.float32);skip=rng.normal(0,.1,(c,t)).astype(np.float32)
                product=np.matmul(w,x);biased=product+b[:,None]
                for mode in (1,2):
                    for order in (0,1):
                        # tc=64 deliberately exceeds C32; tt=128 exceeds many T.
                        plan=lib.fx_create_ordered(c,c,t,128,64,mode,0,order)
                        if not plan:raise AssertionError(f'Plan rejected c{c},t{t},mode{mode}')
                        try:
                            y=np.full_like(x,np.nan);blocks=lib.fx_blocks(plan)
                            # Exercise partial ranges and the final tail category.
                            for first in range(blocks):
                                status=lib.fx_run_range(plan,pointer(w),pointer(x),pointer(b),pointer(skip),pointer(y),first,first+1)
                                if status:raise AssertionError('Run failed')
                            ref=skip+biased if order else biased+skip
                            delta=np.abs(y.astype(np.float64)-ref.astype(np.float64));maximum=float(delta.max(initial=0))
                            passed=bool(np.isfinite(y).all() and np.allclose(y,ref,atol=2e-5,rtol=2e-5))
                            records.append({'c':c,'t':t,'tt':128,'tc':64,'mode':mode,'skip_first':order,'max_abs':maximum,'pass':passed})
                            if not passed:raise AssertionError('Numerical parity failed')
                        finally:lib.fx_destroy(plan)
        report['status']='complete';report['case_count']=len(records)
    except Exception as e:report['status']='failed';report['error']=str(e);save();raise
    save();print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))

if __name__=='__main__':main()
