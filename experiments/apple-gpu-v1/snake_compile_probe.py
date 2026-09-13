"""Single small FP32 Snake fusion probe, never a production selector."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
os.environ.update(PYTORCH_ENABLE_MPS_FALLBACK='0',PYTORCH_MPS_FAST_MATH='0')
from pathlib import Path
import sys, json, time, statistics
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'work/fast-audiovae-apple-gpu/src'))
import torch
from fast_audiovae.mps_decoder import MPSModel, _Snake

def snake(x, alpha, reciprocal):
    sine=torch.sin(alpha*x)
    squared=sine*sine
    return x+reciprocal*squared

def main():
    out=HERE/'snake-compile-probe.json';assert not out.exists()
    r=dict(status='running',torch=torch.__version__,fast_math=False,rows=[])
    def save():out.write_text(json.dumps(r,indent=2)+'\n')
    try:
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        assert torch.backends.mps.is_available()
        model=MPSModel.from_onnx(ROOT/'work/audiovae2_onnx/audio_vae_decoder.onnx')
        fused=torch.compile(snake,backend='inductor',dynamic=True,fullgraph=True)
        with torch.inference_mode():
            for channels,frames in ((1024,16),(512,96),(32,3840)):
                block=next(m for m in model.modules() if isinstance(m,_Snake) and m.alpha.shape[1]==channels)
                generator=torch.Generator().manual_seed(41+channels)
                x=(torch.randn((1,channels,frames),generator=generator)*.2).to('mps')
                args=(x,block.alpha,block.reciprocal)
                torch.mps.synchronize();start=time.perf_counter();yc=fused(*args);torch.mps.synchronize()
                compile_seconds=time.perf_counter()-start
                ye=snake(*args);torch.mps.synchronize()
                difference=(yc-ye).abs().max().item()
                assert torch.allclose(yc,ye,atol=1e-6,rtol=1e-5),difference
                row=dict(shape=list(x.shape),max_abs=difference,first_compiled_call_seconds=compile_seconds,eager_ms=[],compiled_ms=[])
                for rep in range(15):
                    order=[('eager',snake),('compiled',fused)]
                    if rep%2:order.reverse()
                    for name,fn in order:
                        torch.mps.synchronize();start=time.perf_counter();y=fn(*args);torch.mps.synchronize()
                        seconds=time.perf_counter()-start
                        if rep>=5:row[name+'_ms'].append(seconds*1000)
                row['median_eager_ms']=statistics.median(row['eager_ms'])
                row['median_compiled_ms']=statistics.median(row['compiled_ms'])
                r['rows'].append(row);save()
        r['status']='passed'
    except BaseException as error:r.update(status='failed',error=repr(error));raise
    finally:save()
    print(json.dumps(r,indent=2))

if __name__=='__main__':main()
