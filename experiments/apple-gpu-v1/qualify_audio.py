"""Bounded original-weight Apple GPU waveform and streaming qualification."""
from pathlib import Path
import os, sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
os.environ.update(PYTORCH_ENABLE_MPS_FALLBACK='0',PYTORCH_MPS_FAST_MATH='0',ORT_DISABLE_TELEMETRY='1')
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'work/fast-audiovae-apple-gpu/src'))
import json, time, hashlib, argparse
import numpy as np
import torch
import onnxruntime as ort
from fast_audiovae import load

def main(output,smoke):
    assert not output.exists()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    source=ROOT/'work/audiovae2_onnx/audio_vae_decoder.onnx'
    result=dict(status='running',device='Apple GPU / MPS',torch=torch.__version__,onnxruntime=ort.__version__,
        fallback=False,fast_math=False,precision='FP32',calls=0,api_seconds=0.,checks=[],smoke=smoke)
    def save():output.write_text(json.dumps(result,indent=2)+'\n')
    def call(fn):
        assert result['api_seconds']<30
        start=time.perf_counter()
        try:return fn()
        finally:
            result['calls']+=1;result['api_seconds']+=time.perf_counter()-start
            assert result['api_seconds']<30,'Bounded API budget exceeded'
    def check(label,y,ref):
        assert y.shape==ref.shape and y.dtype==ref.dtype==np.float32
        assert np.isfinite(y).all() and np.isfinite(ref).all()
        diff=y-ref;passed=bool(np.allclose(y,ref,atol=1e-5,rtol=1e-4))
        result['checks'].append(dict(label=label,shape=list(y.shape),max_abs=float(np.max(np.abs(diff))) if y.size else 0.,rms_error=float(np.sqrt(np.mean(diff*diff))) if y.size else 0.,passed=passed))
        save();assert passed,label
    try:
        assert torch.__version__.split('+')[0]=='2.14.0' and torch.backends.mps.is_available()
        options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        cpu=ort.InferenceSession(str(source),options,providers=['CPUExecutionProvider']);cpu.disable_fallback()
        gpu=load(device='gpu',source=source,offline=True,cache_dir=HERE/'cache')
        result['info']=gpu.info
        sources=[('English','en_us_00103_1779'),('Bengali','bn_in_00151_1818'),('Spanish','es_419_00060_1994')]
        cases={}
        with np.load(ROOT/'work/codec-quality-inputs/multilingual-encoded/fast_audiovae2.npz',allow_pickle=False) as f:
            for label,key in sources:cases[label]=np.ascontiguousarray(f[key+'__z'][...,:2 if smoke else 7])
        if smoke:cases={'English':cases['English']}
        else:
            with np.load(ROOT/'outputs/apple-streaming-promotion-v1/extra-encoded-v1/latents.npz',allow_pickle=False) as f:
                names=[n for n in f.files if n.endswith('__z')]
                for i,n in enumerate(names[:2]):cases[f'Expressive{i}']=np.ascontiguousarray(f[n][...,:7])
            cases['Zero latent']=np.zeros((1,64,7),np.float32)
            cases['Quiet latent']=cases['English']*np.float32(0.001)
        result['cases']=[dict(name=n,shape=list(z.shape),sha256=hashlib.sha256(z.tobytes()).hexdigest()) for n,z in cases.items()]
        for label,z in cases.items():
            original=z.tobytes();ref=call(lambda:cpu.run(None,{'z':z})[0])
            with gpu.stream() as s:
                full=call(lambda:s.decode_chunk(z));check(label+'/fresh full',full,ref)
                if not smoke:
                    empty=call(s.flush);assert empty.shape==(1,1,0)
                    s.reset();check(label+'/reset repeat',call(lambda:s.decode_chunk(z)),full)
            for step in (1,2):
                with gpu.stream() as s:
                    parts=[]
                    for pos in range(0,z.shape[-1],step):parts.append(call(lambda:s.decode_chunk(np.ascontiguousarray(z[...,pos:pos+step]))))
                    y=np.concatenate(parts,axis=-1);check(label+f'/stream{step*40}',y,ref)
                    assert s.flush().shape==(1,1,0)
                    assert all(t.device.type=='mps' for t in s._history.values())
            assert z.tobytes()==original
        if not smoke:
            a=cases['English'];b=cases['Spanish']
            with gpu.stream() as sa,gpu.stream() as sb:
                ya=call(lambda:sa.decode_chunk(a[...,:2].copy()));yb=call(lambda:sb.decode_chunk(b[...,:1].copy()))
                tail=call(lambda:sa.decode_chunk(a[...,2:].copy()))
                ra=call(lambda:cpu.run(None,{'z':a})[0]);rb=call(lambda:cpu.run(None,{'z':b[...,:1].copy()})[0])
                check('independent streams/A',np.concatenate([ya,tail],-1),ra);check('independent streams/B',yb,rb)
            batch=load(device='gpu',mode='batch',source=source,offline=True,cache_dir=HERE/'cache')
            check('public batch',call(lambda:batch.decode(a)),ra)
        result['status']='passed'
    except BaseException as e:
        result.update(status='failed',error=repr(e));raise
    finally:save()
    print(json.dumps({k:result[k] for k in ('status','calls','api_seconds','checks')},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.output,a.smoke)
