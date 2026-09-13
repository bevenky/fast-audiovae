"""Short, original-weight MPS vs frozen CPU reference qualification."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
os.environ.update(PYTORCH_ENABLE_MPS_FALLBACK='0',PYTORCH_MPS_FAST_MATH='0')
from pathlib import Path
import json, time, hashlib, argparse
import numpy as np
import torch
import onnxruntime as ort
from mimi_mps import MimiMPSAdapter

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=HERE/'mimi-qualification.json');args=p.parse_args()
OUT=args.output
assert not OUT.exists()
torch.set_num_threads(1);torch.set_num_interop_threads(1)
r=dict(status='running',checks=[],call_seconds=0.,calls=0)
def save():OUT.write_text(json.dumps(r,indent=2)+'\n')
def call(fn):
    assert r['call_seconds']<30
    start=time.perf_counter()
    try:
        with torch.inference_mode():
            y=fn();y=y.detach().cpu().numpy().copy() if isinstance(y,torch.Tensor) else y
            torch.mps.synchronize()
            return y
    finally:r['calls']+=1;r['call_seconds']+=time.perf_counter()-start
def check(label,y,ref):
    assert y.shape==ref.shape and y.dtype==ref.dtype==np.float32 and np.isfinite(y).all()
    d=y-ref;ok=bool(np.allclose(y,ref,atol=1e-5,rtol=1e-4))
    r['checks'].append(dict(label=label,shape=list(y.shape),max_abs=float(np.max(np.abs(d))),rms_error=float(np.sqrt(np.mean(d*d))),passed=ok))
    save();assert ok,label
try:
    m=MimiMPSAdapter(max_frames=40);r['model']=m.metadata
    # The SEANet ratios 6*5*4 give 200 Hz before the 12.5 Hz bottleneck.
    assert m.model.encoder_frame_rate/m.model.frame_rate==16
    opts=ort.SessionOptions();opts.intra_op_num_threads=opts.inter_op_num_threads=1
    opts.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    opts.add_session_config_entry('session.intra_op.allow_spinning','0')
    p=ROOT/'work/streaming-comparison/mimi_onnx/mimi_full.onnx'
    assert hashlib.sha256(p.read_bytes()).hexdigest()=='e023777a2cee98a7e4a293f6af7712d4fed6364d1523cfc9a6283f112981e2f1'
    cpu=ort.InferenceSession(str(p),opts,providers=['CPUExecutionProvider']);cpu.disable_fallback()
    cases={}
    with np.load(ROOT/'work/codec-quality-inputs/multilingual-encoded/mimi_vae.npz',allow_pickle=False) as f:
        for uid in ('en_us_00103_1779','bn_in_00151_1818','es_419_00060_1994'):
            z=f[uid+'__z'];start=max(0,(z.shape[-1]-19)//2)
            cases[uid]=np.ascontiguousarray(z[...,start:start+19])
    cases['zero']=np.zeros((1,32,3),np.float32)
    for name,z in cases.items():
        ref=cpu.run(None,{cpu.get_inputs()[0].name:z})[0]
        full=call(lambda:m.full(torch.from_numpy(z).to('mps')));check(name+'/full',full,ref)
        for packet in (1,2):
            with m.stream() as s:
                parts=[]
                for pos in range(0,z.shape[-1],packet):
                    part=np.ascontiguousarray(z[...,pos:pos+packet])
                    parts.append(call(lambda:s.decode_chunk(torch.from_numpy(part).to('mps'))))
                check(name+f'/stream{packet*80}',np.concatenate(parts,-1),ref)
                assert all(t.device.type=='mps' for state in s.states.values() for t in state.values())
                assert s.frames_decoded==z.shape[-1] and s.flush().shape==(1,1,0)
                s.reset();check(name+f'/reset{packet*80}',call(lambda:s.decode_chunk(torch.from_numpy(z).to('mps'))),ref)
    a,b=list(cases.values())[:2]
    with m.stream() as sa,m.stream() as sb:
        ya=call(lambda:sa.decode_chunk(torch.from_numpy(a[...,:1].copy()).to('mps')))
        yb=call(lambda:sb.decode_chunk(torch.from_numpy(b[...,:2].copy()).to('mps')))
        ya2=call(lambda:sa.decode_chunk(torch.from_numpy(a[...,1:].copy()).to('mps')))
        check('independent A',np.concatenate((ya,ya2),-1),cpu.run(None,{cpu.get_inputs()[0].name:a})[0])
        check('independent B',yb,cpu.run(None,{cpu.get_inputs()[0].name:b[...,:2].copy()})[0])
    r['status']='passed'
except BaseException as e:r.update(status='failed',error=repr(e));raise
finally:save()
print(json.dumps({k:r[k] for k in ('status','call_seconds','calls','checks')},indent=2))
