"""CPU-only small-operator checks. No codec weights, corpus, or timings."""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='-1', NVIDIA_VISIBLE_DEVICES='void', ROCR_VISIBLE_DEVICES='-1', HIP_VISIBLE_DEVICES='-1')
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as TP
import onnxruntime as ort
DOMAIN='venky.audio.cpu.portable'


def model(op,co,constants,attrs,combined=False):
    stride=attrs.get('stride',1)
    inputs=[h.make_tensor_value_info('x',TP.FLOAT,['B',co*stride*(2 if combined else 1),'T'])]
    names=['x']
    if op=='PhaseSumBiasInterleaveF32':
        inputs.append(h.make_tensor_value_info('p',TP.FLOAT,['B',co*stride,'T']));names.append('p')
    inits=[nh.from_array(a,n) for n,a in constants.items()]
    node=h.make_node(op,names+list(constants),['y'],domain=DOMAIN,channels=co,native_abi=1,row_batches=0,**attrs)
    g=h.make_graph([node],'tiny_portable',inputs,[h.make_tensor_value_info('y',TP.FLOAT,['B',co,'Y'])],inits)
    return h.make_model(g,opset_imports=[h.make_opsetid('',20),h.make_opsetid(DOMAIN,1)],ir_version=10)


def session(m,library,threads):
    options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1
    options.add_session_config_entry("session.intra_op.allow_spinning","0")
    options.add_session_config_entry("session.inter_op.allow_spinning","0")
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level=3;options.register_custom_ops_library(str(library))
    return ort.InferenceSession(m.SerializeToString(),options,providers=['CPUExecutionProvider'])


def validate(library,output):
    if ort.__version__ != "1.29.0":raise RuntimeError("Validation requires ONNX Runtime 1.29.0")
    library=Path(library).resolve();out=Path(output);out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(910713)
    lib=ctypes.CDLL(str(library));lib.ncc_sine_f32.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64,ctypes.c_int32]
    lib.ncc_sine_f32.restype=ctypes.c_int32
    lib.ncc_backend_available.argtypes=[ctypes.c_int32];lib.ncc_backend_available.restype=ctypes.c_int32
    lib.ncc_snake_math_name.argtypes=[ctypes.c_int32];lib.ncc_snake_math_name.restype=ctypes.c_char_p
    backends=[b for b in (3,4) if lib.ncc_backend_available(b)]
    if not backends:raise RuntimeError('Expected SSE2 or AVX2 CPU')
    numerical=[]
    # Exact FP32 inputs spanning normal/subnormal signs and magnitudes. Compare
    # rounded binary64 libm sine, a numerical probe rather than a proof for all inputs.
    bits=rng.integers(0,2**32,size=20003,dtype=np.uint32)
    values=bits.view(np.float32);values=values[np.isfinite(values)].copy()
    values=np.concatenate([np.array([0.,-0.,np.nextafter(np.float32(0),np.float32(1)),1.,-1.,np.pi],np.float32),values])
    ref=np.array([math.sin(float(v)) for v in values],np.float32)
    def order(a):
        u=a.view(np.uint32).astype(np.int64)
        return np.where(u&0x80000000,0x80000000-(u&0x7fffffff),0x80000000+u)
    for backend in backends:
        y=np.empty_like(values)
        if lib.ncc_sine_f32(values.ctypes.data,y.ctypes.data,len(y),backend)!=0:raise AssertionError("Sine call failed")
        ulp=np.abs(order(y)-order(ref))
        if int(ulp.max())>1:raise AssertionError(f'sine error >1 rounded-reference ULP: {ulp.max()}')
        np.testing.assert_array_equal(y[:2].view(np.uint32),ref[:2].view(np.uint32))
        numerical.append({'backend':backend,'math':lib.ncc_snake_math_name(backend).decode(),'samples':len(y),'max_ulp_vs_rounded_binary64_libm':int(ulp.max()),'max_abs_error':float(np.max(np.abs(y-ref)))})
    checks=[]
    for threads in (1,4):
      for backend in backends:
       for op in ('SnakeF32','CausalDW7F32','CausalDW7SnakeF32'):
        for dilation in ((1,) if op=='SnakeF32' else (1,3,9)):
         co=3; parameters=[]; sessions=[]
         for seed in (1,2):
            cr=np.random.default_rng(seed)
            a=np.exp(cr.normal(0,.8,co)).astype(np.float32)
            r=(1/(a+np.float32(1e-8))).astype(np.float32)
            w=cr.normal(0,.15,(co,1,7)).astype(np.float32);b=cr.normal(0,.1,co).astype(np.float32)
            constants={}
            if op!='SnakeF32':constants.update(w=w,b=b)
            if op!='CausalDW7F32':constants.update(a=a,r=r)
            attrs={'backend':backend}
            if op!='SnakeF32':attrs['dilation']=dilation
            if op!='CausalDW7F32':attrs['require_vector_sine']=1
            sessions.append(session(model(op,co,constants,attrs),library,threads));parameters.append((w,b,a,r))
         xs={t:rng.normal(0,.5,(2,co,t)).astype(np.float32) for t in (1,7,23,257)}
         for time in (257,1,7,23,257):
          x=xs[time];first=None
          for which in (0,1,0):
            w,b,a,r=parameters[which];ref=x
            if op!='SnakeF32':
                pad=np.pad(x,((0,0),(0,0),(6*dilation,0)))
                ref=np.multiply(pad[...,:time],w[:,0,0][None,:,None],dtype=np.float32)
                for k in range(1,7):
                    product=np.multiply(pad[...,k*dilation:k*dilation+time],w[:,0,k][None,:,None],dtype=np.float32)
                    ref=np.add(ref,product,dtype=np.float32)
                ref=np.add(ref,b[None,:,None],dtype=np.float32)
            if op!='CausalDW7F32':
                scaled=np.multiply(a[None,:,None],ref,dtype=np.float32)
                sine=np.sin(scaled.astype(np.float64)).astype(np.float32)
                square=np.multiply(sine,sine,dtype=np.float32)
                correction=np.multiply(r[None,:,None],square,dtype=np.float32)
                ref=np.add(ref,correction,dtype=np.float32)
            y=sessions[which].run(None,{'x':x})[0]
            np.testing.assert_allclose(y,ref,rtol=2e-5,atol=2e-6)
            if op=='CausalDW7F32':np.testing.assert_array_equal(y.view(np.uint32),ref.view(np.uint32))
            if which==0:
                if first is not None:np.testing.assert_array_equal(y.view(np.uint32),first)
                first=y.view(np.uint32).copy()
            checks.append({'op':op,'threads':threads,'backend':backend,'dilation':dilation,'time':time,'session':which,'max_abs_error':float(np.max(np.abs(y-ref)))})
    phasechecks=[]
    for threads in (1,4):
     for co in (1,3,8):
      for stride in (2,5,6,8):
       biases=[rng.normal(0,.5,co).astype(np.float32) for _ in range(2)]
       biases[0][0]=-0.;biases[1][0]=-.75
       sessions=[]
       for combined in (False,True):
        op=('Combined' if combined else '')+'PhaseSumBiasInterleaveF32'
        sessions.append([session(model(op,co,{'bias':bias},{'stride':stride,'previous_shift':1},combined),library,threads) for bias in biases])
       inputs={}
       for batch,time in ((1,257),(2,1),(1,7),(2,17)):
        cur=rng.normal(0,3,(batch,co*stride,time)).astype(np.float32);prev=rng.normal(0,3,cur.shape).astype(np.float32)
        cur[...,0]=-0.;prev[...,-1]=np.nan;inputs[batch,time]=(cur,prev)
       for batch,time in ((1,257),(2,1),(1,7),(2,17),(1,257)):
        cur,prev=inputs[batch,time];shifted=np.zeros_like(prev);shifted[...,1:]=prev[...,:-1]
        base=np.add(cur,shifted,dtype=np.float32).reshape(batch,co,stride,time).transpose(0,1,3,2).reshape(batch,co,time*stride)
        for combined in (False,True):
         first=None
         for which in (0,1,0):
          feeds={'x':np.concatenate([cur,prev],axis=1)} if combined else {'x':cur,'p':prev}
          y=sessions[combined][which].run(None,feeds)[0]
          ref=np.add(base,biases[which][None,:,None],dtype=np.float32)
          np.testing.assert_array_equal(y.view(np.uint32),ref.view(np.uint32))
          if which==0:
           if first is not None:np.testing.assert_array_equal(y.view(np.uint32),first)
           first=y.view(np.uint32).copy()
          phasechecks.append({'combined':combined,'threads':threads,'channels':co,'stride':stride,'batch':batch,'time':time,'session':which,'bitwise':True})
    if len(checks)!=210*len(backends) or len(phasechecks)!=720:raise AssertionError('Incomplete native fixture coverage')
    result={'library':str(library),'library_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),'ort_version':ort.__version__,
            'providers':['CPUExecutionProvider'],'all_passed':True,'sine':numerical,'native_checks':len(checks),'phase_checks':len(phasechecks),
            'native_max_abs_error':max(r['max_abs_error'] for r in checks),'phase_bitwise':True,'native_records':checks,'phase_records':phasechecks,
            'scope':'Small operators only; no full model, timing, quality, or universal bitwise claim'}
    (out/'tiny_portable_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    return {k:v for k,v in result.items() if not k.endswith('_records')}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--library',required=True);parser.add_argument('--output-dir',required=True)
    args=parser.parse_args();print(json.dumps(validate(args.library,args.output_dir),indent=2))
