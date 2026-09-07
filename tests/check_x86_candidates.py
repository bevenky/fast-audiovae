"""CPU-only parity, causality and C-ABI safety checks for experimental native ops.

Run explicitly with a built Linux x86 library. No codec weights or timing.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
from concurrent.futures import ThreadPoolExecutor
os.environ.update(CUDA_VISIBLE_DEVICES='-1',NVIDIA_VISIBLE_DEVICES='void',
                  ROCR_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',
                  OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
import onnxruntime as ort
from onnx import TensorProto as TP,helper as h,numpy_helper as nh
from check_x86 import DOMAIN,session


def graph(kind,co,constants,backend,dilation=1):
    inputs=[h.make_tensor_value_info('x',TP.FLOAT,['B',co,'T'])]
    initializers=[nh.from_array(a,n) for n,a in constants.items()]
    common=dict(domain=DOMAIN,channels=co,native_abi=1,row_batches=0,backend=backend)
    if kind=='triple':
        nodes=[h.make_node('SnakeDW7SnakeF32',['x','w','b','ap','rp','ao','ro'],['y'],
                          dilation=dilation,require_vector_sine=1,**common)]
    elif kind=='chain':
        nodes=[h.make_node('SnakeF32',['x','ap','rp'],['pre'],require_vector_sine=1,**common),
               h.make_node('CausalDW7SnakeF32',['pre','w','b','ao','ro'],['y'],
                           dilation=dilation,require_vector_sine=1,**common)]
    else:
        inputs.append(h.make_tensor_value_info('skip',TP.FLOAT,['B',co,'T']))
        if kind=='bias':
            nodes=[h.make_node('BiasResidualF32',['x','b','skip'],['y'],**common)]
        else:
            initializers.append(nh.from_array(constants['b'].reshape(1,co,1),'bias_bct'))
            nodes=[h.make_node('Add',['x','bias_bct'],['biased']),
                   h.make_node('Add',['skip','biased'],['y'])]
    return h.make_model(h.make_graph(nodes,'candidate_parity',inputs,
        [h.make_tensor_value_info('y',TP.FLOAT,['B',co,'T'])],initializers),
        opset_imports=[h.make_opsetid('',20),h.make_opsetid(DOMAIN,1)],ir_version=10)


def bits_equal(a,b):
    np.testing.assert_array_equal(a.view(np.uint32),b.view(np.uint32))


def validate(library,output):
    if platform.machine().lower() not in ('x86_64','amd64'):
        raise RuntimeError('Run these checks on an x86 CPU')
    if ort.__version__!='1.29.0':raise RuntimeError('ONNX Runtime1.29.0 required')
    path=Path(library).resolve();lib=ctypes.CDLL(str(path));ptr=ctypes.c_void_p
    lib.ncc_backend_available.argtypes=[ctypes.c_int32];lib.ncc_backend_available.restype=ctypes.c_int32
    lib.ncc_selected_backend.argtypes=[];lib.ncc_selected_backend.restype=ctypes.c_int32
    lib.ncc_capabilities.argtypes=[];lib.ncc_capabilities.restype=ctypes.c_uint64
    lib.ncc_snake_math_name.argtypes=[ctypes.c_int32];lib.ncc_snake_math_name.restype=ctypes.c_char_p
    backends=[b for b in (3,4,5) if lib.ncc_backend_available(b)]
    if not backends:raise RuntimeError('SSE2 or newer required')
    selected=lib.ncc_selected_backend()
    if selected==5:raise AssertionError('Experimental AVX512 must not become AUTO')
    if bool(lib.ncc_capabilities()&256)!=bool(lib.ncc_backend_available(5)):
        raise AssertionError('AVX512 capability/backend disagreement')
    if 5 in backends and lib.ncc_snake_math_name(5).decode()!='sleef-u10-avx512f-fma':
        raise AssertionError('AVX512 requires the accurate named sine path')
    triple=lib.ncc_snake_dw7_snake_f32
    triple.argtypes=[ptr]*8+[ctypes.c_int64]*3+[ctypes.c_int32]*3;triple.restype=ctypes.c_int32
    bias=lib.ncc_bias_residual_f32
    bias.argtypes=[ptr]*4+[ctypes.c_int64]*3+[ctypes.c_int32]*2;bias.restype=ctypes.c_int32
    rng=np.random.default_rng(2026090719);records=[];jobs=[]
    lengths=(0,1,6,7,15,16,17,53,54,55,255,256,257,511,512,513,777)
    for threads in (1,2):
      for backend in backends:
       for co in (1,3):
        for dilation in (1,3,9):
          ap=np.exp(rng.normal(0,.5,co)).astype(np.float32)
          ao=np.exp(rng.normal(0,.5,co)).astype(np.float32)
          constants={'w':rng.normal(0,.15,(co,1,7)).astype(np.float32),
                     'b':rng.normal(0,.1,co).astype(np.float32),
                     'ap':ap,'rp':(1/(ap+np.float32(1e-8))).astype(np.float32),
                     'ao':ao,'ro':(1/(ao+np.float32(1e-8))).astype(np.float32)}
          new=session(graph('triple',co,constants,backend,dilation),path,threads)
          old=session(graph('chain',co,constants,backend,dilation),path,threads)
          if new.get_providers()!=['CPUExecutionProvider']:raise AssertionError('CPU provider only')
          first=None;long=rng.normal(0,.4,(2,co,777)).astype(np.float32)
          for t in (*lengths,1,777):
            x=long if t==777 else rng.normal(0,.4,(2,co,t)).astype(np.float32)
            unchanged=x.copy();feeds={'x':x}
            actual=new.run(None,feeds)[0];expected=old.run(None,feeds)[0]
            bits_equal(actual,expected);bits_equal(x,unchanged)
            if t==777:
              if first is None:first=actual.copy()
              else:bits_equal(actual,first)
              future=x.copy();future[...,259:]+=np.float32(1.25)
              changed=new.run(None,{'x':future})[0]
              bits_equal(actual[...,:259],changed[...,:259])
              jobs.append((new,{'x':x},actual.copy()))
            records.append({'op':'SnakeDW7SnakeF32','threads':threads,'backend':backend,
                            'channels':co,'dilation':dilation,'time':t,'bitwise':True})
          empty=np.empty((0,co,17),np.float32)
          if new.run(None,{'x':empty})[0].shape!=empty.shape:raise AssertionError('Empty batch shape')
        b=rng.normal(0,1,co).astype(np.float32);b[0]=np.float32(1e20)
        new=session(graph('bias',co,{'b':b},backend),path,threads)
        old=session(graph('bias_reference',co,{'b':b},backend),path,threads)
        for t in lengths:
          product=rng.normal(0,1,(2,co,t)).astype(np.float32)
          skip=rng.normal(0,1,product.shape).astype(np.float32)
          if t:
            product[:,0,0]=np.float32(-1e20);skip[:,0,0]=np.float32(3.25)
          feeds={'x':product,'skip':skip};actual=new.run(None,feeds)[0]
          expected=np.add(skip,np.add(product,b[None,:,None],dtype=np.float32),dtype=np.float32)
          bits_equal(actual,expected);bits_equal(actual,old.run(None,feeds)[0])
          if t and not np.all(actual[:,0,0]==np.float32(3.25)):raise AssertionError('Addition reassociated')
          records.append({'op':'BiasResidualF32','threads':threads,'backend':backend,
                          'channels':co,'time':t,'bitwise':True})
        # The same immutable input is allowed for both read-only operands.
        x=rng.normal(0,1,(2,co,257)).astype(np.float32)
        expected=np.add(x,np.add(x,b[None,:,None],dtype=np.float32),dtype=np.float32)
        bits_equal(new.run(None,{'x':x,'skip':x})[0],expected)
       for signed_bias in (0.0,-0.0):
        z=np.array([signed_bias],np.float32)
        zeros=session(graph('bias',1,{'b':z},backend),path,threads)
        product=np.array([0.,-0.,0.,-0.]*65,np.float32).reshape(1,1,-1)
        skip=np.array([0.,0.,-0.,-0.]*65,np.float32).reshape(1,1,-1)
        expected=np.add(skip,np.add(product,z[None,:,None],dtype=np.float32),dtype=np.float32)
        bits_equal(zeros.run(None,{'x':product,'skip':skip})[0],expected)
        records.append({'op':'BiasResidualF32','threads':threads,'backend':backend,
                        'signed_zero_bias':str(signed_bias),'bitwise':True})
    # Same and different sessions execute concurrently; no mutable history or
    # shared coefficient buffers may leak between calls.
    parallel_jobs=jobs[:6]+jobs[-6:]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda j:j[0].run(None,j[1])[0],parallel_jobs))
    for actual,j in zip(results,parallel_jobs):bits_equal(actual,j[2])
    safety=[]
    w=np.ones(7,np.float32);c=np.ones(1,np.float32);x=np.ones(32,np.float32);y=np.empty_like(x)
    targs=[x.ctypes.data,w.ctypes.data,c.ctypes.data,c.ctypes.data,c.ctypes.data,c.ctypes.data,c.ctypes.data,y.ctypes.data,1,1,32,1,selected,1]
    bargs=[x.ctypes.data,c.ctypes.data,x.ctypes.data,y.ctypes.data,1,1,32,selected,1]
    def expect(fn,args,status,label):
        got=fn(*args)
        if got!=status:raise AssertionError((label,got,status))
        safety.append(label)
    for fn,args,output_index,batch_index,time_index,backend_index,threads_index in (
            (triple,targs,7,8,10,12,13),(bias,bargs,3,4,6,7,8)):
        a=args.copy();a[batch_index]=0
        for i in range(output_index+1):a[i]=None
        expect(fn,a,0,'empty_noop')
        a=args.copy();a[batch_index]=-1;expect(fn,a,1,'negative_dimension')
        a=args.copy();a[time_index]=2**62;expect(fn,a,4,'size_overflow')
        a=args.copy();a[0]+=1;expect(fn,a,1,'misaligned_input')
        a=args.copy();a[threads_index]=2;expect(fn,a,5,'nested_threads_rejected')
        a=args.copy();a[backend_index]=99;expect(fn,a,2,'unknown_backend')
        for i in range(output_index):
            a=args.copy();a[time_index]=1;a[output_index]=a[i]
            expect(fn,a,3,'output_input_overlap_'+str(i))
        for i in range(output_index):
            a=args.copy();a[i]=None;expect(fn,a,1,'null_required_'+str(i))
    for d in (0,2,10,2**31-1):
        a=targs.copy();a[11]=d;expect(triple,a,1,'unsupported_dilation_'+str(d))
    if 5 not in backends:
        a=targs.copy();a[12]=5;expect(triple,a,2,'avx512_fail_closed')
        a=bargs.copy();a[7]=5;expect(bias,a,2,'avx512_bias_fail_closed')
    # Deliberately malformed custom-op contracts must fail at creation or run.
    for kind in ('triple','bias'):
        co=1;constants={'b':np.ones(2,np.float32)}
        if kind=='triple':constants.update(w=np.ones((1,1,7),np.float32),ap=c,rp=c,ao=c,ro=c)
        try:session(graph(kind,co,constants,selected),path,1)
        except Exception:safety.append('bad_constant_shape_'+kind)
        else:raise AssertionError('Malformed coefficient shape accepted')
    answer={'all_passed':True,'scope':'Small CPU operators only, no model or timing',
            'library_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'onnxruntime':ort.__version__,
            'providers':['CPUExecutionProvider'],'selected_backend':selected,'tested_backends':backends,
            'operator_checks':len(records),'concurrent_checks':len(results),'safety_checks':len(safety),
            'bitwise_vs_unfused_same_backend':True,'records':records,'safety':safety}
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(answer,indent=2)+'\n')
    return {k:v for k,v in answer.items() if k not in ('records','safety')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(validate(a.library,a.output),indent=2))
