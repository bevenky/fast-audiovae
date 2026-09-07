"""CPU-only parity and isolated (MatMul + bias + skip) timings. Never decoder RTF."""
from __future__ import annotations
import argparse
import concurrent.futures
import ctypes as ct
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import time

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()

class Kernel:
    def __init__(self, path):
        self.lib = ct.CDLL(str(path.resolve()))
        self.lib.fx_init.restype = ct.c_int
        self.lib.fx_has_xsmm.restype = ct.c_int
        self.lib.fx_create.argtypes = [ct.c_int]*7
        self.lib.fx_create.restype = ct.c_void_p
        self.lib.fx_blocks.argtypes = [ct.c_void_p]
        self.lib.fx_blocks.restype = ct.c_int
        self.lib.fx_isa.argtypes = [ct.c_void_p]
        self.lib.fx_isa.restype = ct.c_int
        self.lib.fx_run_range.argtypes = [ct.c_void_p]*6+[ct.c_int]*2
        self.lib.fx_run_range.restype = ct.c_int
        self.lib.fx_destroy.argtypes = [ct.c_void_p]
        self.lib.fx_destroy.restype = None
        self.isa = self.lib.fx_init()
        if self.isa < 0: raise RuntimeError('Remove forced LIBXSMM_TARGET')
        self.xsmm = bool(self.lib.fx_has_xsmm())

    def plan(self, w, x, bias, skip, y, mode, isa, tt, tc, threads, pool):
        c, k = w.shape
        t = x.shape[-1]
        p = self.lib.fx_create(c, k, t, tt, tc, mode, isa)
        if not p: raise RuntimeError(f'Unsupported plan {c,k,t,tt,tc,mode,isa}')
        count = self.lib.fx_blocks(p)
        args = (p, w.ctypes.data, x.ctypes.data, bias.ctypes.data,
                skip.ctypes.data if skip is not None else None, y.ctypes.data)
        def worker(first, last):
            status = self.lib.fx_run_range(*args, first, last)
            if status: raise RuntimeError(f'Native kernel failed: {status}')
        ranges = [(i*count//threads, (i+1)*count//threads) for i in range(threads)]
        def run():
            if threads == 1: worker(0, count)
            else:
                futures = [pool.submit(worker, lo, hi) for lo, hi in ranges]
                for future in futures: future.result()
        return p, run, {'mode': mode, 'isa': self.lib.fx_isa(p), 'tile_time': tt,
                        'tile_channels': tc, 'blocks': count, 'threads': threads}

def check(actual, ref, atol=2e-5, rtol=2e-5):
    passed = True; maxabs = energy = error = 0.0
    aa, rr = actual.reshape(-1), ref.reshape(-1)
    for i in range(0, aa.size, 262144):
        a = aa[i:i+262144].astype(np.float64); r = rr[i:i+262144].astype(np.float64)
        d = a-r
        passed = passed and bool(np.all(np.isfinite(a))) and bool(np.all(np.abs(d)<=atol+rtol*np.abs(r)))
        maxabs = max(maxabs, float(np.max(np.abs(d), initial=0)))
        energy += float(np.sum(r*r)); error += float(np.sum(d*d))
    return {'pass': passed, 'max_abs': maxabs, 'relative_l2': (error/max(energy,1e-300))**0.5,
            'atol': atol, 'rtol': rtol}

def ort_callable(w, x, bias, skip, y, threads):
    c, k = w.shape; t = x.shape[-1]
    init = [nh.from_array(w, 'w'), nh.from_array(bias.reshape(1,c,1), 'bias')]
    inputs = [oh.make_tensor_value_info('x', tp.FLOAT, [1,k,t])]
    nodes = [oh.make_node('MatMul', ['w','x'], ['dot']),
             oh.make_node('Add', ['dot','bias'], ['plus_bias' if skip is not None else 'y'])]
    if skip is not None:
        inputs.append(oh.make_tensor_value_info('skip', tp.FLOAT, [1,c,t]))
        nodes.append(oh.make_node('Add', ['plus_bias','skip'], ['y']))
    model = oh.make_model(oh.make_graph(nodes,'ordered_pointwise',inputs,
                         [oh.make_tensor_value_info('y',tp.FLOAT,[1,c,t])],init),
                          opset_imports=[oh.make_opsetid('',20)], ir_version=10)
    so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.add_session_config_entry('session.intra_op.allow_spinning','0')
    so.add_session_config_entry('session.inter_op.allow_spinning','0')
    session = ort.InferenceSession(model.SerializeToString(), so, providers=['CPUExecutionProvider'])
    if session.get_providers() != ['CPUExecutionProvider']: raise RuntimeError('Unexpected provider')
    binding = session.io_binding(); values = []
    for name, arr in [('x',x), ('skip',skip)]:
        if arr is not None:
            value = ort.OrtValue.ortvalue_from_numpy(arr); values.append(value)
            binding.bind_ortvalue_input(name,value)
    out = ort.OrtValue.ortvalue_from_numpy(y); values.append(out)
    binding.bind_ortvalue_output('y',out)
    return lambda: session.run_with_iobinding(binding), (session,binding,values,w,x,bias,skip,y)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path,help='Existing actual_t170 manifest with weights and input arrays')
    p.add_argument('--threads',type=int,default=1,choices=[1,2,4])
    p.add_argument('--cpus',help='Linux allowed CPU IDs; count must equal threads')
    p.add_argument('--tiny-only',action='store_true')
    p.add_argument('--real-shapes',action='store_true',help='Synthetic wide-time shapes if no manifest')
    p.add_argument('--repeats',type=int,default=7)
    p.add_argument('--warmups',type=int,default=2)
    p.add_argument('--tile-time',type=int,help='Override XSMM tile, otherwise 8192/K')
    p.add_argument('--tile-channels',type=int,default=32)
    p.add_argument('--seed',type=int,default=91427)
    a = p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',NVIDIA_VISIBLE_DEVICES='void',
                      ROCR_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1',
                      VECLIB_MAXIMUM_THREADS='1',OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    if 'LIBXSMM_TARGET' in os.environ: raise RuntimeError('Remove forced LIBXSMM_TARGET')
    cpus = None
    if a.cpus:
        cpus = [int(v) for v in a.cpus.split(',')]
        if len(set(cpus)) != a.threads or len(cpus) != a.threads: raise ValueError('CPU count mismatch')
        if not hasattr(os,'sched_setaffinity'): raise ValueError('CPU affinity is Linux-only')
        if not set(cpus)<=os.sched_getaffinity(0): raise ValueError('CPU outside allowed affinity')
        os.sched_setaffinity(0,cpus)
    global np, ort, oh, nh, tp
    import numpy as np
    import onnxruntime as ort
    from onnx import helper as oh, numpy_helper as nh, TensorProto as tp
    if ort.__version__!='1.29.0': raise RuntimeError('Expected ORT 1.29.0')
    kernel = Kernel(a.library); rng = np.random.default_rng(a.seed); ordering = random.Random(a.seed)
    variants = [('direct_auto',0,0)]
    if kernel.isa == 512: variants.append(('direct_avx2',0,256))
    if kernel.xsmm: variants += [('xsmm_cache_fused',1,0),('xsmm_separate_passes',2,0)]
    result = {'scope':'Isolated FP32 MatMul then bias then skip. Not a decoder RTF or quality claim.',
              'library_sha256':sha(a.library),'native_isa':kernel.isa,'xsmm':kernel.xsmm,
              'platform':platform.platform(),'ort':ort.__version__,'threads':a.threads,'cpus':cpus,
              'protocol':{'repeats':a.repeats,'warmups':a.warmups,'seed':a.seed,
              'allocation_and_jit_outside_timing':True,'output_reuse':True,'full_k_reduction':True,
              'nested_threadpools':False,'caller_cost':'ORT IOBinding vs ctypes and persistent Python executor',
              'model_executed':False,'only_cpu_provider':True,'cold_cache_flush':False},
              'tiny':[],'matrices':[]}
    def save():
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,indent=2)+'\n')
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.threads) as pool:
        def one(w,x,bias,skip,label,timed):
            c=w.shape[0]; t=x.shape[-1]
            ref=np.empty((1,c,t),dtype=np.float32); candidate=np.empty_like(ref)
            ort_call, keep=ort_callable(w,x,bias,skip,ref,a.threads); ort_call()
            plans=[]; methods={'ort':ort_call}; info={}; checks={}
            try:
                for name, mode, isa in variants:
                    tt=a.tile_time or (8192//c if mode else 64)
                    plan,run,details=kernel.plan(w,x,bias,skip,candidate,mode,isa,tt,
                                                min(a.tile_channels,c),a.threads,pool)
                    plans.append(plan); methods[name]=run; info[name]=details
                    candidate.fill(np.nan); run(); checks[name]=check(candidate,ref)
                row={'case':label,'C':c,'T':t,'skip':skip is not None,'checks':checks,'settings':info}
                (result['matrices'] if timed else result['tiny']).append(row)
                if not all(item['pass'] for item in checks.values()):
                    save(); raise RuntimeError('Numerical gate failed: '+label)
                if timed:
                    for _ in range(a.warmups):
                        names=list(methods); ordering.shuffle(names)
                        for name in names: methods[name]()
                    samples={name:[] for name in methods}; orders=[]
                    was_gc=gc.isenabled(); gc.disable()
                    try:
                        for _ in range(a.repeats):
                            names=list(methods); ordering.shuffle(names); orders.append(names)
                            for name in names:
                                start=time.perf_counter_ns(); methods[name]()
                                samples[name].append(time.perf_counter_ns()-start)
                    finally:
                        if was_gc: gc.enable()
                    row['median_ms']={name:statistics.median(v)/1e6 for name,v in samples.items()}
                    row['samples_ns']=samples; row['trial_orders']=orders
                    print(json.dumps({k:row[k] for k in ('case','median_ms')}),flush=True)
                save()
            finally:
                for plan in plans: kernel.lib.fx_destroy(plan)
        for c in (32,64,128,256):
            for t in (1,17,63,64,65,257):
                w=rng.uniform(-.1,.1,(c,c)).astype(np.float32)
                x=rng.uniform(-.1,.1,(1,c,t)).astype(np.float32)
                bias=rng.uniform(-.1,.1,c).astype(np.float32)
                skip=rng.uniform(-.1,.1,(1,c,t)).astype(np.float32)
                one(w,x,bias,skip,f'tiny_c{c}_t{t}',False)
        # Adversarial cancellation catches bias seeded before reduction and bias+skip reassociation.
        for sentinel in ('bias_after_dot','skip_after_bias'):
            c,t=32,65; w=np.zeros((c,c),np.float32); x=np.ones((1,c,t),np.float32)
            bias=np.ones(c,np.float32); skip=np.ones((1,c,t),np.float32)
            if sentinel=='bias_after_dot': w[:,0]=1e20; w[:,1]=-1e20
            else: w[:,0]=1e20; bias.fill(-1e20)
            one(w,x,bias,skip,sentinel,False)
        # Exercise the bias-only operator contract too.
        one(w*0,x,np.ones(c,np.float32),None,'bias_only',False)
        print(json.dumps({'tiny_pass':True,'cases':len(result['tiny'])}),flush=True)
        if not a.tiny_only:
            if a.manifest:
                manifest=json.loads(a.manifest.read_text()); result['manifest_sha256']=sha(a.manifest)
                for item in manifest['matrices']:
                    if item['M'] not in (32,64,128,256) or item['M']!=item['K']: continue
                    root=a.manifest.parent
                    for file,expected in item['sha256'].items():
                        if sha(root/file)!=expected: raise RuntimeError('Capture hash mismatch: '+file)
                    w=np.load(root/item['weight'],allow_pickle=False)
                    x=np.load(root/item['input'],allow_pickle=False)
                    if any(v.dtype!=np.float32 or not v.flags.c_contiguous for v in (w,x)):
                        raise ValueError('Expected contiguous captured FP32 inputs')
                    c=w.shape[0]
                    bias=rng.uniform(-.1,.1,c).astype(np.float32)
                    skip=rng.uniform(-.1,.1,(1,c,x.shape[-1])).astype(np.float32)
                    one(w,x,bias,skip,item['id']+'_real_W_X_synthetic_bias_skip',True)
            elif a.real_shapes:
                for c,t in ((256,40800),(128,81600),(64,163200),(32,326400)):
                    w=rng.uniform(-.1,.1,(c,c)).astype(np.float32)
                    x=rng.uniform(-.1,.1,(1,c,t)).astype(np.float32)
                    bias=rng.uniform(-.1,.1,c).astype(np.float32)
                    skip=rng.uniform(-.1,.1,(1,c,t)).astype(np.float32)
                    one(w,x,bias,skip,f'synthetic_c{c}_t{t}',True)
            else: raise ValueError('Choose --manifest, --real-shapes, or --tiny-only')
    save(); print(json.dumps({'output':str(a.output),'all_checks_pass':True}),flush=True)

if __name__=='__main__': main()
