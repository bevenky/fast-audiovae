"""Untimed CPU-only dynamic-shape/concurrency checks for the ORT custom op."""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import os
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',type=int,choices=[0,1,2],default=0)
    p.add_argument('--isa',type=int,choices=[0,1,128,256,512],default=0)
    p.add_argument('--threads',type=int,choices=[1,2,4],default=2)
    a=p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',NVIDIA_VISIBLE_DEVICES='void',
                      ROCR_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    import numpy as np
    import onnxruntime as ort
    from onnx import helper as h,numpy_helper as nh,TensorProto as tp
    import probe
    probe.np=np
    if ort.__version__!='1.29.0':raise RuntimeError('Expected ORT1.29.0')
    rng=np.random.default_rng(617231);results=[]
    for c in (32,64,128,256):
        w=rng.uniform(-.1,.1,(c,c)).astype(np.float32)
        bias=rng.uniform(-.1,.1,c).astype(np.float32)
        inputs=[h.make_tensor_value_info('x',tp.FLOAT,[1,c,'T']),h.make_tensor_value_info('skip',tp.FLOAT,[1,c,'T'])]
        outputs=[h.make_tensor_value_info('y',tp.FLOAT,[1,c,'T'])]
        ref=h.make_model(h.make_graph([
            h.make_node('MatMul',['w','x'],['dot']),h.make_node('Add',['dot','b3'],['biased']),
            h.make_node('Add',['skip','biased'],['y'])],'ref',inputs,outputs,
            [nh.from_array(w,'w'),nh.from_array(bias.reshape(1,c,1),'b3')]),
            ir_version=10,opset_imports=[h.make_opsetid('',20)])
        candidate=h.make_model(h.make_graph([
            h.make_node('PointwiseBiasResidualF32',['w','x','b','skip'],['y'],
                domain='audio.cpu.pointwise.experimental',native_abi=1,channels=c,
                tile_time=64 if a.mode==0 else 8192//c,tile_channels=32,mode=a.mode,
                isa=a.isa,skip_first=1,blocks_per_task=8)],'candidate',inputs,outputs,
            [nh.from_array(w,'w'),nh.from_array(bias,'b')]),ir_version=10,
            opset_imports=[h.make_opsetid('',20),h.make_opsetid('audio.cpu.pointwise.experimental',1)])
        def session(model,custom):
            so=ort.SessionOptions();so.intra_op_num_threads=a.threads;so.inter_op_num_threads=1
            so.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
            so.add_session_config_entry('session.intra_op.allow_spinning','0')
            so.add_session_config_entry('session.inter_op.allow_spinning','0')
            if custom:so.register_custom_ops_library(str(a.library.resolve()))
            s=ort.InferenceSession(model.SerializeToString(),so,providers=['CPUExecutionProvider'])
            if s.get_providers()!=['CPUExecutionProvider']:raise RuntimeError('Unexpected provider')
            return s
        reference=session(ref,False);tested=session(candidate,True)
        feeds=[]
        for t in (1,17,63,64,65,257,96,1,257,1024):
            feeds.append({'x':rng.uniform(-.1,.1,(1,c,t)).astype(np.float32),
                          'skip':rng.uniform(-.1,.1,(1,c,t)).astype(np.float32)})
        references=[reference.run(None,feed)[0] for feed in feeds]
        for feed,ref_y in zip(feeds,references):
            check=probe.check(tested.run(None,feed)[0],ref_y)
            results.append({'C':c,'T':feed['x'].shape[-1],'concurrent':False,**check})
        # One session, changing dimensions and simultaneous calls stress shape-cache ownership.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(tested.run,None,feed) for feed in feeds]
            for feed,ref_y,future in zip(feeds,references,futures):
                check=probe.check(future.result()[0],ref_y)
                results.append({'C':c,'T':feed['x'].shape[-1],'concurrent':True,**check})
    passed=all(r['pass'] for r in results)
    report={'scope':'Untimed isolated custom-op parity, dynamic dimensions and concurrent runs',
            'mode':a.mode,'isa':a.isa,'threads':a.threads,'checks':results,'pass':passed,
            'rtf_measured':False,'codec_model_executed':False,'only_cpu_provider':True}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'output':str(a.output),'pass':passed,'checks':len(results)}))
    if not passed:raise RuntimeError('Bridge parity failed; do not benchmark or promote')

if __name__=='__main__':main()
