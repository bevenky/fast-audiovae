"""Bounded CPU-only adapter fixtures; no codec/model benchmarks."""
import argparse,concurrent.futures,hashlib,json,os
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='-1',NVIDIA_VISIBLE_DEVICES='void',ROCR_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh,TensorProto as T
import onnxruntime as ort
PIN='25cad99a6840855ade0a49871197f48ee0e1d317'
def session(w,library,threads,tiles):
 m,k=w.shape
 node=h.make_node('PackedRowsMatMulF32',['w','x'],['y'],domain='venky.audio.cpu.aocl.rows',rows=m,inner=k,row_tiles=tiles,native_abi=1,aocl_pin=PIN)
 g=h.make_graph([node],'tiny_aocl',[h.make_tensor_value_info('x',T.FLOAT,['B',k,'N'])],[h.make_tensor_value_info('y',T.FLOAT,['B',m,'N'])],[nh.from_array(w,'w')])
 model=h.make_model(g,opset_imports=[h.make_opsetid('',20),h.make_opsetid('venky.audio.cpu.aocl.rows',1)],ir_version=10)
 so=ort.SessionOptions();so.intra_op_num_threads=threads;so.inter_op_num_threads=1;so.add_session_config_entry("session.intra_op.allow_spinning","0");so.add_session_config_entry("session.inter_op.allow_spinning","0");so.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;so.log_severity_level=3
 so.register_custom_ops_library(library)
 return ort.InferenceSession(model.SerializeToString(),so,providers=['CPUExecutionProvider'])
def test(library,output):
 if ort.__version__ != "1.29.0":raise RuntimeError("Validation requires ONNX Runtime 1.29.0")
 rng=np.random.default_rng(71019);records=[];concurrent_count=0
 for threads in (1,4):
  for tiles in (1,4):
   for m,k in ((7,13),(64,128),(129,257)):
    ws=[rng.normal(0,.1,(m,k)).astype(np.float32) for _ in range(2)]
    sessions=[session(w,library,threads,tiles) for w in ws]
    xs={n:rng.normal(0,.3,(2,k,n)).astype(np.float32) for n in (1,7,17,170,1025)}
    expected={(i,n):(w.astype(np.float64)@x.astype(np.float64)).astype(np.float32) for i,w in enumerate(ws) for n,x in xs.items()}
    for n in (1025,1,7,17,170,1025):
     first=None
     for i in (0,1,0):
      y=sessions[i].run(None,{'x':xs[n]})[0];ref=expected[i,n]
      np.testing.assert_allclose(y,ref,atol=1e-5,rtol=1e-4)
      if i==0:
       if first is not None:np.testing.assert_array_equal(y,first)
       first=y.copy()
      records.append({'threads':threads,'tiles':tiles,'M':m,'K':k,'N':n,'batch':2,'session':i,'max_abs_error':float(np.max(np.abs(y-ref)))})
    def concurrent_check(j):
     i=j%2;n=(1,17,170,1025)[j%4]
     y=sessions[i].run(None,{'x':xs[n]})[0]
     np.testing.assert_allclose(y,expected[i,n],atol=1e-5,rtol=1e-4)
     return True
    with concurrent.futures.ThreadPoolExecutor(4) as pool:concurrent_count+=sum(pool.map(concurrent_check,range(12)))
 if len(records)!=216 or concurrent_count!=144:raise AssertionError('Incomplete packed-matrix fixture coverage')
 result={'library':library,'sha256':hashlib.sha256(Path(library).read_bytes()).hexdigest(),'ort':ort.__version__,'checks':len(records),'concurrent_checks':concurrent_count,
         'all_passed':True,'max_abs_error':max(v['max_abs_error'] for v in records),'tolerance':{'atol':1e-5,'rtol':1e-4},'records':records,
         'scope':'Small row-partitioned matrix fixtures only; no codec model or timing. Packed W reused across N and concurrent sessions.'}
 Path(output).write_text(json.dumps(result,indent=2)+'\n');return {k:v for k,v in result.items() if k!='records'}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library',required=True);p.add_argument('--output',required=True)
 print(json.dumps(test(**vars(p.parse_args())),indent=2))
