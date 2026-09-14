"""Capture actual second-pair inputs on Intel; arrays stay on that host."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','OMP_THREAD_LIMIT'):
    os.environ[name]='1'
os.environ.update(CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1')
import hashlib,json,time
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort

ROOT=Path('/dev/shm/fast-audiovae-projection-candidate-20260908-r1')
GRAPH=Path('/dev/shm/intel-streaming-transfer-v1-graphs-r2/combined.onnx')
OUT=Path('/dev/shm/intel-library-screen-v1')
UID='bn_in_00151_1818'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    os.sched_setaffinity(0,{0})
    assert ort.__version__=='1.29.0'
    pin=json.loads((GRAPH.parent/'manifest.json').read_text())
    assert sha(GRAPH)==pin['variants']['combined']['sha256']
    assert sha(GRAPH.parent/pin['shared_weights']['file'])==pin['shared_weights']['sha256']
    bundle=json.loads((ROOT/'bundle.json').read_text());native=bundle['native']['Linux/x86_64']
    entry=bundle['streaming']['models'][native['model']]
    libraries=[ROOT/native['library']]
    libraries += [ROOT/x['library'] for x in entry.get('additional_libraries',native['additional_libraries'])]
    libraries += [Path('/var/tmp/intel-streaming-transfer-v1/build')/x for x in ('libintel_rawhistory.so','libintel_phase.so')]
    m=onnx.load(GRAPH);constants={v.name:v for v in m.graph.initializer}
    observed=['view_15','ncc_up_2_split_matmul_current','ncc_up_2_split_matmul_previous_unshifted']
    for name in observed:m.graph.output.append(onnx.helper.make_tensor_value_info(name,onnx.TensorProto.FLOAT,None))
    opts=ort.SessionOptions();opts.intra_op_num_threads=opts.inter_op_num_threads=1;opts.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    opts.log_severity_level=3
    for name in ('session.intra_op.allow_spinning','session.inter_op.allow_spinning'):opts.add_session_config_entry(name,'0')
    for lib in libraries:opts.register_custom_ops_library(str(lib))
    session=ort.InferenceSession(m.SerializeToString(),opts,providers=['CPUExecutionProvider']);session.disable_fallback()
    assert session.get_providers()==['CPUExecutionProvider']
    arrays={f'weight{i}':onnx.numpy_helper.to_array(constants[f'ncc_up_2_split_matmul_{name}_w']).copy() for i,name in enumerate(('current','previous'))}
    archive=Path('/var/tmp/fast-audiovae-20260907/assets/multilingual/fast_audiovae2.npz')
    with np.load(archive,allow_pickle=False) as ar:z=ar[UID+'__z'][...,:6].copy()
    budget=0;calls=0
    for packet in (1,2):
        state={s['input']:np.zeros(s['shape'],s['dtype']) for s in entry['states']}
        for index in range(3):
            feed={entry['latent_input']:np.ascontiguousarray(z[...,index*packet:(index+1)*packet]),**state}
            t=time.perf_counter();v=session.run(None,feed);budget+=time.perf_counter()-t;calls+=1
            assert budget<1 and all(np.isfinite(x).all() for x in v)
            values=dict(zip([o.name for o in session.get_outputs()],v))
            state={s['input']:values[s['output']] for s in entry['states']}
        size=packet*8
        arrays[f'input_{size}']=np.ascontiguousarray(values['view_15'].reshape(1024,size))
        for i,name in enumerate(observed[1:]):arrays[f'expected{i}_{size}']=np.ascontiguousarray(values[name].reshape(3072,size))
    path=OUT/'matrix-inputs.npz';np.savez(path,**arrays)
    report={'baseline_graph':str(GRAPH),'baseline_graph_sha256':sha(GRAPH),'weights_sha256':pin['shared_weights']['sha256'],
            'native_calls':calls,'native_seconds':budget,'cpu_only':True,'threads':1,'uid':UID,'ort':ort.__version__,
            'capture_npz':str(path),'capture_sha256':sha(path),'shapes':{k:list(v.shape) for k,v in arrays.items()},
            'libraries':{str(x):sha(x) for x in libraries},'scope':'Two warm packets then third packet; original weights and actual second-pair outputs; arrays remain remote'}
    (OUT/'capture.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
