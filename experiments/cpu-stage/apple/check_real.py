"""One real-case Apple stage correctness capture. No warmup or timing loops."""
import argparse,copy,gc,hashlib,json,os
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='-1',NVIDIA_VISIBLE_DEVICES='void',ROCR_VISIBLE_DEVICES='-1',
                  HIP_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
import numpy as np
import onnx
from onnx import helper,TensorProto
from check_stage import compare,session

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','candidate','cases','native-library','stage-library','output'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--uid',required=True);p.add_argument('--latent-frames',required=True,type=int)
    a=p.parse_args()
    if a.latent_frames!=170:raise ValueError('This capture uses an explicit L170 prefix')
    audit=json.loads(a.candidate.with_suffix('.stage.json').read_text())
    if sha(a.source)!=audit['source_sha256'] or sha(a.candidate)!=audit['output_sha256']:raise ValueError('Graph audit mismatch')
    with np.load(a.cases,allow_pickle=False) as cases:z=cases[a.uid+'__z']
    original_shape=list(z.shape);original_hash=hashlib.sha256(z.tobytes()).hexdigest()
    if z.dtype!=np.float32 or z.ndim!=3 or z.shape[:2]!=(1,64) or z.shape[-1]<170:raise ValueError('Real case cannot supply L170 prefix')
    z=np.ascontiguousarray(z[...,:170])
    source=onnx.load(a.source);candidate=onnx.load(a.candidate)
    report={'source_sha256':sha(a.source),'candidate_sha256':sha(a.candidate),
            'native_library_sha256':sha(a.native_library),'stage_library_sha256':sha(a.stage_library),
            'cases_sha256':sha(a.cases),'uid':a.uid,'original_latent_shape':original_shape,'original_latent_sha256':original_hash,
            'explicit_prefix_frames':170,'latent_sha256':hashlib.sha256(z.tobytes()).hexdigest(),
            'runtime':'1.29.0','provider':'CPUExecutionProvider','threads':2,'backend':2,'matrix_isa':128,
            'gpu_used':False,'timing':False,'full_decoder_waveform_gate':False,'checks':[]}
    initializers={t.name:t for t in candidate.graph.initializer}
    geometry={256:40800,128:81600,64:163200,32:326400}
    if sorted(r['channels'] for r in audit['changes'])!=sorted(geometry):
        raise ValueError('This capture requires exactly the four late stages')
    for r in audit['changes']:
        c=r['channels'];t=geometry[c];m=copy.deepcopy(source)
        names=[r['input'],*r['unit_outputs']];del m.graph.output[:]
        m.graph.output.extend(helper.make_tensor_value_info(name,TensorProto.FLOAT,[1,c,t]) for name in names)
        s=session(m,a.native_library,a.stage_library);values=s.run(None,{s.get_inputs()[0].name:z})
        del s,m;gc.collect()
        x=values[0]
        node=next(copy.deepcopy(n) for n in candidate.graph.node if n.name==r['node'])
        attrs={attr.name:helper.get_attribute_value(attr) for attr in node.attribute}
        required={'native_abi':1,'channels':c,'segments':2,'backend':2,'matrix_mode':0,'matrix_isa':128}
        if any(attrs.get(key)!=value for key,value in required.items()) or x.shape!=(1,c,t):
            raise ValueError('Candidate attributes or captured geometry differ from this check contract')
        node.op_type='StageStackDebugF32';del node.output[:];node.output.extend(r['unit_outputs'])
        # Check the same bounded tile choices proposed for the late x86 stages.
        q=128 if c==128 else 256
        for attr in node.attribute:
            if attr.name=='tile_time':attr.i=q
        info=lambda name:helper.make_tensor_value_info(name,TensorProto.FLOAT,[1,c,t])
        used={v for v in node.input if v in initializers}
        cm=helper.make_model(helper.make_graph([node],'Apple stage check',[info(r['input'])],
               [info(v) for v in r['unit_outputs']],[copy.deepcopy(initializers[v]) for v in sorted(used)]),
               opset_imports=list(candidate.opset_import),ir_version=candidate.ir_version)
        onnx.checker.check_model(cm);s=session(cm,a.native_library,a.stage_library)
        actual=s.run(None,{r['input']:x})
        for u in range(3):
            report['checks'].append({'channels':c,'time':t,'tile_time':q,'segments':2,'unit':u,
                  'captured_input_sha256':hashlib.sha256(x.tobytes()).hexdigest(),
                  'captured_output_sha256':hashlib.sha256(values[u+1].tobytes()).hexdigest(),
                  **compare(actual[u],values[u+1],f'Apple real C{c} unit{u}')})
        del s,cm,x,values,actual;gc.collect()
    report['status']='complete';a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'checks':len(report['checks']),
                      'max_abs':max(v['max_abs'] for v in report['checks']),'timing':False},indent=2))

if __name__=='__main__':main()
