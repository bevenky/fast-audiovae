"""CPU-only correctness of the approved two-to-four-worker scheduling change."""
import argparse,copy,hashlib,importlib.util,json,os,platform
from pathlib import Path

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source-config','candidate-config','harness','output'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('source-config-sha256','candidate-config-sha256','harness-sha256'):p.add_argument('--'+k,required=True)
    a=p.parse_args();require(not a.output.exists(),'Fresh report required');require(sha(a.harness)==a.harness_sha256,'Harness mismatch')
    require(platform.system()=='Linux' and 'AuthenticAMD' in Path('/proc/cpuinfo').read_text(),'AMD Linux required')
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):require(os.environ.get(k)=='1',k+' must be one')
    require({0,1,2,3}<=os.sched_getaffinity(0),'Four original AMD CPUs required');os.sched_setaffinity(0,{0,1,2,3})
    spec=importlib.util.spec_from_file_location('public_harness',a.harness);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h);environment=h.require_cpu_environment()
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import numpy_helper
    import stage_reference as ref
    require(ort.__version__=='1.29.0','ORT1.29 required');h.np=np;h.ort=ort
    source,pins0=h.read_config(a.source_config,a.source_config_sha256);cfg,pins1=h.read_config(a.candidate_config,a.candidate_config_sha256)
    require(source['expected_uids']==cfg['expected_uids'] and source['timing_uids']==cfg['timing_uids'],'Cohort changed')
    resolve=lambda path:(a.candidate_config.resolve().parent/path).resolve()
    sm=next(m for m in source['models'] if m['name']=='int8_large');cm=next(m for m in cfg['models'] if m['name']=='int8_large')
    require(sm['custom_libraries']==cm['custom_libraries'],'Library changed during scheduling comparison')
    require(sha(resolve(sm['path']))=='26b545641a43380b2f012c44f600f9e9f4413f424402bb35d9aa758b45849304','Original graph differs')
    libraries=[resolve(x) for x in cm['custom_libraries']]
    def session(model):
        so=ort.SessionOptions();so.intra_op_num_threads=4;so.inter_op_num_threads=1;so.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL;so.log_severity_level=3
        so.add_session_config_entry('session.intra_op.allow_spinning','0');so.add_session_config_entry('session.inter_op.allow_spinning','0')
        for lib in libraries:so.register_custom_ops_library(str(lib))
        s=ort.InferenceSession(model if isinstance(model,bytes) else str(model),so,providers=['CPUExecutionProvider']);s.disable_fallback()
        require(s.get_providers()==['CPUExecutionProvider'],'CPU provider required');return s
    records=[]
    def check(x,y,label,**extra):
        result=h.compare(x,y);require(result['passed'],label+' failed original FP32 tolerance')
        records.append({'test':label,**extra,**result,'bitwise_equal':x.shape==y.shape and np.array_equal(x.view(np.uint32),y.view(np.uint32))})
    s2=session(resolve(sm['path']));s4=session(resolve(cm['path']))
    cases,_=h.read_cases(resolve(cfg['audio_cases']),cfg['kind_contracts']['audio'])
    def run(s,z):
        y=s.run(None,{s.get_inputs()[0].name:z})[0]
        require(y.dtype==np.float32 and y.shape==(1,1,z.shape[-1]*1920) and np.isfinite(y).all(),'Invalid waveform');return y
    for uid in [cfg['timing_uids'][i] for i in (0,6,9)]:
        z=cases[uid]['z'];before=hashlib.sha256(z.tobytes()).hexdigest();check(run(s4,z),run(s2,z),'full_2_vs_4',uid=uid)
        require(hashlib.sha256(z.tobytes()).hexdigest()==before,'Input changed')
    z=np.ascontiguousarray(cases[cfg['timing_uids'][0]]['z'][...,:65])
    for length in (1,2,3,7,8,15,16,17,31,32,33,63,64,65):
        zz=np.ascontiguousarray(z[...,:length]);check(run(s4,zz),run(s2,zz),'short_2_vs_4',length=length)
    # Prove the retained FP32 C64/C32 stages against original native operators
    # and ORT MatMul, using their actual frozen coefficients and adversarial tails.
    model=onnx.load(resolve(cm['path']));initializers={t.name:t for t in model.graph.initializer};rng=np.random.default_rng(20260908)
    stages=[n for n in model.graph.node if n.domain==ref.DOMAIN and n.op_type=='StageStackF32']
    require(len(stages)==2,'Expected exactly two retained FP32 stages')
    for node in stages:
        attrs={v.name:onnx.helper.get_attribute_value(v) for v in node.attribute};c=attrs['channels'];require(c in (32,64) and attrs['segments']==4 and attrs['backend']==5,'Unexpected retained stage')
        original,const=ref.fixture(c,5);values={name:np.asarray(numpy_helper.to_array(initializers[src]),np.float32) for name,src in zip(const,node.input[1:])}
        require(len(values)==24,'Stage constants differ')
        for u in range(3):values[f'u{u}_pb3']=values[f'u{u}_pb'].reshape(1,c,1)
        for index,tensor in enumerate(original.graph.initializer):original.graph.initializer[index].CopyFrom(numpy_helper.from_array(values[tensor.name],tensor.name))
        base=session(original.SerializeToString());two=session(ref.candidate(original,const,c,256,2,5,0,512).SerializeToString());four=session(ref.candidate(original,const,c,256,4,5,0,512).SerializeToString())
        for length in (1,2,3,7,8,15,16,17,53,54,55,77,78,79,255,256,257,513):
            x=rng.normal(0,.25,(1,c,length)).astype(np.float32);y=base.run(None,{'x':x});z2=two.run(None,{'x':x});z4=four.run(None,{'x':x})
            for u in range(3):
                check(z4[u],y[u],'retained_stage_original',channels=c,length=length,unit=u)
                check(z4[u],z2[u],'retained_stage_2_vs_4',channels=c,length=length,unit=u)
        x=rng.normal(0,.25,(1,c,513)).astype(np.float32);other=x.copy();other[...,259:]+=np.float32(.37)
        y=four.run(None,{'x':x});changed=four.run(None,{'x':other})
        for u in range(3):
            require(np.array_equal(y[u][...,:259].view(np.uint32),changed[u][...,:259].view(np.uint32)),'Retained FP32 future dependence')
            records.append({'test':'retained_stage_future','channels':c,'unit':u,'passed':True,'bitwise_equal':True})
            for length in (1,53,54,55,77,78,79,255,256,257):check(four.run(None,{'x':np.ascontiguousarray(x[...,:length])})[u],y[u][...,:length],'retained_stage_prefix',channels=c,unit=u,length=length)
    require(h.read_config(a.source_config,a.source_config_sha256)[1]==pins0 and h.read_config(a.candidate_config,a.candidate_config_sha256)[1]==pins1,'Artifact changed')
    gpu=h.gpu_library_mappings();require(gpu['available'] and not gpu['basenames'],'Unexpected GPU runtime')
    result={'status':'complete','scope':'Correctness only; no timing or perceptual equivalence claim','runtime':'1.29.0','providers':['CPUExecutionProvider'],'gpu_used':False,'gpu_mappings':gpu,'threads':{'ORT':4,'integer_inner':1,'affinity':[0,1,2,3]},'source_config_sha256':a.source_config_sha256,'candidate_config_sha256':a.candidate_config_sha256,'harness_sha256':a.harness_sha256,'script_sha256':sha(Path(__file__)),'reference_helper_sha256':sha(Path(ref.__file__)),'records':records,'record_count':len(records),'passed':all(r['passed'] for r in records),'FP32_tolerance':{'atol':1e-5,'rtol':1e-4},'environment':environment}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'complete','record_count':len(records),'passed':result['passed']}))
if __name__=='__main__':main()
