"""Attribute the real fixed Hindi prefix. Profiles are separate from RTF."""
import argparse,ctypes as ct,hashlib,json,os,collections
from pathlib import Path
for key,value in {'CUDA_VISIBLE_DEVICES':'-1','NVIDIA_VISIBLE_DEVICES':'void','HIP_VISIBLE_DEVICES':'-1','ROCR_VISIBLE_DEVICES':'-1',
                  'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'}.items():os.environ[key]=value
import numpy as np
import onnxruntime as ort
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--config-sha256',required=True);p.add_argument('--profile-build',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists() and sha(a.config)==a.config_sha256 and ort.__version__=='1.29.0'
    cfg=json.loads(a.config.read_text());build=json.loads(a.profile_build.read_text());resolve=lambda v:(a.config.parent/v).resolve()
    for path,digest in cfg['artifact_sha256'].items():assert sha(resolve(path))==digest
    for name,digest in build['libraries'].items():assert sha(a.profile_build.parent/name)==digest
    uid=cfg['timing_uids'][0]
    with np.load(resolve(cfg['audio_cases']),allow_pickle=False) as archive:z=np.ascontiguousarray(archive[uid+'__z'][...,:170])
    a.output.mkdir(parents=True)
    result={'scope':'Separate diagnostic profiles, not decoder RTF; last of two real forwards',
            'config_sha256':a.config_sha256,'profile_build_sha256':sha(a.profile_build),'script_sha256':sha(__file__),
            'GPU_used':False,'ORT':ort.__version__,'threads':4,'uid':uid,'latent_shape':list(z.shape),'latent_sha256':sha_array(z),'models':{}}
    for name in ('fast_fp32','int8_large'):
        model=next(m for m in cfg['models'] if m['name']==name)
        options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_session_config_entry('session.intra_op.allow_spinning','0');options.add_session_config_entry('session.inter_op.allow_spinning','0')
        options.enable_profiling=True;options.profile_file_prefix=str(a.output/name)
        libs=model.get('custom_libraries',[]) or [model['custom_library']]
        for index,lib in enumerate(libs):
            options.register_custom_ops_library(build['ops_path'] if name=='int8_large' and index==len(libs)-1 else str(resolve(lib)))
        session=ort.InferenceSession(str(resolve(model['path'])),options,providers=['CPUExecutionProvider']);session.disable_fallback()
        assert session.get_providers()==['CPUExecutionProvider']
        core=None
        if name=='int8_large':
            core=ct.CDLL(build['core_path']);core.ipa_profile_reset.argtypes=[];core.ipa_profile_reset.restype=None
            core.ipa_profile_json.argtypes=[];core.ipa_profile_json.restype=ct.c_char_p
        first=session.run(None,{'z':z})[0]
        if core:core.ipa_profile_reset()
        second=session.run(None,{'z':z})[0];assert sha_array(first)==sha_array(second) and sha_array(z)==result['latent_sha256']
        native=json.loads(core.ipa_profile_json()) if core else []
        trace=Path(session.end_profiling());events=json.loads(trace.read_text())
        runs=[e for e in events if e.get('name')=='model_run'];assert len(runs)==2
        last=runs[-1];start=last['ts'];end=start+last['dur']
        kernels=[e for e in events if e.get('cat')=='Node' and e.get('name','').endswith('_kernel_time') and start<=e['ts']<end]
        assert kernels and all(e.get('args',{}).get('provider')=='CPUExecutionProvider' for e in kernels)
        totals=collections.defaultdict(float)
        for e in kernels:totals[e['args'].get('op_name','unknown')]+=e['dur']/1000
        result['models'][name]={'profile_path':str(trace.resolve()),'profile_sha256':sha(trace),'waveform_sha256':sha_array(second),
            'repeat_bitwise':True,'CPU_only':True,'last_model_run_ms':last['dur']/1000,'kernel_ms':sum(e['dur'] for e in kernels)/1000,
            'operator_ms':dict(totals),'native_events':native,'native_sums_ms':{kind:sum(e['ns'] for e in native if e['kind']==kind)/1e6 for kind in sorted({e['kind'] for e in native})},
            'selected_kernels':kernels}
        del session
    (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({name:{k:v for k,v in row.items() if k in ('operator_ms','native_sums_ms','kernel_ms','last_model_run_ms')} for name,row in result['models'].items()},indent=2))
def sha_array(x):return hashlib.sha256(x.tobytes()).hexdigest()
if __name__=='__main__':main()
