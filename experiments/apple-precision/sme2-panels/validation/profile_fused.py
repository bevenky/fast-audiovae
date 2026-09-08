"""Frozen Apple FP32/INT8 CPU operator attribution after campaign acceptance.

Two warmups and one profiled forward per model. Profile durations are diagnostic
only; this script never computes RTF or updates a performance decision.
"""
import argparse,ctypes,hashlib,importlib.util,json,os,platform,resource
from pathlib import Path

def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def wave_sha(x):return hashlib.sha256(x.tobytes()).hexdigest()
def inventory():
    lib=ctypes.CDLL(None);lib._dyld_image_count.restype=ctypes.c_uint32
    lib._dyld_get_image_name.argtypes=[ctypes.c_uint32];lib._dyld_get_image_name.restype=ctypes.c_char_p
    images=[lib._dyld_get_image_name(i).decode() for i in range(lib._dyld_image_count())]
    forbidden=('libcuda','libcublas','libcudnn','providers_cuda','providers_tensorrt','librocblas','libamdhip')
    bad=sorted({Path(p).name for p in images if any(v in p.lower() for v in forbidden)})
    require(not bad,'GPU compute library loaded')
    return {'forbidden_compute_libraries':bad,'metal_framework_images':sorted({Path(p).name for p in images if '/Metal.framework/' in p}),
        'scope':'Library presence is not execution evidence; sessions and all custom kernels explicitly use CPU only'}
def summarize(events):
    intervals=[e for e in events if e.get('cat')=='Session' and e.get('name')=='model_run']
    require(len(intervals)==3,'Expected exactly two warmups plus one profiled forward')
    run=intervals[-1];begin=run['ts'];end=begin+run['dur']
    kernels=[e for e in events if e.get('cat')=='Node' and e.get('name','').endswith('_kernel_time') and begin<=e['ts'] and e['ts']+e['dur']<=end]
    require(kernels,'No kernel events inside final model interval')
    require(all(e.get('args',{}).get('provider')=='CPUExecutionProvider' for e in kernels),'Unexpected kernel provider')
    byop={};nodes=[]
    for e in kernels:
        a=e['args'];op=a['op_name'];g=byop.setdefault(op,{'operator':op,'calls':0,'duration_us':0});g['calls']+=1;g['duration_us']+=e['dur']
        nodes.append({'name':e['name'],'operator':op,'duration_us':e['dur'],'timestamp_us':e['ts'],'provider':a['provider'],
            'input_type_shape':a.get('input_type_shape'),'output_type_shape':a.get('output_type_shape'),
            'activation_size':a.get('activation_size'),'parameter_size':a.get('parameter_size'),'output_size':a.get('output_size')})
    total=sum(e['dur'] for e in kernels)
    for g in byop.values():g['percent_kernel_sum']=100*g['duration_us']/total
    return {'selected_model_run':run,'excluded_warmup_intervals':intervals[:2],'kernel_duration_sum_us':total,
        'kernel_events':len(kernels),'operators':sorted(byop.values(),key=lambda g:-g['duration_us']),
        'nodes':sorted(nodes,key=lambda n:-n['duration_us']),
        'denominator':'Sum of nonoverlapping graph kernel events inside the final model interval; not nested worker counters',
        'profiling_overhead_caveat':'ORT profiling changes execution costs. These durations must not be reported as headline latency or RTF.'}
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','campaign-results','harness','output-dir'):p.add_argument('--'+name,required=True,type=Path)
    for name in ('config-sha256','campaign-results-sha256','harness-sha256'):p.add_argument('--'+name,required=True)
    a=p.parse_args()
    require(platform.system()=='Darwin' and platform.machine() in ('arm64','aarch64'),'Apple CPU host required')
    require(sha(a.harness)==a.harness_sha256,'Harness source changed')
    spec=importlib.util.spec_from_file_location('profile_public_harness',a.harness);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    environment=h.require_cpu_environment()
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):require(os.environ.get(name)=='1',name+' must equal1')
    import numpy as np
    import onnxruntime as ort
    require(ort.__version__=='1.29.0','ORT1.29 required');h.np=np;h.ort=ort
    cfg,hashes=h.read_config(a.config,a.config_sha256)
    require(sha(a.campaign_results)==a.campaign_results_sha256,'Campaign result changed')
    campaign=json.loads(a.campaign_results.read_text())
    require(campaign['config_sha256']==a.config_sha256,'Campaign/config mismatch')
    require(campaign.get('status')=='complete','Completed campaign required before profiling')
    require(all(c.get('passed') is True for c in campaign['checks']),'Campaign contains a failed gate')
    uid=cfg['timing_uids'][0];resolve=lambda s:(a.config.parent/s).resolve()
    cases=h.read_cases(resolve(cfg['audio_cases']),cfg['kind_contracts']['audio'])[0]
    z=cases[uid]['z'];require(z.dtype==np.float32 and z.ndim==3 and z.shape[:2]==(1,64),'Unexpected fixed source clip')
    latent_sha=wave_sha(z)
    require(not a.output_dir.exists(),'Fresh output directory required');a.output_dir.mkdir(parents=True)
    record={'status':'running','scope':__doc__,'config_sha256':a.config_sha256,'campaign_results_sha256':a.campaign_results_sha256,
        'harness_sha256':a.harness_sha256,'script_sha256':sha(__file__),'artifact_sha256':hashes,'uid':uid,'latent_shape':list(z.shape),'latent_sha256':latent_sha,
        'runtime':{'ORT':ort.__version__,'numpy':np.__version__,'python':platform.python_version(),'platform':platform.platform(),'ort_build':ort.get_build_info()},
        'cpu_environment':environment,'threads':4,'inter_threads':1,'spinning':False,'gpu_used':False,'models':{},'loaded_libraries_before':inventory(),
        'resource_before':{'maxrss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}}
    for name in ('fast_fp32','int8_large'):
        model=next(m for m in cfg['models'] if m['name']==name)
        require(model['kind']=='audio' and model['causal'],'Causal AudioVAE decoder required')
        refs=[c for c in campaign['checks'] if c.get('model')==name and c.get('uid')==uid and c.get('test')=='full_waveform']
        require(len(refs)==1,'Unique validated waveform hash required');expected=refs[0]['waveform_sha256']
        so=ort.SessionOptions();so.intra_op_num_threads=4;so.inter_op_num_threads=1;so.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        for key in ('session.intra_op.allow_spinning','session.inter_op.allow_spinning'):so.add_session_config_entry(key,'0')
        so.enable_profiling=True;so.profile_file_prefix=str(a.output_dir/name)
        libraries=model.get('custom_libraries',[]) or ([model['custom_library']] if model.get('custom_library') else [])
        for library in libraries:so.register_custom_ops_library(str(resolve(library)))
        session=ort.InferenceSession(str(resolve(model['path'])),sess_options=so,providers=['CPUExecutionProvider']);session.disable_fallback()
        require(session.get_providers()==['CPUExecutionProvider'],'Unexpected provider')
        require(len(session.get_inputs())==len(session.get_outputs())==1,'Single input/output required')
        calls=[]
        for i in range(3):
            y=session.run(None,{session.get_inputs()[0].name:z})[0]
            require(y.dtype==np.float32 and y.shape==(1,1,z.shape[-1]*cfg['kind_contracts']['audio']['hop']) and np.isfinite(y).all(),'Invalid waveform')
            require(wave_sha(z)==latent_sha and wave_sha(y)==expected,'Profile input/output differs from completed campaign bytes')
            calls.append({'index':i,'warmup':i<2,'waveform_sha256':expected,'bitwise_validated':True})
        profile=Path(session.end_profiling());events=json.loads(profile.read_text());summary=summarize(events)
        record['models'][name]={'graph_sha256':sha(resolve(model['path'])),'calls':calls,'profile_path':str(profile),'profile_sha256':sha(profile),'profile':summary}
        del session
    require(h.read_config(a.config,a.config_sha256)[1]==hashes,'Artifacts changed during profiling')
    require(sha(a.harness)==a.harness_sha256 and sha(a.campaign_results)==a.campaign_results_sha256 and sha(__file__)==record['script_sha256'],'Source or campaign changed')
    record.update(status='complete',loaded_libraries_after=inventory(),resource_after={'maxrss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},headline_rtf_updated=False)
    (a.output_dir/'results.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'status':record['status'],'uid':uid,'models':list(record['models']),'headline_rtf_updated':False}))
if __name__=='__main__':main()
