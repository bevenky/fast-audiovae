"""One bounded AMD CPU attribution pass: two warmups and three profiled decodes."""
import argparse,ctypes,hashlib,importlib.util,json,os,platform,time
from pathlib import Path
from profile_parser import summarize_profile
FIELDS=('abi','fields','enabled','active','prepare_calls','prepare_ns','prepare_failures','row_calls','row_ns','row_failures','gemm_calls','gemm_ns','nested_gemm_calls','nested_gemm_ns','outside_row_gemm_calls','outside_row_gemm_ns','clock_errors','resolution_errors','control_errors','resolved')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','harness','completed-screen','probe','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('config-sha256','harness-sha256','screen-sha256','probe-sha256','uid'):p.add_argument('--'+name,required=True)
    p.add_argument('--backend',choices=('mkl','aocl'),required=True);a=p.parse_args()
    initial_self=sha(Path(__file__));require(sha(a.harness)==a.harness_sha256 and sha(a.completed_screen)==a.screen_sha256 and sha(a.probe)==a.probe_sha256,'Input hash mismatch')
    require(platform.system()=='Linux' and 'AuthenticAMD' in Path('/proc/cpuinfo').read_text(),'AMD Linux required')
    for key in ('MKL_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):require(os.environ.get(key)=='1',key+' must be one')
    cpus={0,1,2,3};require(cpus<=os.sched_getaffinity(0),'Four original AMD physical CPUs unavailable');os.sched_setaffinity(0,cpus)
    spec=importlib.util.spec_from_file_location('public_harness',a.harness);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h);environment=h.require_cpu_environment()
    import numpy as np
    import onnxruntime as ort
    require(ort.__version__=='1.29.0','ORT1.29 required');h.np=np;h.ort=ort
    cfg,pins=h.read_config(a.config,a.config_sha256);resolve=lambda x:(a.config.resolve().parent/x).resolve()
    report=json.loads(a.completed_screen.read_text())
    require(report['status']=='complete' and report['mode']=='screen' and report['platform']=='amd' and report['threads']==4 and report['affinity']==[0,1,2,3],'Completed matching AMD screen required')
    require(report['config_sha256']==a.config_sha256 and report['harness_sha256']==a.harness_sha256,'Screen provenance differs')
    require(not report['failures'] and len(report['checks'])==84 and all(x['passed'] for x in report['checks']) and len(report['measurements'])==60,'Screen coverage incomplete')
    model=next(m for m in cfg['models'] if m['name']=='int8_large');require(model.get('approximate') is True,'Explicit approximate candidate required')
    references=[x for x in report['checks'] if x.get('model')=='int8_large' and x.get('uid')==a.uid and x['test']=='full_waveform'];require(len(references)==1,'Unique candidate-own waveform hash required');reference=references[0]
    cases,_=h.read_cases(resolve(cfg['audio_cases']),cfg['kind_contracts']['audio']);z=np.ascontiguousarray(cases[a.uid]['z']);zhash=hashlib.sha256(z.tobytes()).hexdigest();require(zhash==reference['latent_sha256'],'Frozen latent differs')
    require(not a.output_dir.exists(),'Fresh output directory required');a.output_dir.mkdir(parents=True)
    artifacts={resolve(k):v for k,v in pins.items()}
    core=Path(os.environ['IP_PROBE_CORE_LIBRARY']).resolve();gemm=Path(os.environ['IP_PROBE_GEMM_LIBRARY']).resolve();probe=a.probe.resolve()
    require(Path(os.environ['LD_PRELOAD']).resolve()==probe and core in artifacts and gemm in artifacts,'Probe resolver paths not pinned in config')
    require(sha(core)==artifacts[core] and sha(gemm)==artifacts[gemm],'Resolved dependency hash differs')
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1;options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry('session.intra_op.allow_spinning','0');options.add_session_config_entry('session.inter_op.allow_spinning','0');options.enable_profiling=True;options.profile_file_prefix=str(a.output_dir/'ort')
    for path in model['custom_libraries']:options.register_custom_ops_library(str(resolve(path)))
    session=ort.InferenceSession(str(resolve(model['path'])),options,providers=['CPUExecutionProvider']);session.disable_fallback();require(session.get_providers()==['CPUExecutionProvider'],'CPU only required')
    native=ctypes.CDLL(str(probe),mode=os.RTLD_NOW|os.RTLD_NOLOAD)
    for name in ('ip_probe_initialize','ip_probe_reset'):
        fn=getattr(native,name);fn.argtypes=[];fn.restype=ctypes.c_int
    native.ip_probe_set_enabled.argtypes=[ctypes.c_int];native.ip_probe_set_enabled.restype=ctypes.c_int
    native.ip_probe_snapshot.argtypes=[ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t];native.ip_probe_snapshot.restype=ctypes.c_int
    require(native.ip_probe_initialize()==0,'Probe resolution failed')
    def run():
        y=session.run(None,{session.get_inputs()[0].name:z})[0]
        require(y.dtype==np.float32 and y.shape==(1,1,z.shape[-1]*1920) and np.isfinite(y).all(),'Invalid waveform')
        require(hashlib.sha256(y.tobytes()).hexdigest()==reference['waveform_sha256'],'Profiled waveform differs from completed screen')
        require(hashlib.sha256(z.tobytes()).hexdigest()==zhash,'Latent mutated')
    for _ in range(2):run()
    require(native.ip_probe_reset()==0 and native.ip_probe_set_enabled(1)==0,'Probe start failed')
    for _ in range(3):run()
    require(native.ip_probe_set_enabled(0)==0,'Probe stop failed');raw=(ctypes.c_uint64*20)();require(native.ip_probe_snapshot(raw,20)==0,'Snapshot failed');counts=dict(zip(FIELDS,map(int,raw)))
    require(counts['abi']==1 and counts['fields']==20 and counts['resolved']==1,'Bad probe ABI/state')
    require(all(counts[k]==0 for k in ('enabled','active','prepare_failures','row_failures','clock_errors','resolution_errors','control_errors','outside_row_gemm_calls','outside_row_gemm_ns')),'Probe attribution failed')
    require(counts['prepare_calls']>0 and counts['row_calls']>0 and counts['gemm_calls']>0 and counts['nested_gemm_calls']==counts['gemm_calls'] and counts['nested_gemm_ns']==counts['gemm_ns']<=counts['row_ns'],'Missing or inconsistent native attribution')
    counts['row_remainder_ns']=counts['row_ns']-counts['gemm_ns']
    trace=Path(session.end_profiling());parsed=summarize_profile(json.loads(trace.read_text()))
    inventory=h.gpu_library_mappings();require(inventory['available'] and not inventory['basenames'],'GPU runtime mapping found')
    mappings=Path('/proc/self/maps').read_text().splitlines();loaded=sorted({x.split()[-1] for x in mappings if any(v in x for v in ('mkl','aocl','gomp','precision'))})
    require(h.read_config(a.config,a.config_sha256)[1]==pins and sha(a.completed_screen)==a.screen_sha256 and sha(a.harness)==a.harness_sha256 and sha(a.probe)==a.probe_sha256 and sha(Path(__file__))==initial_self,'Artifact changed during attribution')
    result={'status':'complete','scope':'One AMD CPU attribution pass, excluded from all headline speed decisions','backend':a.backend,'gpu_used':False,'providers':session.get_providers(),'environment':environment,'threads':{'ORT':4,'native_shards_segments':cfg.get('aocl_derivation',{}).get('outer_shards_segments',2),'integer_inner':1,'affinity':[0,1,2,3]},'uid':a.uid,'latent_shape':list(z.shape),'waveform_reference':'Exact bytes of this candidate from its completed 3-clip screen, verified by SHA256; no FP32 equivalence claim','waveform_sha256':reference['waveform_sha256'],'native':counts,'ORT_profile':parsed,'gpu_library_mappings':inventory,'loaded_cpu_libraries':loaded,'provenance':{'config_sha256':a.config_sha256,'screen_sha256':a.screen_sha256,'harness_sha256':a.harness_sha256,'probe_sha256':a.probe_sha256,'core_sha256':artifacts[core],'gemm_library_sha256':artifacts[gemm],'script_sha256':initial_self,'profile_parser_sha256':sha(Path(__file__).with_name('profile_parser.py'))},'limits':{'preparation':'Allocation and per-column quantization, combined','GEMM':'Library integer GEMM including internal packing; public API does not expose a separate packing timer','row_remainder':'Dequantization, allocation, bounds checks and attribution overhead, combined','denominator':'Summed worker-call durations overlap and are not fractions of decoder wall time','headline_speed_updated':False}}
    (a.output_dir/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'complete','native':counts},indent=2))
if __name__=='__main__':main()
