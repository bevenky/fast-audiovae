"""Profile one frozen CPU decoder candidate without changing headline RTF.

Launch with CPU visibility variables already set. Verification and waveform
gates reuse the public compare_decoders.py harness. Exactly two warmups and
three subsequent calls run in a profiled session. Only kernel events contained
in the last three model_run intervals contribute to bottleneck aggregates.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
import time


def _number(value):
    return type(value) in (int,float) and math.isfinite(value)


def profile_frame_count(original,requested,kind,causal):
    """Cropping requires an explicit request and the causal AudioVAE contract."""
    if type(original) is not int or original<=0:raise ValueError('Invalid original frame count')
    if requested is None:return original
    if type(requested) is not int or requested<=0 or requested>original:raise ValueError('Requested frame count must be positive and no larger than the real case')
    if requested<original and (kind!='audio' or causal is not True):raise ValueError('Prefix cropping requires causal AudioVAE and its original reference')
    return requested


def summarize_profile(events, expected_runs=5, selected_calls=3):
    """Pure parsing/aggregation, usable in offline tests without ORT import."""
    if not isinstance(events,list):raise ValueError('Expected an ORT event list')
    runs=sorted([e for e in events if e.get('name')=='model_run' and e.get('ph')=='X'],key=lambda e:e.get('ts',-1))
    if len(runs)!=expected_runs:raise ValueError(f'Expected exactly {expected_runs} model_run intervals, got {len(runs)}')
    for e in runs:
        if not _number(e.get('ts')) or not _number(e.get('dur')) or e['dur']<=0:raise ValueError('Invalid model_run timestamp/duration')
    if any(a['ts']+a['dur']>b['ts']+1e-6 for a,b in zip(runs,runs[1:])):raise ValueError('Overlapping model_run intervals')
    picked=runs[-selected_calls:];counts=[0]*selected_calls;rows=[]
    for e in events:
        if e.get('cat')!='Node' or not e.get('name','').endswith('_kernel_time'):continue
        args=e.get('args',{})
        if args.get('provider')!='CPUExecutionProvider':raise ValueError('Missing or non-CPU kernel provider')
        if e.get('ph')!='X' or not _number(e.get('ts')) or not _number(e.get('dur')) or e['dur']<0:raise ValueError('Malformed kernel interval')
        owners=[i for i,r in enumerate(picked) if e['ts']>=r['ts']-1e-6 and e['ts']+e['dur']<=r['ts']+r['dur']+1e-6]
        touches=any(e['ts']<r['ts']+r['dur'] and e['ts']+e['dur']>r['ts'] for r in picked)
        if len(owners)>1 or (touches and not owners):raise ValueError('Kernel crosses a selected model_run boundary')
        if not owners:continue
        run=owners[0];counts[run]+=1
        if not isinstance(args.get('op_name'),str) or not args['op_name']:raise ValueError('Kernel lacks operator name')
        rows.append({'run':run,'name':e['name'],'op_name':args['op_name'],'duration_us':e['dur'],
                     'timestamp_us':e['ts'],'pid':e.get('pid'),'tid':e.get('tid'),'args':args})
    if not all(counts):raise ValueError('Every selected run must contain CPU kernel events')
    operators={};nodes={}
    for row in rows:
        args=row['args'];op=row['op_name'];run=row['run'];duration=row['duration_us']
        oi=operators.setdefault(op,{'op_name':op,'calls':0,'run_duration_us':[0.]*selected_calls,'node_names':set()})
        oi['calls']+=1;oi['run_duration_us'][run]+=duration;oi['node_names'].add(row['name'])
        # Retain shape variants explicitly instead of silently merging them.
        signature=(row['name'],op,json.dumps(args.get('input_type_shape'),sort_keys=True),json.dumps(args.get('output_type_shape'),sort_keys=True))
        ni=nodes.setdefault(signature,{'name':row['name'],'op_name':op,'calls':0,'run_duration_us':[0.]*selected_calls,
            'input_type_shape':args.get('input_type_shape'),'output_type_shape':args.get('output_type_shape'),
            'activation_size':args.get('activation_size'),'parameter_size':args.get('parameter_size'),
            'output_size':args.get('output_size'),'provider':args['provider'],'node_index':args.get('node_index')})
        ni['calls']+=1;ni['run_duration_us'][run]+=duration
    total=sum(row['duration_us'] for row in rows)
    def finish(values):
        output=[]
        for item in values:
            item['sum_duration_us']=sum(item['run_duration_us'])
            item['mean_us_per_model_call']=item['sum_duration_us']/selected_calls
            item['median_us_per_model_call']=statistics.median(item['run_duration_us'])
            item['percent_of_selected_kernel_time']=100*item['sum_duration_us']/total if total else 0.
            if 'node_names' in item:item['node_names']=sorted(item['node_names'])
            output.append(item)
        return sorted(output,key=lambda v:v['sum_duration_us'],reverse=True)
    return {'model_run_count':len(runs),'excluded_warmup_intervals':runs[:-selected_calls],
            'selected_intervals':picked,'selected_kernel_counts':counts,'selected_kernel_events':rows,
            'selected_kernel_duration_sum_us':total,'operators':finish(operators.values()),'nodes':finish(nodes.values()),
            'time_units':'ORT timestamps and durations are microseconds',
            'percentage_denominator':'Sum of selected kernel-event durations, not wall time; overlapping events can double count',
            'profiling_overhead_caveat':'Profiling adds tracing/scheduling overhead. These values locate bottlenecks and must not replace unprofiled headline RTF.'}


def usage_snapshot():
    r=resource.getrusage(resource.RUSAGE_SELF)
    fields=('ru_utime','ru_stime','ru_maxrss','ru_minflt','ru_majflt','ru_inblock','ru_oublock','ru_nvcsw','ru_nivcsw')
    out={k:getattr(r,k) for k in fields}
    out['ru_maxrss_raw_units']='bytes' if sys.platform=='darwin' else 'KiB'
    out['process_max_rss_bytes']=int(r.ru_maxrss*(1 if sys.platform=='darwin' else 1024))
    out['scope']='Cumulative whole-process resource usage; maximum RSS includes setup, model, outputs and profiling buffers'
    return out


def load_harness(path):
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location('frozen_decoder_comparison',path)
    if spec is None or spec.loader is None:raise ValueError('Cannot load comparison harness')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--harness',required=True,type=Path,help='Unmodified public benchmarks/compare_decoders.py')
    p.add_argument('--config',required=True,type=Path);p.add_argument('--config-sha256',required=True)
    p.add_argument('--model',required=True);p.add_argument('--threads',required=True,type=int)
    p.add_argument('--affinity',required=True,help='Explicit comma-separated allowed logical CPUs')
    p.add_argument('--uid',required=True,help='Explicit source UID present in the frozen case archive')
    p.add_argument('--latent-frames',type=int,help='Explicit causal AudioVAE prefix length; omitted keeps the entire real case')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--hardware',type=Path,help='Optional existing diagnose_cpu.py report for cgroup paths')
    a=p.parse_args();h=load_harness(a.harness.resolve());cpu_env=h.require_cpu_environment()
    if a.threads<=0:raise ValueError('Positive explicit thread count required')
    cpus=[int(v) for v in a.affinity.split(',')]
    if len(cpus)!=len(set(cpus)) or len(cpus)<a.threads or any(v<0 for v in cpus):raise ValueError('Invalid explicit affinity mask')
    original=h.current_affinity()
    if original is None or not set(cpus).issubset(original):raise ValueError('Affinity unavailable or outside allowed CPUs')
    os.sched_setaffinity(0,set(cpus))
    if h.current_affinity()!=set(cpus):raise RuntimeError('Affinity was not applied exactly')
    import numpy as np
    import onnxruntime as ort
    h.np=np;h.ort=ort
    if ort.__version__!='1.29.0':raise RuntimeError('Expected ORT1.29.0, got '+ort.__version__)
    config,verified=h.read_config(a.config.resolve(),a.config_sha256)
    base=a.config.resolve().parent;resolve=lambda v:h.relative_path(base,v)
    matches=[m for m in config['models'] if m['name']==a.model]
    if len(matches)!=1:raise ValueError('Model name not uniquely present in frozen config')
    model=matches[0];kind=model['kind'];contract=config['kind_contracts'][kind]
    cases,meta=h.read_cases(resolve(config[kind+'_cases']),contract)
    if set(cases)!=set(config['expected_uids']) or len(cases)!=config['expected_case_count']:raise ValueError('Frozen corpus membership changed')
    if a.uid not in cases:raise ValueError('Requested case UID missing')
    case=cases[a.uid];original_frames=int(case['z'].shape[-1])
    frames=profile_frame_count(original_frames,a.latent_frames,kind,model['causal'])
    z=np.ascontiguousarray(case['z'][...,:frames]);ref=np.ascontiguousarray(case['ref'][...,:frames*contract['hop']])
    extra={}
    if model.get('extra_inputs'):
        with np.load(resolve(config[kind+'_cases']),allow_pickle=False) as arrays:
            for name,key in model['extra_inputs'].items():
                v=np.ascontiguousarray(arrays[key])
                if v.dtype!=np.float32 or not np.isfinite(v).all():raise ValueError('Invalid fixed FP32 auxiliary input')
                extra[name]=v
    libs=model.get('custom_libraries')
    if libs is None:
        single=model.get('custom_library') or config.get('custom_library');libs=[single] if single else []
    libraries=[resolve(v) for v in libs];output=a.output.resolve();temp=output.with_suffix('.tmp')
    artifacts={a.config.resolve(),a.harness.resolve(),*(resolve(v) for v in verified)}
    if output in artifacts or temp in artifacts or output.exists() or temp.exists():raise ValueError('Output must be a new file separate from inputs')
    output.parent.mkdir(parents=True,exist_ok=True)
    prefix=output.parent/(output.stem+'.ort-profile')
    if list(output.parent.glob(prefix.name+'*')):raise ValueError('Profiling prefix already exists')
    stat_paths=h.cpu_stat_paths(a.hardware)
    record={'status':'running','started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'scope':'Representative decoder bottleneck profile only; not headline RTF or full-corpus quality acceptance',
            'model':model,'config_sha256':a.config_sha256.lower(),'verified_artifact_sha256':verified,
            'harness_path':str(a.harness.resolve()),'harness_sha256':h.sha(a.harness),'script_sha256':h.sha(__file__),
            'model_sha256':h.sha(resolve(model['path'])),'case_archive_sha256':h.sha(resolve(config[kind+'_cases'])),
            'case_metadata_sha256':h.sha(resolve(config[kind+'_cases']).with_suffix('.json')),
            'uid':a.uid,'source_uid':a.uid,'latent_shape':list(z.shape),'reference_shape':list(ref.shape),
            'case_selection':{'explicit_requested_latent_frames':a.latent_frames,'original_latent_frames':original_frames,
                              'profile_latent_frames':frames,'cropped':frames<original_frames,
                              'original_latent_shape':list(case['z'].shape),'original_reference_shape':list(case['ref'].shape),
                              'original_latent_sha256':hashlib.sha256(case['z'].tobytes()).hexdigest(),
                              'original_reference_sha256':hashlib.sha256(case['ref'].tobytes()).hexdigest(),
                              'reference_selection':'First profile_latent_frames*hop samples of the original saved causal waveform' if frames<original_frames else 'Complete original saved waveform',
                              'scope':'Explicit isolated profiling prefix only; frozen corpus files and multilingual timing selection are unchanged'},
            'latent_sha256':hashlib.sha256(z.tobytes()).hexdigest(),'reference_sha256':hashlib.sha256(ref.tobytes()).hexdigest(),
            'sample_rate':contract['sample_rate'],'generated_seconds':ref.shape[-1]/contract['sample_rate'],
            'custom_libraries':[{'path':str(v),'sha256':h.sha(v)} for v in libraries],
            'runtime':{'onnxruntime':ort.__version__,'ort_build':ort.get_build_info(),'numpy':np.__version__,
                       'python':sys.version,'platform':platform.platform(),'machine':platform.machine()},
            'cpu_only_environment':cpu_env,'gpu_used':False,'affinity_before':sorted(original),'affinity':sorted(cpus),
            'protocol':{'threads':a.threads,'inter_threads':1,'execution':'sequential','graph_optimization':'all',
                        'spinning':False,'warmups':2,'profile_calls':3,'profile_interval_selection':'last3 of exactly5 model_run intervals',
                        'waveform_gate':'Unchanged public harness compare: FP32 finite, exact shape, atol1e-5 rtol1e-4 against saved original waveform',
                        'headline_rtf_updated':False,'provider':'CPUExecutionProvider'},
            'resource_before':usage_snapshot(),'host_before':h.host_snapshot(),'cgroup_before':h.cpu_stat_snapshot(stat_paths),
            'gpu_library_mappings_before':h.gpu_library_mappings(),'calls':[]}
    def save():temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(output)
    save();session=None;profile_path=None
    try:
        options=ort.SessionOptions();options.intra_op_num_threads=a.threads;options.inter_op_num_threads=1
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_session_config_entry('session.intra_op.allow_spinning','0');options.add_session_config_entry('session.inter_op.allow_spinning','0')
        for library in libraries:options.register_custom_ops_library(str(library))
        options.enable_profiling=True;options.profile_file_prefix=str(prefix)
        session=ort.InferenceSession(str(resolve(model['path'])),options,providers=['CPUExecutionProvider']);session.disable_fallback()
        if session.get_providers()!=['CPUExecutionProvider']:raise RuntimeError('Refusing non-CPU provider')
        record['session_providers']=session.get_providers()
        inputs=session.get_inputs();primary=[v for v in inputs if v.name not in extra];outputs=session.get_outputs()
        if len(primary)!=1 or any(v.type!='tensor(float)' for v in inputs) or set(v.name for v in inputs)!=set(extra)|{primary[0].name}:raise RuntimeError('Unexpected FP32 input contract')
        if len(outputs)!=1 or outputs[0].type!='tensor(float)':raise RuntimeError('Expected one FP32 waveform output')
        feeds={primary[0].name:z,**extra};original_z=z.copy();previous=None
        for call in range(5):
            started=time.perf_counter_ns();y=session.run(None,feeds)[0];elapsed=time.perf_counter_ns()-started
            gate=h.compare(y,ref)
            repeated=previous is None or bool(np.array_equal(y.view(np.uint32),previous.view(np.uint32)))
            unchanged=bool(np.array_equal(z.view(np.uint32),original_z.view(np.uint32)))
            record['calls'].append({'call':call,'phase':'warmup' if call<2 else 'profile','profiled_wrapper_wall_ns':elapsed,
                                    'waveform_gate':gate,'repeat_bitwise':repeated,'latent_unchanged':unchanged})
            if not gate['passed'] or not repeated or not unchanged:raise RuntimeError('Waveform, repeat or input-integrity gate failed')
            previous=y.copy();save()
        profile_path=Path(session.end_profiling());session=None
        events=json.loads(profile_path.read_text());provider=h.cpu_profile_report(events)
        if not provider['passed']:raise RuntimeError('CPU profile provider gate failed')
        record['profile_provider_check']=provider;record['profile']=summarize_profile(events)
        record['raw_profile']={'path':str(profile_path.resolve()),'sha256':h.sha(profile_path),'event_count':len(events)}
        record['runtime']['loaded_ort_binary_hashes']={str(Path(m.__file__).resolve()):h.sha(Path(m.__file__))
            for name,m in list(sys.modules.items()) if name.startswith('onnxruntime.') and getattr(m,'__file__',None)
            and Path(m.__file__).suffix in ('.so','.dylib','.pyd')}
        record['status']='complete'
    except Exception as exc:
        record['status']='failed';record['error']=repr(exc)
        if session is not None:
            try:
                profile_path=Path(session.end_profiling());record['failed_raw_profile']={'path':str(profile_path.resolve()),'sha256':h.sha(profile_path)}
            except Exception as flush_error:record['profile_flush_error']=repr(flush_error)
        raise
    finally:
        record['resource_after']=usage_snapshot();record['host_after']=h.host_snapshot();record['cgroup_after']=h.cpu_stat_snapshot(stat_paths)
        record['host_delta']=h.host_delta(record['host_before'],record['host_after'])
        record['cgroup_delta']=h.cpu_stat_delta(record['cgroup_before'],record['cgroup_after'])
        record['gpu_library_mappings_after']=h.gpu_library_mappings();record['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());save()
    print(json.dumps({'output':str(output),'status':record['status'],'selected_kernel_counts':record['profile']['selected_kernel_counts'],
                      'top_operators':record['profile']['operators'][:8]},indent=2))


if __name__=='__main__':main()
