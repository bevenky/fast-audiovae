"""Profile a frozen CPU precision candidate without changing headline RTF.

Launch with CPU visibility variables already set. Verification and waveform
gates reuse the public compare_decoders.py harness. Exactly two warmups and
three subsequent calls run in a profiled session. Only kernel events contained
in the last three model_run intervals contribute to bottleneck aggregates.
Approximate models require their own complete campaign's unchanged FLOAT WAV
and exact uint32 waveform reproduction. This is not FP32 perceptual parity.
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


def completed_reference(path, config, config_sha256, verified, model, uid,
                        shape, sample_rate, harness_sha256, cpus, threads, h, np):
    """Read one full, untrimmed, hash-bound candidate export without inference."""
    path=path.resolve();campaign_sha256=h.sha(path)
    report=json.loads(path.read_text())
    if report.get('status')!='complete' or report.get('screen') is not False:
        raise ValueError('Approximate reference requires a complete full campaign, not a screen')
    if report.get('failures')!=[] or report.get('timed_outputs_match_validated_waveforms') is not True:
        raise ValueError('Campaign failed or lacks validated timing-output checks')
    if report.get('config_sha256')!=config_sha256.lower() or report.get('config')!=config:
        raise ValueError('Campaign configuration does not exactly match the frozen config')
    if report.get('verified_artifact_sha256')!=verified or report.get('harness_sha256')!=harness_sha256:
        raise ValueError('Campaign artifact or harness identity changed')
    if report.get('script_sha256')!=config.get('campaign_script_sha256'):
        raise ValueError('Campaign runner does not match the config pin')
    if report.get('gpu_used') is not False or report.get('onnxruntime')!='1.29.0':
        raise ValueError('Reference campaign must use CPU-only ORT1.29.0')
    if report.get('threads')!=threads or report.get('affinity')!=sorted(cpus):
        raise ValueError('Profile thread/affinity settings differ from the campaign')
    if report.get('protocol',{}).get('validation_uids')!=config['expected_uids']:
        raise ValueError('Reference campaign did not validate the unchanged full cohort')
    checks=report.get('checks',[])
    if not checks or any(v.get('passed') is not True for v in checks):
        raise ValueError('Reference campaign has missing or failing validation checks')
    validated=[v for v in checks if v.get('model')==model['name'] and v.get('uid')==uid
               and v.get('test')=='shape_finite_input_unchanged_and_declared_fp32_gate']
    if len(validated)!=1 or validated[0].get('fp32_parity_required') is not False:
        raise ValueError('Approximate candidate/UID lacks one validated waveform record')
    providers=[v for v in report.get('providers',[]) if v.get('model')==model['name']]
    if len(providers)!=1 or providers[0].get('providers')!=['CPUExecutionProvider']:
        raise ValueError('Candidate campaign provider is missing or not CPU-only')
    export_name=model.get('quality_name',model['name'])
    if sum(m.get('quality_name',m['name'])==export_name for m in config['models'])!=1:
        raise ValueError('Candidate export name is not unique in config')
    exports=[v for v in report.get('exports',[]) if v.get('model')==export_name and v.get('uid')==uid]
    if len(exports)!=1:raise ValueError('Expected one unique candidate/UID waveform export')
    export=exports[0];wav=Path(export['path'])
    if not wav.is_absolute():wav=path.parent/wav
    wav=wav.resolve()
    if h.sha(wav)!=export.get('sha256'):raise ValueError('Exported waveform hash changed')
    if export.get('rate')!=sample_rate or export.get('subtype')!='FLOAT':
        raise ValueError('Exported waveform rate or FLOAT subtype is wrong')
    if len(shape)!=3 or tuple(shape[:2])!=(1,1) or export.get('samples')!=shape[-1]:
        raise ValueError('Exported waveform sample count does not match full decoder shape')
    import soundfile as sf
    info=sf.info(wav)
    if info.samplerate!=sample_rate or info.channels!=1 or info.frames!=shape[-1] or info.subtype!='FLOAT' or info.format!='WAV':
        raise ValueError('Actual WAV header does not match untrimmed mono FLOAT contract')
    audio,rate=sf.read(wav,dtype='float32',always_2d=True)
    if rate!=sample_rate or audio.dtype!=np.float32 or audio.shape!=(shape[-1],1) or not np.isfinite(audio).all():
        raise ValueError('Invalid exported waveform samples')
    ref=np.ascontiguousarray(audio[:,0].reshape(shape))
    if h.sha(path)!=campaign_sha256 or h.sha(wav)!=export['sha256']:
        raise ValueError('Reference inputs changed while reading')
    provenance={'kind':'candidate_own_validated_raw_float_wav','candidate':model['name'],
        'export_model':export_name,'uid':uid,'campaign_path':str(path),'campaign_sha256':campaign_sha256,
        'campaign_config_sha256':report['config_sha256'],'wav_path':str(wav),'wav_sha256':export['sha256'],
        'sample_rate':sample_rate,'shape':list(ref.shape),'dtype':'float32','cropped':False,
        'raw_waveform_sha256':hashlib.sha256(ref.tobytes()).hexdigest(),
        'gate':'Exact uint32 bit equality to this candidate own validated full waveform',
        'quality_scope':'Deterministic replay only; this does not assert FP32 parity or perceptual quality'}
    return ref,provenance,{path:campaign_sha256,wav:export['sha256']}


def exact_waveform_gate(actual, reference, np):
    shape_ok=actual.shape==reference.shape
    dtype_ok=actual.dtype==np.float32 and reference.dtype==np.float32
    finite=bool(np.isfinite(actual).all())
    bitwise=bool(shape_ok and dtype_ok and np.array_equal(actual.view(np.uint32),reference.view(np.uint32)))
    return {'passed':bool(shape_ok and dtype_ok and finite and bitwise),'shape_equal':shape_ok,
            'dtype_float32':dtype_ok,'finite':finite,'uint32_bitwise_equal':bitwise,
            'reference':'candidate_own_validated_raw_float_wav','fp32_parity_asserted':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--harness',required=True,type=Path,help='Unmodified public benchmarks/compare_decoders.py')
    p.add_argument('--config',required=True,type=Path);p.add_argument('--config-sha256',required=True)
    p.add_argument('--model',required=True);p.add_argument('--threads',required=True,type=int)
    p.add_argument('--affinity',required=True,help='Explicit comma-separated allowed logical CPUs')
    p.add_argument('--uid',required=True,help='Explicit source UID present in the frozen case archive')
    p.add_argument('--latent-frames',type=int,help='Explicit causal AudioVAE prefix length; omitted keeps the entire real case')
    p.add_argument('--completed-campaign',type=Path,help='Required for approximate models: completed full precision campaign results.json')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--hardware',type=Path,help='Optional existing diagnose_cpu.py report for cgroup paths')
    p.add_argument('--probe',type=Path,required=True,help='Exact preloaded optional CPU attribution shim')
    p.add_argument('--probe-sha256',required=True)
    a=p.parse_args();h=load_harness(a.harness.resolve());cpu_env=h.require_cpu_environment()
    if a.threads!=2:raise ValueError('This Intel profile requires exactly two CPU threads')
    cpus=[int(v) for v in a.affinity.split(',')]
    if len(cpus)!=len(set(cpus)) or len(cpus)!=2 or any(v<0 for v in cpus):raise ValueError('Require an explicit two-CPU affinity mask')
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
    approximate=model.get('approximate',False)
    if type(approximate) is not bool:raise ValueError('Model approximate flag must be boolean')
    if not approximate:raise ValueError('Internal precision probe is for an INT8 candidate only')
    if approximate and a.completed_campaign is None:raise ValueError('Approximate model requires --completed-campaign')
    if approximate and a.latent_frames is not None:raise ValueError('Approximate profiles must use the complete latent and waveform; no cropping')
    if not approximate and a.completed_campaign is not None:raise ValueError('FP32 controls use the original reference; omit --completed-campaign')
    cases,meta=h.read_cases(resolve(config[kind+'_cases']),contract)
    if set(cases)!=set(config['expected_uids']) or len(cases)!=config['expected_case_count']:raise ValueError('Frozen corpus membership changed')
    if a.uid not in cases:raise ValueError('Requested case UID missing')
    case=cases[a.uid];original_frames=int(case['z'].shape[-1])
    frames=profile_frame_count(original_frames,a.latent_frames,kind,model['causal'])
    z=np.ascontiguousarray(case['z'][...,:frames]);ref=np.ascontiguousarray(case['ref'][...,:frames*contract['hop']])
    reference_artifacts={}
    waveform_reference={'kind':'original_saved_fp32_waveform','gate':'Unchanged public harness FP32 finite/shape, atol1e-5 rtol1e-4',
                        'fp32_parity_asserted':True,'cropped':frames<original_frames}
    if approximate:
        ref,waveform_reference,reference_artifacts=completed_reference(a.completed_campaign,config,
            a.config_sha256,verified,model,a.uid,case['ref'].shape,contract['sample_rate'],
            h.sha(a.harness),cpus,a.threads,h,np)
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
    artifacts={a.config.resolve(),a.harness.resolve(),*(resolve(v) for v in verified),*reference_artifacts}
    pinned_artifacts={resolve(v):digest for v,digest in verified.items()}
    pinned_artifacts.update(reference_artifacts)
    pinned_artifacts[a.config.resolve()]=a.config_sha256.lower()
    pinned_artifacts[a.harness.resolve()]=h.sha(a.harness)
    pinned_artifacts[Path(__file__).resolve()]=h.sha(__file__)
    if h.sha(a.probe)!=a.probe_sha256:raise ValueError('Probe artifact changed')
    pinned_artifacts[a.probe.resolve()]=a.probe_sha256
    helper=Path(__file__).with_name('native_probe.py')
    pinned_artifacts[helper]=h.sha(helper)
    if any(h.sha(v)!=digest for v,digest in pinned_artifacts.items()):
        raise ValueError('A verified input changed before profiling setup')
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
            'waveform_reference':waveform_reference,
            'case_selection':{'explicit_requested_latent_frames':a.latent_frames,'original_latent_frames':original_frames,
                              'profile_latent_frames':frames,'cropped':frames<original_frames,
                              'original_latent_shape':list(case['z'].shape),'original_reference_shape':list(case['ref'].shape),
                              'original_latent_sha256':hashlib.sha256(case['z'].tobytes()).hexdigest(),
                              'original_reference_sha256':hashlib.sha256(case['ref'].tobytes()).hexdigest(),
                              'reference_selection':('Complete candidate-own validated raw FLOAT WAV' if approximate else
                                  'First profile_latent_frames*hop samples of the original saved causal waveform' if frames<original_frames else 'Complete original saved waveform'),
                              'scope':'Explicit isolated profiling prefix only; frozen corpus files and multilingual timing selection are unchanged'},
            'latent_sha256':hashlib.sha256(z.tobytes()).hexdigest(),'reference_sha256':hashlib.sha256(ref.tobytes()).hexdigest(),
            'sample_rate':contract['sample_rate'],'generated_seconds':ref.shape[-1]/contract['sample_rate'],
            'custom_libraries':[{'path':str(v),'sha256':h.sha(v)} for v in libraries],
            'runtime':{'onnxruntime':ort.__version__,'ort_build':ort.get_build_info(),'numpy':np.__version__,
                       'python':sys.version,'platform':platform.platform(),'machine':platform.machine()},
            'cpu_only_environment':cpu_env,'gpu_used':False,'affinity_before':sorted(original),'affinity':sorted(cpus),
            'protocol':{'threads':a.threads,'inter_threads':1,'execution':'sequential','graph_optimization':'all',
                        'spinning':False,'warmups':2,'profile_calls':3,'profile_interval_selection':'last3 of exactly5 model_run intervals',
                        'waveform_gate':waveform_reference['gate'],
                        'headline_rtf_updated':False,'provider':'CPUExecutionProvider'},
            'resource_before':usage_snapshot(),'host_before':h.host_snapshot(),'cgroup_before':h.cpu_stat_snapshot(stat_paths),
            'gpu_library_mappings_before':h.gpu_library_mappings(),'calls':[]}
    def save():temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(output)
    save();session=None;profile_path=None
    try:
        mappings=record['gpu_library_mappings_before']
        if not mappings.get('available') or mappings.get('basenames'):raise RuntimeError('Cannot verify absence of GPU libraries')
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
        from native_probe import NativeProbe
        probe=NativeProbe(a.probe,a.probe_sha256,pinned_artifacts,h.sha)
        record['native_probe_provenance']=probe.provenance
        for call in range(5):
            if call>=2:probe.start()
            started=time.perf_counter_ns()
            try:y=session.run(None,feeds)[0]
            finally:
                elapsed=time.perf_counter_ns()-started
                counters=probe.stop() if call>=2 else None
            gate=exact_waveform_gate(y,ref,np) if approximate else h.compare(y,ref)
            repeated=previous is None or bool(np.array_equal(y.view(np.uint32),previous.view(np.uint32)))
            unchanged=bool(np.array_equal(z.view(np.uint32),original_z.view(np.uint32)))
            record['calls'].append({'call':call,'phase':'warmup' if call<2 else 'profile','profiled_wrapper_wall_ns':elapsed,
                                    'waveform_gate':gate,'repeat_bitwise':repeated,'latent_unchanged':unchanged,'native_probe':counters})
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
        record['artifact_hashes_after']={str(v):h.sha(v) for v in pinned_artifacts}
        record['artifact_hashes_unchanged']=all(record['artifact_hashes_after'][str(v)]==digest for v,digest in pinned_artifacts.items())
        if not record['artifact_hashes_unchanged']:raise RuntimeError('A pinned input changed during profiling')
        mappings=h.gpu_library_mappings()
        if not mappings.get('available') or mappings.get('basenames'):raise RuntimeError('GPU library mapping check failed')
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
