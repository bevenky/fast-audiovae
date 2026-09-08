"""Compare prepared decoder latents with ONNX Runtime 1.29 on CPU.

This public refactor preserves the timing and gate protocol of the measured
parent harness SHA256
cbd847dd70398c811cec1b9a6e412b3486fb14f7df678879817742f891ec3922.
It is not the exact source file used for the published measurements.

This script consumes user-supplied models and prepared FP32 case archives.
It does not download, encode audio, or export AudioVAE2, Mimi or Meta models.
Each NPZ contains UID__z [1,C,L] and UID__ref [1,1,hop*L], with full padded
waveforms from the original model. Its companion JSON supplies timed_ids in
frozen order and may include source/checkpoint/corpus provenance. All model
families must cover the same UIDs. Fixed auxiliary arrays, such as a watermark
message, may share the NPZ. References must not be trimmed to source duration.

Copy example-comparison.json, supply relative paths and SHA256 hashes, then
freeze the config. Pass its SHA256 as --config-sha256. artifact_sha256 must
cover each model, external weight file, custom library, NPZ and companion JSON.
Only trusted custom libraries should be supplied: ORT loads executable code.
Launch with CUDA_VISIBLE_DEVICES=-1 NVIDIA_VISIBLE_DEVICES=void
ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 before importing ORT.
Defaults are full validation, one warmup per timed shape and three repetitions.
RTF uses complete generated duration; encoding, loading, warmup and profiling
are outside the timed fresh decoder call. This is not cached streaming RTF.
"""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import platform
import random
import re
import time

np = None
ort = None
CPU_ENV = {'CUDA_VISIBLE_DEVICES':'-1','NVIDIA_VISIBLE_DEVICES':'void',
           'ROCR_VISIBLE_DEVICES':'-1','HIP_VISIBLE_DEVICES':'-1'}
SOURCE_CAMPAIGN_SHA256 = 'cbd847dd70398c811cec1b9a6e412b3486fb14f7df678879817742f891ec3922'


def require_cpu_environment():
    actual={k:os.environ.get(k) for k in CPU_ENV}
    wrong={k:{'required':v,'actual':actual[k]} for k,v in CPU_ENV.items() if actual[k]!=v}
    if wrong:
        raise RuntimeError('Set CPU-only visibility variables at process launch, before imports: '+json.dumps(wrong))
    return actual


def gpu_library_mappings():
    """Mapping inventory only; loading a library does not establish device use."""
    try:
        lines=Path('/proc/self/maps').read_text().splitlines()
    except OSError as exc:
        return {'available':False,'basenames':None,'error':repr(exc)}
    pattern=re.compile(r'^(?:lib(?:cuda|cudart|nvrtc|nvjitlink|cublas|cudnn|cusparse|cusolver|curand|cufft|nvidia|nvoptix|nvinfer|nvonnxparser|amdhip|hip|hsa-runtime|rocblas|rocfft|rocrand|rocsolver|rocsparse|miopen|opencl|vulkan|ze_loader|torch_cuda|torch_hip)|libonnxruntime_providers_(?:cuda|tensorrt|rocm|migraphx))',re.I)
    names=set()
    for line in lines:
        parts=line.split(maxsplit=5)
        if len(parts)<6 or not parts[5].startswith('/'):
            continue
        name=Path(parts[5].removesuffix(' (deleted)')).name
        if pattern.match(name):names.add(name)
    return {'available':True,'basenames':sorted(names),
            'scope':'Read /proc/self/maps only; no device query. Mapped libraries alone do not prove GPU execution.'}


def cpu_profile_report(events):
    nodes=[e for e in events if e.get('cat')=='Node']
    kernels=[e for e in nodes if e.get('name','').endswith('_kernel_time')]
    unexpected=[]
    for event in nodes:
        provider=event.get('args',{}).get('provider')
        if (provider is not None and provider!='CPUExecutionProvider') or (
                provider is None and event.get('name','').endswith('_kernel_time')):
            unexpected.append({'node':event.get('name'),'provider':provider})
    return {'kernel_events':len(kernels),'passed':bool(kernels) and not unexpected,
            'unexpected':unexpected}


def cpu_stat_paths(hardware_path):
    if hardware_path:
        hardware=json.loads(hardware_path.read_text())
        cgroups=hardware.get('cgroups') or {}
    else:
        from diagnose_cpu import collect_cgroups
        cgroups=collect_cgroups()
    return sorted({str(Path(row['path'])/'cpu.stat')
                   for row in cgroups.get('visible_hierarchy',[]) if 'cpu.stat' in row.get('files',{})})


def cpu_stat_snapshot(paths):
    rows=[]
    for path in paths:
        try:
            raw=Path(path).read_text().strip()
            values={}
            for line in raw.splitlines():
                parts=line.split()
                if len(parts)==2:
                    try:values[parts[0]]=int(parts[1])
                    except ValueError:pass
            rows.append({'path':path,'values':values,'raw':raw})
        except OSError as exc:
            rows.append({'path':path,'error':repr(exc)})
    return {'captured_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'monotonic_seconds':time.perf_counter(),'stats':rows}


def cpu_stat_delta(before,after):
    earlier={r['path']:r for r in before['stats']}
    rows=[]
    for row in after['stats']:
        old=earlier.get(row['path'],{}).get('values',{})
        new=row.get('values',{})
        common=set(old)&set(new)
        reset=sorted(k for k in common if new[k]<old[k])
        rows.append({'path':row['path'],'delta':{k:new[k]-old[k] for k in sorted(common) if k not in reset},
                     'counters_decreased_or_reset':reset})
    return {'elapsed_seconds':after['monotonic_seconds']-before['monotonic_seconds'],'stats':rows,
            'units':{'throttled_time':'nanoseconds (cgroup v1)','throttled_usec':'microseconds (cgroup v2)',
                     'usage_usec':'microseconds','nr_throttled':'period count','nr_periods':'period count'},
            'scope':'Cgroup-wide counters include other processes and inter-call orchestration. No counters are read inside inference timers.'}


def affinity_plan(path,threads,original):
    if path is None:return None
    if original is None:raise RuntimeError('CPU affinity plans require os.sched_getaffinity/setaffinity')
    value=json.loads(path.read_text())
    if not isinstance(value,dict):raise ValueError('Affinity plan must map thread-count strings to CPU lists')
    plan={}
    for count in threads:
        cpus=value.get(str(count))
        if not isinstance(cpus,list) or not cpus or any(isinstance(c,bool) or not isinstance(c,int) or c<0 for c in cpus):
            raise ValueError(f'Affinity plan needs a nonempty integer CPU list for {count} threads')
        if len(set(cpus))!=len(cpus):raise ValueError(f'Duplicate CPUs in {count}-thread affinity mask')
        if not set(cpus).issubset(original):raise ValueError(f'{count}-thread mask exceeds original allowed CPUs')
        if len(cpus)<count:raise ValueError(f'{count}-thread mask contains fewer than {count} allowed logical CPUs')
        plan[count]=sorted(cpus)
    return plan


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def require_hash(path, expected):
    if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', expected):
        raise ValueError('A 64-character SHA256 is required for ' + str(path))
    actual = sha(path)
    if actual != expected.lower():
        raise ValueError('SHA256 mismatch for ' + str(path))
    return actual


def relative_path(base, value):
    if (not isinstance(value, str) or not value or Path(value).is_absolute()
            or PureWindowsPath(value).drive or '\\' in value):
        raise ValueError('Artifact paths must be relative paths using forward slashes')
    return (base / value).resolve()


def positive_integer(value):
    return type(value) is int and value > 0


def uid_list(value, label):
    if (not isinstance(value, list) or not value
            or any(not isinstance(v, str) or not v for v in value)
            or len(set(value)) != len(value)):
        raise ValueError(label + ' must be a nonempty list of unique UID strings')
    return value


def model_external_data(path):
    """Inspect external tensor locations without loading weights or executing ONNX."""
    import onnx
    model = onnx.load(str(path), load_external_data=False)
    locations = set()

    def visit(message):
        if isinstance(message, onnx.TensorProto) and message.data_location == onnx.TensorProto.EXTERNAL:
            entries = {v.key: v.value for v in message.external_data}
            location = entries.get('location')
            if not location:
                raise ValueError('External tensor lacks a location in ' + str(path))
            locations.add(relative_path(path.parent, location))
        for field, value in message.ListFields():
            if field.type != field.TYPE_MESSAGE:
                continue
            children = value if field.is_repeated else (value,)
            for child in children:
                visit(child)

    visit(model)
    return locations


def read_config(path, expected_hash):
    """Pin all file bytes before any ORT session or custom library is loaded."""
    require_hash(path, expected_hash)
    config = json.loads(path.read_text())
    if not isinstance(config, dict):
        raise ValueError('Config must be an object')
    models = config.get('models')
    if not isinstance(models, list) or not models or any(not isinstance(m, dict) for m in models):
        raise ValueError('A nonempty model list is required')
    names = [m.get('name') for m in models]
    uid_list(names, 'Model names')
    if models[0].get('name') != 'audio_stock' or models[0].get('kind') != 'audio':
        raise ValueError('First model must be audio_stock with kind audio')
    contracts = config.get('kind_contracts')
    if not isinstance(contracts, dict):
        raise ValueError('Explicit kind_contracts are required')
    required = set()
    if config.get('custom_library'):
        required.add(config['custom_library'])
    for model in models:
        if model.get('backend', 'ort') != 'ort':
            raise ValueError('Only the ORT backend is supported')
        if type(model.get('causal')) is not bool:
            raise ValueError('Each model must explicitly declare causal true or false')
        kind = model.get('kind')
        if not isinstance(kind, str) or kind not in contracts:
            raise ValueError('Missing model kind contract')
        contract = contracts[kind]
        if not isinstance(contract, dict) or any(not positive_integer(contract.get(k))
                for k in ('sample_rate', 'hop', 'latent_channels')):
            raise ValueError('Positive sample_rate, hop and latent_channels are required')
        required.add(model['path'])
        case_path = config[kind + '_cases']
        required.update((case_path, str(Path(case_path).with_suffix('.json'))))
        if model.get('custom_library') and model.get('custom_libraries'):
            raise ValueError('Use either custom_library or custom_libraries')
        if model.get('custom_library'):
            required.add(model['custom_library'])
        libraries = model.get('custom_libraries', [])
        external = model.get('external_data', [])
        for values in (libraries, external):
            if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
                raise ValueError('custom_libraries and external_data must be path lists')
            required.update(values)
        extra = model.get('extra_inputs', {})
        if not isinstance(extra, dict) or any(not isinstance(k, str) or not k
                or not isinstance(v, str) or not v for k, v in extra.items()):
            raise ValueError('extra_inputs must map ONNX input names to NPZ keys')
    if not positive_integer(config.get('expected_case_count')):
        raise ValueError('expected_case_count must be a positive integer')
    expected_uids = uid_list(config.get('expected_uids'), 'expected_uids')
    if len(expected_uids) != config['expected_case_count']:
        raise ValueError('expected_case_count and expected_uids disagree')
    if 'timing_uids' in config:
        selected = uid_list(config['timing_uids'], 'timing_uids')
        if not set(selected).issubset(expected_uids):
            raise ValueError('Timing UID outside expected_uids')
    hashes = config.get('artifact_sha256')
    if not isinstance(hashes, dict) or not required.issubset(hashes):
        raise ValueError('artifact_sha256 must cover every input artifact and case companion JSON')
    base = path.resolve().parent
    verified = {name: require_hash(relative_path(base, name), value) for name, value in hashes.items()}
    for model in models:
        declared = {relative_path(base, p) for p in model.get('external_data', [])}
        actual = model_external_data(relative_path(base, model['path']))
        if actual != declared:
            raise ValueError('external_data must exactly list graph tensor files for ' + model['name'])
    return config, verified


def compare(actual,reference):
    if actual.shape!=reference.shape or actual.dtype!=np.float32 or reference.dtype!=np.float32:
        return {"passed":False,"actual_shape":list(actual.shape),"reference_shape":list(reference.shape),
                "actual_dtype":str(actual.dtype),"reference_dtype":str(reference.dtype)}
    delta=actual.astype(np.float64)-reference
    return {"passed":bool(np.isfinite(actual).all() and np.isfinite(reference).all() and np.allclose(actual,reference,atol=1e-5,rtol=1e-4)),
            "max_abs":float(np.max(np.abs(delta))),"rmse":float(np.sqrt(np.mean(delta*delta)))}


def validate_case(z, reference, contract, uid):
    if (z.dtype != np.float32 or z.ndim != 3 or z.shape[:2] != (1, contract['latent_channels'])
            or z.shape[-1] < 1 or not np.isfinite(z).all()):
        raise ValueError('Invalid finite FP32 [1,C,L] latent for ' + uid)
    expected = (1, 1, contract['hop'] * z.shape[-1])
    if reference.dtype != np.float32 or reference.shape != expected or not np.isfinite(reference).all():
        raise ValueError('Expected a complete finite FP32 reference of shape ' + str(expected) + ' for ' + uid)
    return {'z': np.ascontiguousarray(z), 'ref': np.ascontiguousarray(reference)}


def read_cases(path, contract):
    meta=json.loads(path.with_suffix(".json").read_text())
    with np.load(path,allow_pickle=False) as z:
        if len(z.files) != len(set(z.files)):
            raise ValueError('Duplicate NPZ keys')
        ids = [k[:-3] for k in z.files if k.endswith('__z')]
        uid_list(ids, 'Case identifiers')
        if {k[:-5] for k in z.files if k.endswith('__ref')} != set(ids):
            raise ValueError('Every latent must have exactly one matching full reference')
        cases = {uid: validate_case(z[uid+'__z'], z[uid+'__ref'], contract, uid) for uid in ids}
    timed_ids = uid_list(meta.get('timed_ids'), 'Case metadata timed_ids')
    if not set(timed_ids).issubset(cases):
        raise ValueError('Case metadata timed_ids contains an unknown UID')
    if cases[timed_ids[0]]['z'].shape[-1] < 17:
        raise ValueError('First timed case must contain at least 17 latent frames for fixed probes')
    return cases,meta



def select_timing_ids(data,meta,scope,override=None):
    """Keep the frozen case metadata intact while validating a common timing subset."""
    if not data:
        raise ValueError('No codec cases supplied')
    if override is not None:
        if not isinstance(override,list) or not override or any(not isinstance(v,str) or not v for v in override):
            raise ValueError('timing_uids must be a nonempty list of UID strings')
        if len(set(override))!=len(override):
            raise ValueError('Duplicate timing UID')
        if any(not set(override).issubset(cases) for cases in data.values()):
            raise ValueError('Timing UID outside the frozen common corpus')
        return {kind:list(override) for kind in data}
    result={kind:list(cases) if scope=='all' else list(meta[kind]['timed_ids']) for kind,cases in data.items()}
    for kind,ids in result.items():
        if not ids or len(set(ids))!=len(ids) or not set(ids).issubset(data[kind]):
            raise ValueError('Invalid legacy timing identifiers')
    if any(ids!=next(iter(result.values())) for ids in result.values()):
        raise ValueError('Timing identifiers must match across codecs')
    return result


def current_affinity():
    try:
        if not hasattr(os, 'sched_setaffinity'):
            return None
        return set(os.sched_getaffinity(0))
    except (AttributeError, OSError, NotImplementedError):
        return None


def optional_text(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, UnicodeError):
        return None


def host_snapshot():
    try:
        load = list(os.getloadavg())
    except (AttributeError, OSError, NotImplementedError):
        load = None
    cpu = None
    raw = optional_text('/proc/stat')
    if raw:
        try:
            first = raw.splitlines()[0].split()
            values = [int(v) for v in first[1:]]
            if first[0] == 'cpu' and len(values) >= 8 and all(v >= 0 for v in values):
                cpu = values
        except (ValueError, IndexError):
            pass
    try:
        ticks = os.sysconf('SC_CLK_TCK')
    except (AttributeError, OSError, ValueError):
        ticks = None
    return {'cpu_ticks': cpu, 'loadavg': load,
            'meminfo': optional_text('/proc/meminfo'), 'clock_ticks_per_second': ticks,
            'scope': 'Optional OS counters outside inference timers; missing counters are null'}


def host_delta(before, after):
    # user/nice already include guest ticks: do not count guest twice.
    if (before['cpu_ticks'] is None or after['cpu_ticks'] is None
            or len(before['cpu_ticks']) < 8 or len(after['cpu_ticks']) < 8):
        return {'steal_percent': None, 'iowait_percent': None, 'scope': 'Linux CPU counters unavailable on this platform'}
    d = [b-a for a,b in zip(before['cpu_ticks'][:8], after['cpu_ticks'][:8])]
    if any(v < 0 for v in d):
        return {'steal_percent': None, 'iowait_percent': None, 'scope': 'OS counters decreased or reset'}
    total = sum(d)
    return {'ticks': d, 'total_ticks': total, 'steal_percent': 100*d[7]/total if total else None,
            'iowait_percent': 100*d[4]/total if total else None,
            'scope': 'Whole-VM /proc/stat counters around timing, outside inference timers'}


def _main():
    global np,ort
    parser=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--config-sha256",required=True,help="SHA256 of the frozen comparison config")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--threads",default="1,4")
    parser.add_argument("--validation",choices=("timed","all"),default="all")
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--timing-cases",choices=("timed","all"),default="all")
    parser.add_argument("--warmups",type=int,default=1)
    parser.add_argument("--affinity-plan",type=Path)
    parser.add_argument("--hardware",type=Path,help="JSON from diagnose_cpu.py; used to locate current cgroup cpu.stat files")
    parser.add_argument("--profile",action="store_true")
    args=parser.parse_args()
    cpu_environment=require_cpu_environment()
    import numpy as np
    import onnxruntime as ort
    if args.repeats<1 or args.warmups<1:raise ValueError('Positive repeats and warmups required')
    thread_counts=[int(t) for t in args.threads.split(',')]
    if not thread_counts or any(t<1 for t in thread_counts) or len(set(thread_counts))!=len(thread_counts):
        raise ValueError('Thread counts must be distinct positive integers')
    original_affinity=current_affinity()
    plan=affinity_plan(args.affinity_plan,thread_counts,original_affinity)
    stat_paths=cpu_stat_paths(args.hardware)
    if ort.__version__!="1.29.0":raise RuntimeError(f"Expected ORT1.29.0; got {ort.__version__}")
    config,verified_hashes=read_config(args.config,args.config_sha256)
    base=args.config.resolve().parent
    resolve=lambda p:relative_path(base,p)
    if {args.output.resolve(),args.output.with_suffix('.tmp').resolve()} & {args.config.resolve(),*(resolve(p) for p in verified_hashes)}:
        raise ValueError("Output must not overwrite a config or input artifact")
    models=config["models"]
    if len({m['name'] for m in models})!=len(models):raise ValueError("Unique model names required")
    if models[0]["name"]!="audio_stock" or models[0]["kind"]!="audio":
        raise ValueError("First model must be audio_stock")
    model_paths={m["name"]:resolve(m["path"]) for m in models}
    data={};meta={};extra_inputs={}
    contracts=config['kind_contracts']
    for kind in {m["kind"] for m in models}:
        data[kind],meta[kind]=read_cases(resolve(config[kind+"_cases"]),contracts[kind])
    for kind in data:
        if kind not in contracts or contracts[kind]['sample_rate']<=0 or contracts[kind]['hop']<=0:
            raise ValueError('Missing positive sample-rate/hop contract: '+kind)
        if config.get('expected_case_count') is not None and len(data[kind])!=config['expected_case_count']:
            raise ValueError('Incomplete frozen-corpus coverage for '+kind)
        if config.get('expected_uids') is not None and set(data[kind])!=set(config['expected_uids']):
            raise ValueError('Frozen-corpus identifiers changed for '+kind)
        if meta['audio']['timed_ids']!=meta[kind]['timed_ids'] or set(data['audio'])!=set(data[kind]):
            raise ValueError('Matched complete corpus and representative IDs required')
    for m in models:
        feeds={}
        if m.get('extra_inputs'):
            with np.load(resolve(config[m['kind']+'_cases']),allow_pickle=False) as arrays:
                for name,key in m['extra_inputs'].items():
                    feeds[name]=np.ascontiguousarray(arrays[key])
                    if feeds[name].dtype!=np.float32 or not np.isfinite(feeds[name]).all():raise ValueError('Invalid fixed FP32 auxiliary input')
        extra_inputs[m['name']]=feeds
    timing_ids=select_timing_ids(data,meta,args.timing_cases,config.get('timing_uids'))
    if args.profile and any(m['name']=='fast_default' and len(meta[m['kind']]['timed_ids'])<2 for m in models):
        raise ValueError('Profiling fast_default requires at least two metadata timed_ids')
    library=resolve(config["custom_library"]) if config.get("custom_library") else None
    record={"status":"running","started":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
            "onnxruntime":ort.__version__,"ort_build":ort.get_build_info(),"numpy":np.__version__,
            "platform":platform.platform(),"machine":platform.machine(),"config":config,
            "config_sha256":args.config_sha256.lower(),"harness_sha256":sha(__file__),
            "verified_artifact_sha256":verified_hashes,
            "source_campaign_sha256":SOURCE_CAMPAIGN_SHA256,
            "cpu_only_visibility_environment":cpu_environment,
            "available_providers_inventory_only":ort.get_available_providers(),
            "session_provider_checks":[],"profile_provider_checks":[],
            "original_allowed_cpus":sorted(original_affinity) if original_affinity is not None else None,
            "affinity_plan":{str(k):v for k,v in plan.items()} if plan else None,
            "affinity_plan_sha256":sha(args.affinity_plan) if args.affinity_plan else None,
            "effective_affinity_by_threads":{},
            "hardware_diagnostic":str(args.hardware.resolve()) if args.hardware else None,
            "hardware_diagnostic_sha256":sha(args.hardware) if args.hardware else None,
            "cpu_stat_paths":stat_paths,"cpu_stat_by_threads":{},
            "gpu_library_mappings_start":gpu_library_mappings(),
            "library_sha256":sha(library) if library else None,
            "models":[dict(m,sha256=verified_hashes[m['path']]) for m in models],
            "fixed_auxiliary_inputs":{name:{key:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':hashlib.sha256(v.tobytes()).hexdigest()} for key,v in feeds.items()} for name,feeds in extra_inputs.items()},
            "case_metadata":meta,"case_hashes":{k:sha(resolve(config[k+'_cases'])) for k in data},
            "protocol":{"threads":args.threads,"precision":"FP32","batch":1,"provider":"CPUExecutionProvider",
                        "inter_op_threads":1,"execution":"sequential","graph_optimization":"all","spinning":False,
                        "warmups_per_timed_shape":args.warmups,"warmup_method":"separate calls after full validation","repeats":args.repeats,"validation_scope":args.validation,
                        "timing_scope":"frozen_config_override" if config.get("timing_uids") is not None else args.timing_cases,"timing_ids":timing_ids,"stock_full_output_reused_for_validation":True,
                        "timing":"Python s.run wrapper including outputs/shape checks; disk I/O, encoding, setup, validation, warmup and profiling excluded",
                        "denominator":"Complete generated audio duration including hop padding; decoder-only fresh full calls, not cached streams",
                        "gate":"atol1e-5 rtol1e-4 vs original saved waveform; AudioVAE2 also vs stock ONNX. Exact repeated-call gates for all; future-prefix gates only for codecs declaring a causal contract."},
            "validation":[],"failed":[],"measurements":[],"summary":{},"profiles":[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        temp=args.output.with_suffix(".tmp");temp.write_text(json.dumps(record,indent=2)+"\n");temp.replace(args.output)
    save()
    last_progress=[time.monotonic()]
    def pulse(phase,**fields):
        # Progress reporting is outside every inference timer.
        if time.monotonic()-last_progress[0]>=30:
            record['progress']={'phase':phase,**fields}
            save();print('PROGRESS',json.dumps(record['progress']),flush=True)
            last_progress[0]=time.monotonic()
    def session(m,threads,profile=None):
        opt=ort.SessionOptions();opt.intra_op_num_threads=threads;opt.inter_op_num_threads=1
        opt.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;opt.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opt.add_session_config_entry("session.intra_op.allow_spinning","0")
        opt.add_session_config_entry("session.inter_op.allow_spinning","0")
        if m.get('custom_libraries'):
            if m.get('custom_library'):
                raise ValueError('Use either custom_library or custom_libraries')
            if not isinstance(m['custom_libraries'],list) or not all(isinstance(v,str) for v in m['custom_libraries']):
                raise ValueError('custom_libraries must be a list of paths')
            model_libraries=[resolve(v) for v in m['custom_libraries']]
        else:
            model_library=resolve(m['custom_library']) if m.get('custom_library') else library
            model_libraries=[model_library] if model_library else []
        for model_library in model_libraries:
            opt.register_custom_ops_library(str(model_library))
        if profile:opt.enable_profiling=True;opt.profile_file_prefix=str(profile)
        s=ort.InferenceSession(str(model_paths[m['name']]),sess_options=opt,providers=['CPUExecutionProvider'])
        s.disable_fallback()
        actual_providers=s.get_providers()
        record['session_provider_checks'].append(dict(model=m['name'],threads=threads,
                                                     profile=bool(profile),providers=actual_providers,
                                                     custom_libraries=[{'path':str(v),'sha256':sha(v)} for v in model_libraries]))
        if actual_providers!=['CPUExecutionProvider']:raise RuntimeError("Refusing non-CPU execution provider")
        inputs=s.get_inputs();extra=extra_inputs[m['name']]
        outputs=s.get_outputs()
        if len(outputs)!=1 or outputs[0].type!='tensor(float)':
            raise RuntimeError('Expected one FP32 waveform output')
        primary=[v for v in inputs if v.name not in extra]
        if len(primary)!=1 or any(v.type!='tensor(float)' for v in inputs) or set(v.name for v in inputs)!=set(extra)|{primary[0].name}:
            raise RuntimeError('Unexpected FP32 input contract')
        return s,{'latent':primary[0].name,'extra':extra}
    def run(s,kind,z):
        y=s[0].run(None,{s[1]['latent']:z,**s[1]['extra']})[0]
        expected=contracts[kind]['hop']*z.shape[-1]
        if y.dtype!=np.float32 or y.shape!=(1,1,expected):raise ValueError(f"Wrong output shape {y.shape}")
        return y.reshape(1,1,-1)
    for threads in thread_counts:
        if plan is not None:
            os.sched_setaffinity(0,set(plan[threads]))
        effective=sorted(current_affinity()) if original_affinity is not None else None
        if plan is not None and effective!=plan[threads]:raise RuntimeError('Requested CPU affinity was not applied exactly')
        record['effective_affinity_by_threads'][str(threads)]={'cpus':effective,
            'applied_before_session_creation':True,
            'scope':'Calling thread mask inherited by subsequently created ORT workers; no GPU API called.'}
        thread_stats={'campaign_before':cpu_stat_snapshot(stat_paths)}
        record['cpu_stat_by_threads'][str(threads)]=thread_stats
        save()
        sessions={};passed=[]
        for m in models:
            try:sessions[m['name']]=session(m,threads)
            except Exception as exc:
                record['failed'].append({'model':m['name'],'threads':threads,'phase':'session','error':repr(exc)});save()
                if m['name']=='audio_stock':record['status']='failed';save();raise
        stock=sessions['audio_stock']
        ids=list(data['audio']) if args.validation=='all' else meta['audio']['timed_ids']
        stock_ref={}
        for number,uid in enumerate(ids,1):
            stock_ref[uid]=run(stock,'audio',data['audio'][uid]['z'])
            pulse('stock_reference',threads=threads,completed=number,total=len(ids))
        first=data['audio'][meta['audio']['timed_ids'][0]]['z']
        short_ref={length:run(stock,'audio',np.ascontiguousarray(first[...,:length])) for length in (1,2,7,17)}
        for m in models:
            name,kind=m['name'],m['kind']
            if name not in sessions:continue
            s=sessions[name];valid_ids=list(data[kind]) if args.validation=='all' else meta[kind]['timed_ids']
            try:
                for number,uid in enumerate(valid_ids,1):
                    case=data[kind][uid];y=stock_ref[uid] if name=='audio_stock' else run(s,kind,case['z']);ref=case['ref']
                    result=compare(y,ref)
                    if kind=='audio':result['vs_stock']=compare(y,stock_ref[uid]);result['passed']&=result['vs_stock']['passed']
                    row=dict(model=name,threads=threads,test='real_waveform',uid=uid,**result)
                    record['validation'].append(row)
                    if not row['passed']:raise ValueError(str(row))
                    pulse('validation',model=name,threads=threads,completed=number,total=len(valid_ids))
                if kind=='audio':
                    for length,ref in short_ref.items():
                        result=compare(run(s,kind,np.ascontiguousarray(first[...,:length])),ref)
                        row=dict(model=name,threads=threads,test='dynamic_length',length=length,**result)
                        record['validation'].append(row)
                        if not result['passed']:raise ValueError(str(row))
                z=np.ascontiguousarray(data[kind][meta[kind]['timed_ids'][0]]['z'][...,:17]);cut=7
                changed=z.copy();changed[...,cut:]=np.random.default_rng(829).standard_normal(changed[...,cut:].shape).astype(np.float32)*0.1
                y=run(s,kind,z);future=run(s,kind,changed);repeated=run(s,kind,z)
                hop=contracts[kind]['hop'];requires_causality=m.get('causal',True)
                difference=float(np.max(np.abs(y[...,:cut*hop]-future[...,:cut*hop])))
                repeat_diff=float(np.max(np.abs(y-repeated)))
                row=dict(model=name,threads=threads,test='future_and_repeated_call',prefix_max_abs=difference,
                         repeated_max_abs=repeat_diff,causality_required=requires_causality,
                         future_invariant=difference==0,passed=(difference==0 or not requires_causality) and repeat_diff==0)
                record['validation'].append(row)
                if not row['passed']:raise ValueError(str(row))
                prefix=run(s,kind,np.ascontiguousarray(z[...,:cut]))
                if requires_causality:
                    row=dict(model=name,threads=threads,test='shorter_prefix',**compare(prefix,y[...,:cut*hop]))
                    record['validation'].append(row)
                    if not row['passed']:raise ValueError(str(row))
                else:
                    record['validation'].append(dict(model=name,threads=threads,test='noncausal_shorter_shape',
                        passed=bool(np.isfinite(prefix).all()),causality_required=False,
                        note='A shorter-input prefix is not required to match for this explicitly noncausal codec.'))
                    if record['validation'][-1]['passed'] is not True:raise ValueError('Nonfinite noncausal shorter output')
                after_short=run(s,kind,z)
                difference=float(np.max(np.abs(y-after_short)))
                row=dict(model=name,threads=threads,test='long_short_long',max_abs=difference,passed=difference==0)
                record['validation'].append(row)
                if not row['passed']:raise ValueError(str(row))
                passed.append(m)
                for uid in timing_ids[kind]:
                    for _ in range(args.warmups):run(s,kind,data[kind][uid]['z'])
                print('PASS',threads,name,flush=True)
            except Exception as exc:
                record['failed'].append(dict(model=name,threads=threads,phase='validation',error=repr(exc)))
                print('FAIL',threads,name,repr(exc),flush=True)
                if name=='audio_stock':record['status']='failed';save();raise
            save()
        if record['failed'] or len(passed)!=len(models) or any(v.get('passed') is not True for v in record['validation']):
            record['status']='failed';save();raise RuntimeError('Refusing timing after failed or missing model validation')
        thread_stats['timing_before']=cpu_stat_snapshot(stat_paths)
        thread_stats['host_timing_before']=host_snapshot()
        for repeat in range(args.repeats):
            jobs=[(m,uid) for m in passed for uid in timing_ids[m['kind']]]
            random.Random(9207+threads+repeat).shuffle(jobs)
            for number,(m,uid) in enumerate(jobs,1):
                z=data[m['kind']][uid]['z'];start=time.perf_counter();y=run(sessions[m['name']],m['kind'],z)
                elapsed=time.perf_counter()-start;rate=contracts[m['kind']]['sample_rate']
                record['measurements'].append(dict(model=m['name'],backend='ort',threads=threads,repeat=repeat,uid=uid,
                                                   seconds=elapsed,audio_s=y.shape[-1]/rate,samples=y.shape[-1]))
                pulse('timing',threads=threads,repeat=repeat+1,completed=number,total=len(jobs))
            save();print('TIMED',threads,'repeat',repeat+1,flush=True)
        thread_stats['host_timing_after']=host_snapshot()
        thread_stats['host_timing_delta']=host_delta(thread_stats['host_timing_before'],thread_stats['host_timing_after'])
        thread_stats['timing_after']=cpu_stat_snapshot(stat_paths)
        thread_stats['timing_delta']=cpu_stat_delta(thread_stats['timing_before'],thread_stats['timing_after'])
        for m in passed:
            rows=[r for r in record['measurements'] if r['model']==m['name'] and r['threads']==threads]
            reps=[]
            for r in range(args.repeats):
                subset=[x for x in rows if x['repeat']==r]
                reps.append(sum(x['seconds'] for x in subset)/sum(x['audio_s'] for x in subset))
            record['summary'].setdefault(m['name'],{})[str(threads)]={
                'decoder_rtf':sum(x['seconds'] for x in rows)/sum(x['audio_s'] for x in rows),
                'repeat_rtfs':reps,'observations':len(rows)}
        for obj,_ in sessions.values():
            if hasattr(obj,'close'):obj.close()
        del stock,stock_ref,short_ref,sessions,s,obj
        gc.collect();save()
        if args.profile:
            for m in passed:
                if m['name']!='fast_default':continue
                prefix=args.output.parent/(args.output.stem+'_'+m['name']+f'_{threads}t')
                s=session(m,threads,prefix);z=data[m['kind']][meta[m['kind']]['timed_ids'][1]]['z']
                for _ in range(5):run(s,m['kind'],z)
                path=Path(s[0].end_profiling());del s
                events=json.loads(path.read_text());runs=sorted([e for e in events if e.get('name')=='model_run' and e.get('ph')=='X'],key=lambda e:e['ts'])[2:]
                check=cpu_profile_report(events)
                record['profile_provider_checks'].append(dict(model=m['name'],threads=threads,**check))
                if not check['passed']:
                    record['failed'].append(dict(model=m['name'],threads=threads,phase='profile_provider',
                        error='Missing CPU provider proof or non-CPU node event',unexpected=check['unexpected']))
                    record['status']='failed';record['gpu_library_mappings_end']=gpu_library_mappings();save()
                    raise RuntimeError('Refusing profile without exclusively CPUExecutionProvider kernel events')
                nodes={}
                for e in events:
                    if e.get('cat')!='Node' or not e.get('name','').endswith('_kernel_time'):continue
                    if not any(r['ts']<=e['ts']<r['ts']+r['dur'] for r in runs):continue
                    a=e.get('args',{});n=nodes.setdefault(e['name'],dict(name=e['name'],op=a.get('op_name'),provider=a.get('provider'),calls=0,us=0.,
                                                                       input=a.get('input_type_shape'),output=a.get('output_type_shape')))
                    n['calls']+=1;n['us']+=e['dur']
                ops=defaultdict(float)
                for n in nodes.values():ops[n['op']]+=n['us']
                record['profiles'].append(dict(model=m['name'],threads=threads,nodes=list(nodes.values()),
                                               operator_us=dict(ops),trace=str(path),scope='Separate profiled kernel wall durations'))
                save()
        thread_stats['campaign_after']=cpu_stat_snapshot(stat_paths)
        thread_stats['campaign_delta']=cpu_stat_delta(thread_stats['campaign_before'],thread_stats['campaign_after'])
        save()
    record['gpu_library_mappings_end']=gpu_library_mappings()
    if original_affinity is not None:
        os.sched_setaffinity(0,original_affinity)
        record['restored_allowed_cpus']=sorted(os.sched_getaffinity(0))
    if record['failed'] or len(record['measurements'])!=len(models)*len(thread_counts)*args.repeats*len(next(iter(timing_ids.values()))):
        record['status']='failed';save();raise RuntimeError('Incomplete or failing campaign')
    record['status']='complete';record['finished']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());save()
    print(json.dumps(record['summary'],indent=2),flush=True)


def main():
    original = current_affinity()
    try:
        _main()
    finally:
        if original is not None and current_affinity() != original:
            os.sched_setaffinity(0, original)


if __name__ == '__main__':
    main()
