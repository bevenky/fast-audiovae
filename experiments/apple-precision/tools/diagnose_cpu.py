#!/usr/bin/env python3
"""Read-only CPU/protocol diagnostics; no GPU query, model run, or network call.

Default requires only Python's standard library. --ort-probe additionally
creates a tiny CPUExecutionProvider-only Identity session WITHOUT calling run.
The exported cpu_session() helper enforces the same provider and pool policy
for a caller's actual benchmark. Available providers are inventory, not usage.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

THREAD_ENV = (
    'OMP_NUM_THREADS', 'OMP_WAIT_POLICY', 'OMP_PROC_BIND', 'OMP_PLACES',
    'OMP_DYNAMIC', 'OMP_THREAD_LIMIT', 'KMP_BLOCKTIME', 'KMP_AFFINITY',
    'GOMP_CPU_AFFINITY', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'MKL_DYNAMIC', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
)
CPU_ENV = {'CUDA_VISIBLE_DEVICES': '-1', 'NVIDIA_VISIBLE_DEVICES': 'void',
           'ROCR_VISIBLE_DEVICES': '-1', 'HIP_VISIBLE_DEVICES': '-1'}
CGROUP_FILES = (
    'cpu.max', 'cpu.max.burst', 'cpu.weight', 'cpu.stat', 'cpu.pressure',
    'cpuset.cpus', 'cpuset.cpus.effective', 'cpuset.mems', 'cpuset.mems.effective',
    'cpu.cfs_quota_us', 'cpu.cfs_period_us', 'cpu.shares',
    'cpuset.effective_cpus', 'cpuset.effective_mems',
)
# ONNX IR8/opset13, one FP32[1] input/output and one Identity. Checker verified
# during preparation. Session construction below performs no model execution.
PROBE = bytes.fromhex(
    '08083a480a100a017812017922084964656e7469747912126370755f70726f7669646572'
    '5f70726f62655a0f0a0178120a0a08080112040a020801620f0a0179120a0a08080112'
    '040a02080142040a00100d'
)


def read_text(path):
    try:
        return Path(path).read_text(errors='replace').strip()
    except (OSError, ValueError):
        return None


def current_affinity():
    try:
        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError, NotImplementedError):
        return None


def load_average():
    try:
        return list(os.getloadavg())
    except (AttributeError, OSError, NotImplementedError):
        return None


def cpu_list(value):
    if value is None:
        return None
    result = set()
    if not value.strip():
        return []
    for part in value.split(','):
        match = re.fullmatch(r'\s*(\d+)(?:-(\d+))?\s*', part)
        if not match:
            raise ValueError('Invalid CPU list: ' + value)
        low, high = int(match[1]), int(match[2] or match[1])
        if high < low or high - low > 1_000_000:
            raise ValueError('Invalid CPU range: ' + part)
        result.update(range(low, high + 1))
    return sorted(result)


def parse_cpuinfo(value):
    records = []
    for section in (value or '').split('\n\n'):
        record = {}
        for line in section.splitlines():
            if ':' in line:
                k, v = line.split(':', 1)
                record[k.strip()] = v.strip()
        if record:
            records.append(record)
    return records


def mount_decode(value):
    return re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), value)


def cgroup_mounts(text):
    result = []
    for line in (text or '').splitlines():
        if ' - ' not in line:
            continue
        left, right = line.split(' - ', 1)
        fields, extra = left.split(), right.split()
        if len(fields) >= 5 and len(extra) >= 3 and extra[0] in ('cgroup', 'cgroup2'):
            result.append({'type': extra[0], 'root': mount_decode(fields[3]),
                           'mount': mount_decode(fields[4]),
                           'controllers': extra[2].split(',')})
    return result


def resolve_cgroup(mount, process_path):
    root, current = Path(mount['root']), Path(process_path)
    if '..' in current.parts:
        return None, 'unsupported parent-relative cgroup path'
    try:
        suffix = current.relative_to(root)
        return Path(mount['mount']) / suffix, 'mount-root relative'
    except ValueError:
        # A private cgroup namespace can report / relative to the mounted
        # container hierarchy while mountinfo exposes a host-relative root.
        return Path(mount['mount']) / process_path.lstrip('/'), 'namespace-relative fallback'


def quota_from_files(files):
    value = files.get('cpu.max')
    if value:
        fields = value.split()
        if len(fields) == 2 and fields[0] != 'max':
            try:
                quota, period = int(fields[0]), int(fields[1])
            except ValueError:
                return None
            if quota > 0 and period > 0:
                return quota / period
    quota, period = files.get('cpu.cfs_quota_us'), files.get('cpu.cfs_period_us')
    if quota is not None and period is not None:
        try:
            quota, period = int(quota), int(period)
        except ValueError:
            return None
        if quota > 0 and period > 0:
            return quota / period
    return None


def collect_cgroups():
    membership = read_text('/proc/self/cgroup')
    mounts = cgroup_mounts(read_text('/proc/self/mountinfo'))
    records, seen = [], set()
    for line in (membership or '').splitlines():
        fields = line.split(':', 2)
        if len(fields) != 3:
            continue
        _, controllers, process_path = fields
        names = set(controllers.split(',')) if controllers else set()
        for mount in mounts:
            matched = (mount['type'] == 'cgroup2' and not names) or (
                mount['type'] == 'cgroup' and bool(names & set(mount['controllers'])))
            if not matched:
                continue
            directory, method = resolve_cgroup(mount, process_path)
            if directory is None:
                continue
            boundary = Path(mount['mount'])
            # Inspect all visible ancestors: a parent's quota can be tighter
            # than cpu.max at the process's own node.
            for _ in range(64):
                if directory not in seen:
                    seen.add(directory)
                    files = {name: value for name in CGROUP_FILES
                             if (value := read_text(directory / name)) is not None}
                    if files:
                        records.append({'path': str(directory), 'type': mount['type'],
                                        'resolution': method, 'files': files,
                                        'quota_cpu_equivalents': quota_from_files(files)})
                if directory == boundary or boundary not in directory.parents:
                    break
                directory = directory.parent
    quotas = [r['quota_cpu_equivalents'] for r in records if r['quota_cpu_equivalents'] is not None]
    return {'membership': membership, 'mounts': mounts, 'visible_hierarchy': records,
            'tightest_visible_quota_cpu_equivalents': min(quotas) if quotas else None,
            'visibility_note': 'Host ancestors outside the mounted namespace may be hidden. A quota is CPU time per period, not a dedicated-core guarantee. cpu.stat is a snapshot, not a benchmark delta.'}


def collect_cache_topology(affinity):
    """Read actual sibling/cache maps instead of assuming logical CPU numbering."""
    allowed = affinity if affinity is not None else cpu_list(read_text('/sys/devices/system/cpu/online'))
    if allowed is None:
        return {'cores': [], 'l3_groups': [], 'affinity_suggestions': None}
    core_rows, groups = [], {}
    for cpu in allowed:
        base = Path(f'/sys/devices/system/cpu/cpu{cpu}')
        topology = base / 'topology'
        row = {'cpu': cpu, 'package_id': read_text(topology / 'physical_package_id'),
               'die_id': read_text(topology / 'die_id'), 'core_id': read_text(topology / 'core_id'),
               'thread_siblings': cpu_list(read_text(topology / 'thread_siblings_list'))}
        if row['core_id'] is not None or row['thread_siblings'] is not None:
            core_rows.append(row)
        for index in sorted((base / 'cache').glob('index*')):
            if read_text(index / 'level') != '3':
                continue
            shared = cpu_list(read_text(index / 'shared_cpu_list'))
            if not shared:
                continue
            key = (read_text(index / 'type'), tuple(shared))
            if key not in groups:
                groups[key] = {'level': 3, 'type': key[0], 'shared_cpus': shared,
                               'allowed_cpus': sorted(set(shared) & set(allowed)),
                               'id': read_text(index / 'id'), 'size': read_text(index / 'size'),
                               'line_bytes': read_text(index / 'coherency_line_size'),
                               'ways': read_text(index / 'ways_of_associativity')}
    by_cpu = {r['cpu']: r for r in core_rows}

    def core_key(cpu):
        row = by_cpu.get(cpu, {})
        if row.get('thread_siblings'):
            return ('siblings', *row['thread_siblings'])
        if row.get('package_id') is not None and row.get('core_id') is not None:
            return ('topology', row['package_id'], row.get('die_id'), row['core_id'])
        return None  # No proof of different physical cores means no suggestion.

    def representatives(cpus):
        selected, seen = [], set()
        for cpu in sorted(cpus):
            key = core_key(cpu)
            if key is not None and key not in seen:
                selected.append(cpu)
                seen.add(key)
        return selected

    l3_groups = sorted(groups.values(), key=lambda g: min(g['allowed_cpus']))
    for group in l3_groups:
        group['physical_core_representatives'] = representatives(group['allowed_cpus'])
    candidates = [g for g in l3_groups if len(g['physical_core_representatives']) >= 4]
    suggestion = None
    if candidates:
        chosen = candidates[0]
        four = chosen['physical_core_representatives'][:4]
        package = by_cpu[four[0]].get('package_id')
        spread, used_cores = [], set()
        for group in l3_groups:
            for cpu in group['physical_core_representatives']:
                if by_cpu[cpu].get('package_id') == package and core_key(cpu) not in used_cores:
                    spread.append(cpu)
                    used_cores.add(core_key(cpu))
                    break
            if len(spread) == 4:
                break
        suggestion = {
            'status': 'Suggested CPU masks only; no affinity or scheduling was changed.',
            'same_l3_one_thread': four[:1], 'same_l3_four_threads': four,
            'l3_shared_cpus': chosen['shared_cpus'], 'l3_size': chosen['size'],
            'package_id': package,
            'spread_l3_four_threads_same_package': spread if len(spread) == 4 else None,
            'notes': [
                'Each selected logical CPU belongs to a different reported physical core; SMT siblings are excluded.',
                'Start with the same L3 group for repeatable one/four-thread comparisons. This is a locality choice, not proof of the fastest affinity.',
                'On EPYC, shared L3 groups identify a useful locality boundary; read the actual groups rather than inferring CCD placement from CPU numbering.',
                'The spread-L3 alternative may cross NUMA nodes. Compare against numa_nodes before choosing memory binding.',
                'Apply one CPU mask consistently to every model in a comparison, before session creation and warmup. Repeat the diagnostic inside that mask.',
            ],
        }
    return {'cores': core_rows, 'l3_groups': l3_groups, 'affinity_suggestions': suggestion}


def cpu_session(model, threads, *, expected_version='1.29.0', custom_library=None):
    """Create a CPU-only ORT session; never execute it or select a GPU EP."""
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError('threads must be a positive integer')
    if any(os.environ.get(key) != value for key, value in CPU_ENV.items()):
        raise RuntimeError('Set CPU visibility variables at launch before --ort-probe: ' + json.dumps(CPU_ENV))
    import onnxruntime as ort
    if expected_version and ort.__version__ != expected_version:
        raise RuntimeError(f'Expected ORT {expected_version}; found {ort.__version__}')
    if 'CPUExecutionProvider' not in ort.get_available_providers():
        raise RuntimeError('CPUExecutionProvider is unavailable')
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry('session.intra_op.allow_spinning', '0')
    options.add_session_config_entry('session.inter_op.allow_spinning', '0')
    if custom_library:
        options.register_custom_ops_library(str(custom_library))
    session = ort.InferenceSession(model, sess_options=options, providers=['CPUExecutionProvider'])
    session.disable_fallback()
    if session.get_providers() != ['CPUExecutionProvider']:
        raise RuntimeError('Refusing a session with a non-CPU execution provider')
    return session


def collect(ort_probe=False, expected_ort_version='1.29.0', requested_threads=(1, 4)):
    affinity = current_affinity()
    status = {k.strip(): v.strip() for line in (read_text('/proc/self/status') or '').splitlines()
              if ':' in line for k, v in [line.split(':', 1)]
              if k in ('Cpus_allowed_list', 'Mems_allowed_list', 'Threads')}
    infos = parse_cpuinfo(read_text('/proc/cpuinfo'))
    selected = [r for r in infos if affinity is None or not r.get('processor', '').isdigit()
                or int(r['processor']) in affinity]
    flags = [set((r.get('flags') or r.get('Features') or '').split()) for r in selected]
    flags = [f for f in flags if f]
    topology = [{'cpu': r.get('processor'), 'package': r.get('physical id'), 'core': r.get('core id'),
                 'model': r.get('model name') or r.get('Processor'), 'mhz_snapshot': r.get('cpu MHz')}
                for r in selected if r.get('processor', '').isdigit()]
    numa = []
    for node in sorted(Path('/sys/devices/system/node').glob('node[0-9]*')):
        cpus = cpu_list(read_text(node / 'cpulist'))
        numa.append({'node': node.name, 'cpus': cpus,
                     'allowed_cpus': sorted(set(cpus or []) & set(affinity)) if affinity is not None else None,
                     'distance': read_text(node / 'distance')})
    packages = {}
    for name in ('onnxruntime', 'onnxruntime-gpu', 'onnx', 'numpy', 'torch', 'scipy'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cgroups = collect_cgroups() if platform.system() == 'Linux' else None
    cache_topology = collect_cache_topology(affinity) if platform.system() == 'Linux' else None
    affinity_count = len(affinity) if affinity is not None else None
    quota = cgroups['tightest_visible_quota_cpu_equivalents'] if cgroups else None
    capacity = min(x for x in (affinity_count, quota) if x is not None) if any(
        x is not None for x in (affinity_count, quota)) else None
    notes = []
    for threads in requested_threads:
        if capacity is not None and threads > capacity:
            notes.append(f'{threads} threads exceed the visible affinity/quota upper bound of {capacity:g} CPU equivalents; report this constraint with timings.')
    if numa and sum(bool(n['allowed_cpus']) for n in numa) > 1:
        notes.append('Allowed CPUs span multiple NUMA nodes. Record any benchmark affinity/memory-binding choice; this diagnostic does not change it.')
    scheduler = {}
    for label, function in (
        ('policy', lambda: os.sched_getscheduler(0)),
        ('priority', lambda: os.sched_getparam(0).sched_priority),
        ('nice', lambda: os.getpriority(os.PRIO_PROCESS, 0)),
    ):
        try:
            scheduler[label] = function()
        except (AttributeError, OSError, NotImplementedError):
            scheduler[label] = None
    clock = time.get_clock_info('perf_counter')
    result = {
        'schema_version': 1, 'captured_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Read-only diagnostic; no model execution, GPU query, network request, affinity change, or timing benchmark.',
        'os': {'platform': platform.platform(), 'system': platform.system(), 'release': platform.release(),
               'machine': platform.machine(), 'python': sys.version, 'executable': sys.executable,
               'os_release': read_text('/etc/os-release')},
        'cpu': {'logical_cpus_visible_to_os': os.cpu_count(), 'process_affinity': affinity,
                'affinity_count': affinity_count, 'proc_status': status,
                'online': read_text('/sys/devices/system/cpu/online'),
                'models': sorted({r.get('model name') or r.get('Processor') or r.get('Hardware')
                                  for r in selected if r.get('model name') or r.get('Processor') or r.get('Hardware')}),
                'vendors': sorted({r['vendor_id'] for r in selected if 'vendor_id' in r}),
                'isa_intersection': sorted(set.intersection(*flags)) if flags else [],
                'isa_union': sorted(set.union(*flags)) if flags else [],
                'isa_note': 'OS-reported flags are inventory. A native dispatcher must still verify usable ISA/OS vector state before executing specialized kernels.',
                'allowed_topology': topology, 'numa_nodes': numa, 'cache_topology': cache_topology},
        'cgroups': cgroups, 'visible_cpu_capacity_upper_bound': capacity,
        'scheduler': scheduler, 'thread_environment': {k: os.environ[k] for k in THREAD_ENV if k in os.environ},
        'load_average_snapshot': load_average(),
        'cpu_pressure_snapshot': read_text('/proc/pressure/cpu'),
        'package_versions': packages, 'requested_benchmark_threads': list(requested_threads),
        'timer': {'implementation': clock.implementation, 'monotonic': clock.monotonic, 'resolution_s': clock.resolution},
        'notes': notes,
    }
    if platform.system() == 'Darwin':
        keys = ['machdep.cpu.brand_string', 'hw.model', 'hw.physicalcpu', 'hw.logicalcpu', 'hw.optional.neon']
        try:
            command = subprocess.run(['sysctl', *keys], capture_output=True, text=True, timeout=5)
            result['cpu']['darwin_sysctl'] = {'values': command.stdout.strip(), 'stderr': command.stderr.strip()}
        except (OSError, subprocess.TimeoutExpired) as exc:
            result['cpu']['darwin_sysctl'] = {'error': str(exc)}
    if ort_probe:
        try:
            import onnxruntime as ort
            session = cpu_session(PROBE, 1, expected_version=expected_ort_version)
            result['ort_cpu_probe'] = {'passed': True, 'version': ort.__version__,
                                       'build': ort.get_build_info(),
                                       'available_providers_inventory_only': ort.get_available_providers(),
                                       'selected_providers': session.get_providers(),
                                       'model_execution': False, 'scope': 'Tiny Identity session initialized only; run was not called.'}
        except Exception as exc:
            result['ort_cpu_probe'] = {'passed': False, 'error': repr(exc), 'model_execution': False}
    else:
        result['ort_cpu_probe'] = {'performed': False, 'note': 'Use --ort-probe to verify CPU-only session construction. No ORT module was imported by this script.'}
    return result


def self_test():
    assert cpu_list('0-3,8,10-11') == [0, 1, 2, 3, 8, 10, 11]
    assert cpu_list('') == [] and cpu_list(None) is None
    for invalid in ('3-1', 'x', '-2', '1,,2'):
        try:
            cpu_list(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(invalid)
    assert quota_from_files({'cpu.max': '150000 100000'}) == 1.5
    assert quota_from_files({'cpu.max': 'max 100000'}) is None
    assert quota_from_files({'cpu.cfs_quota_us': '-1', 'cpu.cfs_period_us': '100000'}) is None
    assert quota_from_files({'cpu.cfs_quota_us': '200000', 'cpu.cfs_period_us': '100000'}) == 2
    mount = {'root': '/container', 'mount': '/sys/fs/cgroup'}
    assert resolve_cgroup(mount, '/container/task')[0] == Path('/sys/fs/cgroup/task')
    assert resolve_cgroup(mount, '/')[0] == Path('/sys/fs/cgroup')
    assert resolve_cgroup(mount, '/../outside')[0] is None
    parsed = cgroup_mounts('10 1 0:10 / /sys/fs/cgroup rw - cgroup2 cgroup rw')
    assert parsed[0]['type'] == 'cgroup2'
    print('PASS: CPU-list, quota, cgroup mount and namespace parser checks; no model/ORT execution.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--ort-probe', action='store_true')
    parser.add_argument('--expected-ort-version', default='1.29.0')
    parser.add_argument('--threads', default='1,4', help='Planned benchmark thread counts; diagnostic changes no affinity.')
    parser.add_argument('--self-test', action='store_true', help='Run pure parser checks only.')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    threads = tuple(int(x) for x in args.threads.split(','))
    if not threads or any(t < 1 for t in threads):
        parser.error('--threads requires positive integers')
    result = collect(args.ort_probe, args.expected_ort_version, threads)
    encoded = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
        print(json.dumps({'output': str(args.output.resolve()), 'notes': result['notes'],
                          'ort_cpu_probe': result['ort_cpu_probe']}, indent=2))
    else:
        print(encoded, end='')
    if args.ort_probe and not result['ort_cpu_probe'].get('passed'):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
