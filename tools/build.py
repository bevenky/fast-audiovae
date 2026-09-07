"""Explicit CPU-only native build, including missing pinned dependencies.

Apple ARM uses Accelerate. Linux x86 uses SLEEF; --amd-packed adds the optional
serial AOCL adapter on compatible AMD hosts. Nothing runs on module import.
Use --offline to require existing verified dependencies and avoid downloads.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[1]
# This is a git commit, independent of the release tag used for the archive.
SLEEF_PIN = '906ca7512ee483296780a81a21b9ca715d40dfe1'
AOCL_PIN = '25cad99a6840855ade0a49871197f48ee0e1d317'
AMD_FLAGS = frozenset(('avx', 'avx2', 'fma', 'xsave', 'avx512f', 'avx512dq', 'avx512bw', 'avx512vl'))


def _module(name):
    spec = importlib.util.spec_from_file_location('_native_'+name, Path(__file__).with_name(name+'.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _amd_cpu_check():
    """Pre-screen Linux CPU flags; the built adapter also probes CPUID/XCR0."""
    info = Path('/proc/cpuinfo').read_text()
    allowed = os.sched_getaffinity(0) if hasattr(os, 'sched_getaffinity') else None
    cpus = []
    for block in info.strip().split('\n\n'):
        fields = {k.strip(): v.strip() for line in block.splitlines() if ':' in line
                  for k, v in [line.split(':', 1)]}
        if 'processor' not in fields:
            continue
        index = int(fields['processor'])
        if allowed is None or index in allowed:
            cpus.append(fields)
    if not cpus or (allowed is not None and {int(c['processor']) for c in cpus} != set(allowed)):
        raise RuntimeError('Could not verify every permitted Linux CPU for --amd-packed')
    for cpu in cpus:
        if cpu.get('vendor_id') != 'AuthenticAMD' or not AMD_FLAGS.issubset(cpu.get('flags', '').split()):
            raise RuntimeError('--amd-packed requires an AMD CPU with AVX2/FMA and AVX512F/DQ/BW/VL enabled')
    return {'vendor': 'AuthenticAMD', 'verified_cpus': len(cpus),
            'precheck': 'Linux CPU flags; the native adapter additionally verifies CPUID and OS register state'}


def _dependency_ready(kind, prefix):
    """Reuse only a complete dependency with matching recorded binary hashes."""
    record_path = prefix/'dependency-build.json'
    if not record_path.is_file():
        return False
    record = json.loads(record_path.read_text())
    if kind == 'sleef':
        if record.get('commit') != SLEEF_PIN or record.get('tag') != '3.9.0' or record.get('openmp') is not False:
            raise RuntimeError('SLEEF dependency record does not match the pinned CPU-only build')
        header = prefix/'include/sleef.h'
        archives = [p for p in (prefix/'lib/libsleef.a', prefix/'lib64/libsleef.a') if p.is_file()]
        if len(archives) != 1:
            return False
        library = archives[0]
    else:
        if record.get('pin') != AOCL_PIN or record.get('threading') != 'no' or record.get('target') != 'zen4':
            raise RuntimeError('AOCL dependency record does not match the pinned serial Zen4 build')
        header, library = prefix/'include/blis/blis.h', prefix/'lib/libblis.a'
    if not header.is_file() or not library.is_file():
        return False
    for key, path in (('header_sha256', header), ('library_sha256', library)):
        if not isinstance(record.get(key), str) or record[key] != _sha256(path):
            raise RuntimeError(kind.upper()+' dependency artifact hash mismatch: '+key)
    return True


def _ensure_dependency(kind, offline, jobs):
    prefix = ROOT/'.deps'/('sleef' if kind == 'sleef' else 'aocl-blas')
    if not _dependency_ready(kind, prefix):
        if offline:
            raise RuntimeError('Missing verified '+kind.upper()+' dependency. Run tools/build.py without --offline to build it.')
        _module('build_sleef' if kind == 'sleef' else 'build_aocl').build(prefix=prefix, jobs=jobs)
        if not _dependency_ready(kind, prefix):
            raise RuntimeError('Dependency build did not produce verified '+kind.upper()+' artifacts')
    return {'prefix': str(prefix), 'manifest': str(prefix/'dependency-build.json')}


def _library_summary(record):
    keys = ('library', 'library_sha256', 'domain', 'native_abi', 'ort_api_version', 'build_manifest', 'cache_hit')
    answer = {key: record[key] for key in keys}
    if answer['native_abi'] != 1 or answer['ort_api_version'] != 29:
        raise RuntimeError('Unexpected native or ONNX Runtime API version')
    path = Path(answer['library'])
    if not path.is_file() or _sha256(path) != answer['library_sha256']:
        raise RuntimeError('Built native library hash mismatch')
    return answer


def build(amd_packed=False, offline=False, jobs=2):
    """Build for this host and return headers, dependency and library summaries."""
    if type(jobs) is not int or jobs not in (1, 2):
        raise ValueError('Dependency build jobs must be 1 or 2')
    system, machine = platform.system(), platform.machine().lower()
    apple = system == 'Darwin' and machine in ('arm64', 'aarch64')
    x86 = system == 'Linux' and machine in ('x86_64', 'amd64')
    if not apple and not x86:
        raise RuntimeError('Supported builds are native Apple ARM macOS and Linux x86_64')
    if amd_packed and not x86:
        raise RuntimeError('--amd-packed requires Linux x86_64 with a compatible AMD CPU')
    cpu = _amd_cpu_check() if amd_packed else None
    headers = _module('fetch_headers').fetch(offline=offline)
    dependencies, libraries = {}, {}
    if apple:
        libraries['apple'] = _library_summary(_module('build_apple').build())
    else:
        dependencies['sleef'] = _ensure_dependency('sleef', offline, jobs)
        libraries['x86'] = _library_summary(_module('build_x86').build())
        if amd_packed:
            dependencies['aocl'] = _ensure_dependency('aocl', offline, jobs)
            libraries['amd'] = _library_summary(_module('build_amd').build())
    result = {'platform': {'system': system, 'machine': machine}, 'headers': headers,
              'dependencies': dependencies, 'libraries': libraries, 'amd_cpu': cpu,
              'cpu_only': True, 'offline': offline, 'scope': 'Native libraries only; no model execution or timing',
              'source_sha256': {name: _sha256(ROOT/'tools'/name) for name in ('build.py', 'fetch_headers.py')}}
    out = ROOT/'.build'; out.mkdir(parents=True, exist_ok=True)
    manifest = out/'build.json'; temporary = manifest.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2)+'\n'); temporary.replace(manifest)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--amd-packed', action='store_true', help='Also build the optional serial AMD packed-row adapter')
    parser.add_argument('--offline', action='store_true', help='Reuse verified local dependencies only; never download')
    parser.add_argument('--jobs', type=int, choices=(1, 2), default=2, help='Parallel jobs for missing native dependencies (default: 2)')
    print(json.dumps(build(**vars(parser.parse_args())), indent=2))
