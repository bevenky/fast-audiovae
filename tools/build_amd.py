"""Build the optional serial AOCL packed-row operator for supported AMD CPUs.

Defaults: .deps/aocl-blas, .deps/onnxruntime/include, .build/amd.
The generic x86 library remains separate. This script runs no models.
"""
from __future__ import annotations
import argparse
import ctypes
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location('_native_build_x86', Path(__file__).with_name('build_x86.py'))
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)
ROOT = Path(__file__).resolve().parents[1]
PIN = '25cad99a6840855ade0a49871197f48ee0e1d317'
DOMAIN = 'venky.audio.cpu.aocl.rows'


def _probe(library):
    handle = ctypes.CDLL(str(library))
    handle.ncc_aocl_cpu_supported.argtypes = []; handle.ncc_aocl_cpu_supported.restype = ctypes.c_int
    handle.ncc_aocl_adapter_abi.argtypes = []; handle.ncc_aocl_adapter_abi.restype = ctypes.c_uint32
    supported, abi = int(handle.ncc_aocl_cpu_supported()), int(handle.ncc_aocl_adapter_abi())
    if not supported or abi != 1:
        raise RuntimeError('AMD vendor, ISA/OS support or packed adapter ABI probe failed')
    return {'cpu_supported': supported, 'native_abi': abi}


def build(prefix=None, ort_include=None, dependency_record=None, cc=None, cxx=None):
    """Return library, domain, native_abi and build_manifest; no models."""
    common.require_linux_x86()
    prefix = Path(prefix or ROOT/'.deps/aocl-blas').resolve()
    include = Path(ort_include or ROOT/'.deps/onnxruntime/include').resolve()
    common.require_headers(include)
    header, archive = prefix/'include/blis/blis.h', prefix/'lib/libblis.a'
    if not header.is_file() or not archive.is_file():
        raise RuntimeError('Expected include/blis/blis.h and serial PIC lib/libblis.a')
    metadata = Path(dependency_record) if dependency_record else prefix/'dependency-build.json'
    dep = json.loads(metadata.read_text())
    if dep.get('pin') != PIN or dep.get('threading') != 'no' or dep.get('target') != 'zen4':
        raise RuntimeError('Pinned serial Zen4 dependency metadata is required')
    for key, artifact in (('header_sha256', header), ('library_sha256', archive)):
        if key in dep and dep[key] != common.sha256(artifact):
            raise RuntimeError('AOCL dependency metadata hash mismatch: ' + key)
    cc, cxx, versions = common.compilers(cc, cxx)
    names = ['native/amd/packed_a.c', 'native/amd/packed_a.h', 'native/amd/custom_op.cpp',
             'tools/build_amd.py', 'tools/build_x86.py']
    fingerprint = {'source_sha256': {p: common.sha256(ROOT/p) for p in names},
                   'header_sha256': common.HEADERS, 'blis_header_sha256': common.sha256(header),
                   'blis_library_sha256': common.sha256(archive), 'dependency': dep,
                   'compilers': versions, 'common_flags': common.COMMON_FLAGS,
                   'domain': DOMAIN, 'native_abi': 1, 'ort_api_version': 29, 'openmp': False}
    build_id = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    out = ROOT/'.build/amd'; out.mkdir(parents=True, exist_ok=True)
    with (out/'.build.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        library = out/('libfast_audiovae_amd_'+build_id[:12]+'.so')
        manifest = out/'build.json'
        old = json.loads(manifest.read_text()) if manifest.exists() else {}
        if (old.get('fingerprint') == fingerprint and library.is_file()
                and old.get('library_sha256') == common.sha256(library)):
            _probe(library)
            return {**old, 'library': str(library), 'build_manifest': str(manifest), 'cache_hit': True}
        a, b = out/'packed.o', out/'custom.o'
        commands = [
            [cc, *common.COMMON_FLAGS, '-std=c11', '-D_POSIX_C_SOURCE=200112L',
             '-I'+str(header.parent), '-Inative/amd', '-c', 'native/amd/packed_a.c', '-o', str(a)],
            [cxx, *common.COMMON_FLAGS, '-std=c++17', '-I'+str(include), '-Inative/amd',
             '-c', 'native/amd/custom_op.cpp', '-o', str(b)],
            [cxx, '-shared', '-pthread', str(a), str(b), str(archive), '-lm',
             '-Wl,-Bsymbolic', '-Wl,--exclude-libs,ALL', '-Wl,-z,defs', '-Wl,-soname,'+library.name, '-o', str(library)],
        ]
        logs = [common.run(command) for command in commands]
        deps = common.run(['readelf', '-d', str(library)])['stdout']
        if any(name in deps for name in ('libgomp', 'libomp', 'libblis')):
            raise RuntimeError('Unexpected dynamic BLAS or OpenMP dependency')
        for name, expected in fingerprint['source_sha256'].items():
            if common.sha256(ROOT/name) != expected:
                raise RuntimeError('Source changed during build: ' + name)
        common.require_headers(include)
        if (common.sha256(header) != fingerprint['blis_header_sha256']
                or common.sha256(archive) != fingerprint['blis_library_sha256']):
            raise RuntimeError('AOCL dependency changed during build')
        result = {'library': str(library), 'library_sha256': common.sha256(library),
                  'build_manifest': str(manifest), 'fingerprint': fingerprint, 'build_id': build_id,
                  'commands': commands, 'logs': logs, 'dependencies': deps, 'aocl_pin': PIN,
                  'domain': DOMAIN, 'native_abi': 1, 'ort_api_version': 29,
                  'openmp': False, 'static_blis': True, 'cache_hit': False,
                  'operators': ['PackedRowsMatMulF32'], 'probe': _probe(library),
                  'scope': 'CPU-only library build and metadata probe; no model execution or timing'}
        common.write_record(manifest, result)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix'); parser.add_argument('--ort-include'); parser.add_argument('--dependency-record')
    parser.add_argument('--cc'); parser.add_argument('--cxx')
    print(json.dumps(build(**vars(parser.parse_args())), indent=2))
