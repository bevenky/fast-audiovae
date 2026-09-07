"""Build the Linux x86 FP32 operators with static SLEEF and pinned ORT headers.

Defaults: .deps/sleef, .deps/onnxruntime/include, .build/x86.
No model execution or timing. Build records contain absolute output paths.
"""
from __future__ import annotations
import argparse
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DOMAIN = 'venky.audio.cpu.portable'
SLEEF = {'tag': '3.9.0', 'commit': '906ca7512ee483296780a81a21b9ca715d40dfe1',
         'archive_url': 'https://codeload.github.com/shibatch/sleef/tar.gz/refs/tags/3.9.0',
         'archive_sha256': 'af60856abac08a3b5e72a8d156dd71fec1f7ac23de8ee67793f45f9edcdf0908'}
HEADERS = {
    'onnxruntime_c_api.h': 'acc0cf4b3f28d39339c76770d76164bb7a0637dc89f5fde764b4017b632f6743',
    'onnxruntime_cxx_api.h': '9c63ed5bf0427ddf1c10a99aca0beeec77318ce77d2ea6f7ae76d7d933660f67',
    'onnxruntime_cxx_inline.h': 'c508dbb7d8203003a2584b8712118cb88e5b4ef1e1c7876a9974156fb7489301',
    'onnxruntime_ep_c_api.h': 'e6c986c9e98583f8113b2c6bc3864814883b806d501cf24da4d239c45753e235',
    'onnxruntime_error_code.h': '5ce3b054e798eced8d14f5b86e98692fd33470463f96194ce0700a2d53dd8721',
    'onnxruntime_float16.h': '88b242845d25981633a0bbd1c148e273cf8bfb016ea3f57c4af41a06530f72b0',
}
COMMON_FLAGS = ['-O3', '-fPIC', '-fno-fast-math', '-ffp-contract=off', '-fvisibility=hidden',
                '-Wall', '-Wextra', '-march=x86-64', '-mtune=generic']
SLEEF_SYMBOLS = ('Sleef_sinf4_u10sse2', 'Sleef_sinf8_u10avx2', 'Sleef_sinf16_u10avx512f')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu_environment():
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES='-1', NVIDIA_VISIBLE_DEVICES='void',
               ROCR_VISIBLE_DEVICES='-1', HIP_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1',
               BLIS_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', OMP_WAIT_POLICY='PASSIVE',
               KMP_BLOCKTIME='0', GOMP_SPINCOUNT='0')
    return env


def run(command, cwd=None):
    result = subprocess.run(command, cwd=cwd or ROOT, env=cpu_environment(), capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('Command failed: ' + repr(command) + '\n' + result.stdout + result.stderr)
    return {'command': command, 'stdout': result.stdout, 'stderr': result.stderr}


def require_linux_x86():
    if platform.system() != 'Linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
        raise RuntimeError('This builder requires native Linux x86_64')


def require_headers(include):
    for name, expected in HEADERS.items():
        if not (include/name).is_file() or sha256(include/name) != expected:
            raise RuntimeError('Missing or mismatched pinned ORT 1.29 header: ' + name)


def compilers(cc=None, cxx=None):
    cc = cc or os.environ.get('CC') or shutil.which('gcc')
    cxx = cxx or os.environ.get('CXX') or shutil.which('g++')
    if not cc or not cxx:
        raise RuntimeError('GCC-compatible C11 and C++17 compilers are required')
    return cc, cxx, {'cc': run([cc, '--version'])['stdout'].splitlines()[0],
                     'cxx': run([cxx, '--version'])['stdout'].splitlines()[0]}


def write_record(path, record):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, indent=2)+'\n')
    temporary.replace(path)


def probe(record):
    lib = ctypes.CDLL(record['library'])
    answer = {}
    for name in ('ncc_compiled_tile', 'ncc_abi_version', 'ncc_selected_backend', 'ncc_phase_finish_abi'):
        fn = getattr(lib, name); fn.argtypes = []; fn.restype = ctypes.c_uint32
        answer[name] = int(fn())
    lib.ncc_capabilities.argtypes = []; lib.ncc_capabilities.restype = ctypes.c_uint64
    answer['capabilities'] = int(lib.ncc_capabilities())
    lib.ncc_snake_math_name.argtypes = [ctypes.c_int32]; lib.ncc_snake_math_name.restype = ctypes.c_char_p
    answer['sine_math'] = lib.ncc_snake_math_name(0).decode()
    lib.ncc_backend_available.argtypes = [ctypes.c_int32]
    lib.ncc_backend_available.restype = ctypes.c_int32
    answer['explicit_avx512_available'] = bool(lib.ncc_backend_available(5))
    answer['explicit_avx512_math'] = lib.ncc_snake_math_name(5).decode()
    lib.ncc_portable_build_id.argtypes = []; lib.ncc_portable_build_id.restype = ctypes.c_char_p
    answer['build_id'] = lib.ncc_portable_build_id().decode()
    if (answer['ncc_compiled_tile'] != 256 or answer['ncc_abi_version'] != 1
            or answer['ncc_phase_finish_abi'] != 1 or answer['build_id'] != record['build_id']
            or not answer['capabilities'] & 64 or answer['capabilities'] & 16
            or answer['ncc_selected_backend'] == 5):
        raise RuntimeError('Native ABI, tile, SLEEF or OpenMP metadata probe failed')
    return answer


def build(sleef_prefix=None, ort_include=None, cc=None, cxx=None, sleef_archive=None):
    """Return library, domain, native_abi, hashes and build_manifest; no models."""
    require_linux_x86()
    prefix = Path(sleef_prefix or ROOT/'.deps/sleef').resolve()
    include = Path(ort_include or ROOT/'.deps/onnxruntime/include').resolve()
    require_headers(include)
    header = prefix/'include/sleef.h'
    archives = [p for p in (prefix/'lib/libsleef.a', prefix/'lib64/libsleef.a') if p.is_file()]
    if not header.is_file() or len(archives) != 1:
        raise RuntimeError('Expected include/sleef.h and one PIC static lib/libsleef.a or lib64/libsleef.a')
    archive = archives[0]
    nm = shutil.which('nm')
    if not nm:
        raise RuntimeError('nm is required to verify the pinned SLEEF vector symbols')
    symbols = run([nm, '-g', '--defined-only', str(archive)])['stdout'].split()
    missing = [name for name in SLEEF_SYMBOLS if name not in symbols]
    if missing:
        raise RuntimeError('SLEEF static archive lacks required CPU vector symbols: ' + ', '.join(missing))
    if sleef_archive and sha256(sleef_archive) != SLEEF['archive_sha256']:
        raise RuntimeError('SLEEF source archive checksum mismatch')
    for line in ('#define SLEEF_VERSION_MAJOR 3', '#define SLEEF_VERSION_MINOR 9', '#define SLEEF_VERSION_PATCHLEVEL 0'):
        if line not in header.read_text():
            raise RuntimeError('SLEEF header version mismatch: ' + line)
    cc, cxx, versions = compilers(cc, cxx)
    sources = ['native/x86/native_kernels.c', 'native/x86/native_kernels.h',
               'native/x86/custom_ops.cpp', 'tools/build_x86.py']
    novec = ['-fno-vectorize', '-fno-slp-vectorize'] if 'clang' in versions['cc'].lower() else ['-fno-tree-vectorize']
    fingerprint = {'source_sha256': {p: sha256(ROOT/p) for p in sources},
                   'header_sha256': HEADERS, 'sleef': SLEEF,
                   'sleef_header_sha256': sha256(header), 'sleef_library_sha256': sha256(archive),
                   'compilers': versions, 'domain': DOMAIN, 'native_abi': 1, 'ort_api_version': 29,
                   'common_flags': COMMON_FLAGS, 'c_novec_flags': novec, 'openmp': False, 'tile': 256,
                   'required_sleef_symbols': SLEEF_SYMBOLS, 'explicit_avx512_backend': 5,
                   'auto_backend_policy': 'AVX2 then SSE2; AVX512 remains opt-in'}
    build_id = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    out = ROOT/'.build/x86'; out.mkdir(parents=True, exist_ok=True)
    with (out/'.build.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        library = out/('libfast_audiovae_x86_'+build_id[:12]+'.so')
        manifest = out/'build.json'
        old = json.loads(manifest.read_text()) if manifest.exists() else {}
        if (old.get('fingerprint') == fingerprint and library.is_file()
                and old.get('library_sha256') == sha256(library)):
            result = {**old, 'library': str(library), 'build_manifest': str(manifest), 'cache_hit': True}
            probe(result)
            return result
        wrapper = out/'native_wrapper.c'
        wrapper.write_text('#include "native_kernels.c"\n'
            f'NCC_API const char *ncc_portable_build_id(void) {{ return "{build_id}"; }}\n'
            'NCC_API uint32_t ncc_phase_finish_abi(void) { return 1; }\n')
        c_obj, cpp_obj = out/'native.o', out/'bridge.o'
        commands = [
            [cc, *COMMON_FLAGS, *novec, '-std=c11', '-DNCC_USE_SLEEF=1', '-I'+str(prefix/'include'),
             '-Inative/x86', '-c', str(wrapper), '-o', str(c_obj)],
            [cxx, *COMMON_FLAGS, '-std=c++17', '-I'+str(include), '-Inative/x86',
             '-c', 'native/x86/custom_ops.cpp', '-o', str(cpp_obj)],
            [cxx, '-shared', '-pthread', str(c_obj), str(cpp_obj), str(archive), '-lm',
             '-Wl,-Bsymbolic', '-Wl,--exclude-libs,ALL', '-Wl,-z,defs', '-Wl,-soname,'+library.name, '-o', str(library)],
        ]
        logs = [run(command) for command in commands]
        for name, expected in fingerprint['source_sha256'].items():
            if sha256(ROOT/name) != expected:
                raise RuntimeError('Source changed during build: ' + name)
        require_headers(include)
        if sha256(header) != fingerprint['sleef_header_sha256'] or sha256(archive) != fingerprint['sleef_library_sha256']:
            raise RuntimeError('SLEEF dependency changed during build')
        result = {'build_id': build_id, 'library': str(library), 'library_sha256': sha256(library),
                  'build_manifest': str(manifest), 'fingerprint': fingerprint, 'commands': commands, 'logs': logs,
                  'domain': DOMAIN, 'native_abi': 1, 'ort_api_version': 29, 'tile': 256, 'openmp': False,
                  'sine': 'SLEEF u10; guarded explicit AVX512F, AVX2+FMA, SSE2 fallback',
                  'sleef_source_archive_verified': bool(sleef_archive), 'cache_hit': False,
                  'operators': ['SnakeF32', 'CausalDW7F32', 'CausalDW7SnakeF32',
                                'SnakeDW7SnakeF32', 'BiasResidualF32',
                                'PhaseSumBiasInterleaveF32', 'CombinedPhaseSumBiasInterleaveF32'],
                  'scope': 'CPU-only library build and metadata probe; no model execution or timing'}
        result['probe'] = probe(result)
        write_record(manifest, result)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sleef-prefix'); parser.add_argument('--ort-include'); parser.add_argument('--sleef-archive')
    parser.add_argument('--cc'); parser.add_argument('--cxx')
    print(json.dumps(build(**vars(parser.parse_args())), indent=2))
