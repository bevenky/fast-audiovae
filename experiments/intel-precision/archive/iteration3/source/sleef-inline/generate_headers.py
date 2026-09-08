"""Generate inline CPU headers from the pinned, already installed SLEEF source archive."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

ROOT = Path('/dev/shm/fast-audiovae-intel-iteration3/sleef-inline-generation-r1')
ARCHIVE = Path('/var/tmp/fast-audiovae-20260907/repo/.deps/sleef-source-build/sleef-3.9.0.tar.gz')
EXPECTED = 'af60856abac08a3b5e72a8d156dd71fec1f7ac23de8ee67793f45f9edcdf0908'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if ROOT.exists() or sha(ARCHIVE) != EXPECTED:
        raise ValueError('Fresh output and exact pinned archive required')
    ROOT.mkdir()
    with tarfile.open(ARCHIVE) as archive:
        archive.extractall(ROOT, filter='data')
    source = ROOT/'sleef-3.9.0'
    before = {str(p.relative_to(source)): sha(p) for p in source.rglob('*') if p.is_file()}
    build = ROOT/'build'
    commands = [
        ['cmake', '-S', str(source), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
         '-DBUILD_SHARED_LIBS=OFF', '-DCMAKE_POSITION_INDEPENDENT_CODE=ON',
         '-DSLEEF_BUILD_INLINE_HEADERS=ON', '-DSLEEF_ENABLE_LTO=OFF',
         '-DSLEEF_BUILD_DFT=OFF', '-DSLEEF_BUILD_QUAD=OFF', '-DSLEEF_BUILD_GNUABI_LIBS=OFF',
         '-DSLEEF_BUILD_TESTS=OFF', '-DSLEEF_ENABLE_TESTER4=OFF', '-DSLEEF_ENABLE_TLFLOAT=OFF',
         '-DSLEEF_DISABLE_OPENMP=ON', '-DSLEEF_ENABLE_CUDA=OFF', '-DSLEEF_BUILD_BENCH=OFF'],
        ['cmake', '--build', str(build), '--target', 'inline_headers_util', '--parallel', '2'],
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', NVIDIA_VISIBLE_DEVICES='void',
               HIP_VISIBLE_DEVICES='-1', ROCR_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1')
    for key in ('LD_PRELOAD', 'CFLAGS', 'CXXFLAGS', 'LDFLAGS'):
        env.pop(key, None)
    with (ROOT/'generation.log').open('w') as log:
        for command in commands:
            subprocess.run(command, check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
    after = {str(p.relative_to(source)): sha(p) for p in source.rglob('*') if p.is_file()}
    if before != after or sha(ARCHIVE) != EXPECTED:
        raise ValueError('Input mutated during generation')
    header = build/'include/sleefinline_avx512f.h'
    record = {'status': 'generated', 'gpu_used': False, 'models_executed': False,
              'source_archive': str(ARCHIVE), 'source_archive_sha256': EXPECTED,
              'source_file_sha256': before, 'commands': commands,
              'compiler': subprocess.check_output(['cc', '--version'], text=True).splitlines()[0],
              'header': str(header), 'header_sha256': sha(header),
              'script_sha256': sha(Path(__file__).resolve()), 'log_sha256': sha(ROOT/'generation.log')}
    (ROOT/'generation.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({key: record[key] for key in ('status', 'header', 'header_sha256')}))


if __name__ == '__main__':
    main()
