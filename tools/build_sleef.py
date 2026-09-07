"""Fetch and build pinned CPU-only SLEEF 3.9.0 as a PIC static dependency.

Source, build and install trees remain ignored under .deps. Requires Python 3.12,
CMake and a native Linux x86_64 C/C++ toolchain. No models or timing.
"""
from __future__ import annotations
import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import tarfile
import urllib.request

_spec = importlib.util.spec_from_file_location('_native_build_x86', Path(__file__).with_name('build_x86.py'))
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)
ROOT = Path(__file__).resolve().parents[1]


def build(prefix=None, jobs=2):
    """Return prefix and dependency metadata; default prefix is .deps/sleef."""
    common.require_linux_x86()
    if type(jobs) is not int or jobs not in (1, 2):
        raise ValueError('Dependency build jobs must be 1 or 2')
    prefix = Path(prefix or ROOT/'.deps/sleef').resolve()
    stage = prefix.parent/(prefix.name+'-source-build')
    stage.mkdir(parents=True, exist_ok=True)
    with (stage/'.build.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        archive = stage/'sleef-3.9.0.tar.gz'
        if not archive.exists():
            temporary = archive.with_suffix('.download')
            urllib.request.urlretrieve(common.SLEEF['archive_url'], temporary)
            temporary.replace(archive)
        if common.sha256(archive) != common.SLEEF['archive_sha256']:
            raise RuntimeError('SLEEF source archive checksum mismatch')
        source, build_dir = stage/'sleef-3.9.0', stage/'build'
        if not source.exists():
            with tarfile.open(archive) as tar:
                tar.extractall(stage, filter='data')
        command = ['cmake', '-S', str(source), '-B', str(build_dir), '-G', 'Unix Makefiles',
                   '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_INSTALL_PREFIX='+str(prefix),
                   '-DBUILD_SHARED_LIBS=OFF', '-DCMAKE_POSITION_INDEPENDENT_CODE=ON',
                   '-DSLEEF_BUILD_DFT=OFF', '-DSLEEF_BUILD_QUAD=OFF', '-DSLEEF_BUILD_GNUABI_LIBS=OFF',
                   '-DSLEEF_BUILD_TESTS=OFF', '-DSLEEF_ENABLE_TESTER4=OFF', '-DSLEEF_ENABLE_TLFLOAT=OFF',
                   '-DSLEEF_DISABLE_OPENMP=ON', '-DSLEEF_ENABLE_CUDA=OFF', '-DSLEEF_BUILD_BENCH=OFF']
        commands = [command, ['cmake', '--build', str(build_dir), '--parallel', str(jobs)],
                    ['cmake', '--install', str(build_dir)]]
        logs = [common.run(command, cwd=stage) for command in commands]
        archives = [p for p in (prefix/'lib/libsleef.a', prefix/'lib64/libsleef.a') if p.is_file()]
        header = prefix/'include/sleef.h'
        if len(archives) != 1 or not header.is_file():
            raise RuntimeError('SLEEF static archive or header was not installed')
        record = {**common.SLEEF, 'prefix': str(prefix), 'library': str(archives[0]),
                  'header_sha256': common.sha256(header), 'library_sha256': common.sha256(archives[0]),
                  'commands': commands, 'logs': logs, 'gpu_hidden': True, 'openmp': False,
                  'pic': True, 'static': True, 'jobs': jobs}
        common.write_record(prefix/'dependency-build.json', record)
        return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix'); parser.add_argument('--jobs', type=int, choices=(1, 2), default=2)
    print(json.dumps(build(**vars(parser.parse_args())), indent=2))
