"""Fetch and build pinned serial AOCL BLAS as a CPU-only PIC static dependency.

Source, build and install trees remain ignored under .deps. This is the Zen4
LP64 dependency for the optional AMD adapter, without OpenMP or GPU code.
"""
from __future__ import annotations
import argparse
import fcntl
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location('_native_build_x86', Path(__file__).with_name('build_x86.py'))
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)
ROOT = Path(__file__).resolve().parents[1]
PIN = '25cad99a6840855ade0a49871197f48ee0e1d317'
REPOSITORY = 'https://github.com/amd/blis.git'


def build(prefix=None, jobs=2):
    """Return prefix and dependency metadata; default prefix is .deps/aocl-blas."""
    common.require_linux_x86()
    if type(jobs) is not int or jobs not in (1, 2):
        raise ValueError('Dependency build jobs must be 1 or 2')
    prefix = Path(prefix or ROOT/'.deps/aocl-blas').resolve()
    stage = prefix.parent/(prefix.name+'-source-build'); stage.mkdir(parents=True, exist_ok=True)
    with (stage/'.build.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        source = stage/'source'
        logs = []
        if not source.exists():
            logs.append(common.run(['git', 'init', str(source)], stage))
            logs.append(common.run(['git', 'remote', 'add', 'origin', REPOSITORY], source))
            logs.append(common.run(['git', 'fetch', '--depth', '1', 'origin', PIN], source))
            logs.append(common.run(['git', 'checkout', '--detach', 'FETCH_HEAD'], source))
        actual = common.run(['git', 'rev-parse', 'HEAD'], source)['stdout'].strip()
        if actual != PIN:
            raise RuntimeError('AOCL source commit mismatch')
        if common.run(['git', 'status', '--porcelain', '--untracked-files=no'], source)['stdout'].strip():
            raise RuntimeError('AOCL tracked sources have local modifications')
        commands = [
            ['./configure', '--prefix='+str(prefix), '--enable-static', '--disable-shared',
             '--disable-threading', '--enable-cblas', '--int-size=64', '--blas-int-size=32', 'zen4'],
            ['make', '-j'+str(jobs)], ['make', 'install'],
        ]
        logs.extend(common.run(command, source) for command in commands)
        header, library = prefix/'include/blis/blis.h', prefix/'lib/libblis.a'
        if not header.is_file() or not library.is_file():
            raise RuntimeError('Serial BLIS static archive or header was not installed')
        record = {'repository': REPOSITORY, 'pin': PIN, 'version': '5.3.2', 'prefix': str(prefix),
                  'library': str(library), 'header_sha256': common.sha256(header),
                  'library_sha256': common.sha256(library), 'threading': 'no', 'target': 'zen4',
                  'commands': commands, 'logs': logs, 'jobs': jobs, 'gpu_hidden': True,
                  'blas_integer_bits': 32, 'internal_integer_bits': 64, 'pic': True, 'static': True}
        common.write_record(prefix/'dependency-build.json', record)
        return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix'); parser.add_argument('--jobs', type=int, choices=(1, 2), default=2)
    print(json.dumps(build(**vars(parser.parse_args())), indent=2))
