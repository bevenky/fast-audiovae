"""Finish the isolated environment after the scheduled model checks stop.

Only cuda-pathfinder changes. The original lock and all model-check provenance
remain untouched; final package/import evidence is written separately.
"""
from pathlib import Path
import hashlib
import importlib.metadata as metadata
import json
import os
import subprocess
import sys

BASE = Path('/tmp/fast-audiovae-recovery-20260909')
VENV = BASE / 'venv214'
INITIAL = BASE / 'venv214-evidence'
OUT = BASE / 'venv214-final-evidence'


def packages():
    return {d.metadata['Name'].lower().replace('_', '-'): d.version
            for d in metadata.distributions()}


def native_stats():
    root = next((VENV / 'lib').glob('python*/site-packages'))
    paths = list((root / 'torch/lib').glob('*')) + list((root / 'nvidia').rglob('*.so*'))
    return {str(p.relative_to(root)): (p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns)
            for p in paths if p.is_file()}


def main():
    if Path(sys.prefix).resolve() != VENV.resolve():
        raise RuntimeError('Run inside the isolated venv214')
    if OUT.exists():
        raise RuntimeError('Refusing to replace final environment evidence')
    OUT.mkdir()
    before = packages()
    if before['cuda-pathfinder'] != '1.6.0':
        raise RuntimeError('Unexpected pre-update helper version')
    native_before = native_stats()
    env = dict(os.environ)
    for name in ('PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH', 'LD_PRELOAD',
                 'CUDA_HOME', 'CUDA_PATH', 'VIRTUAL_ENV'):
        env.pop(name, None)
    env.update(PYTHONNOUSERSITE='1', CUDA_VISIBLE_DEVICES='',
               PIP_CONFIG_FILE='/dev/null', PIP_NO_CACHE_DIR='1',
               PIP_DISABLE_PIP_VERSION_CHECK='1')
    commands = []
    def run(name, args, allowed=0):
        with (OUT / (name + '.log')).open('w') as log:
            result = subprocess.run(args, cwd=BASE, env=env, stdout=log, stderr=subprocess.STDOUT)
        commands.append({'name': name, 'argv': args, 'exit_code': result.returncode})
        (OUT / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        if result.returncode != allowed:
            raise RuntimeError(f'{name} failed: {(OUT / (name + ".log")).read_text()[-4000:]}')
    pip = [sys.executable, '-m', 'pip']
    run('pathfinder-update', pip + ['install', '--upgrade', '--only-binary=:all:',
        '--index-url', 'https://pypi.org/simple', '--report', str(OUT / 'pathfinder-report.json'),
        'cuda-pathfinder==1.8.1'])
    # Refresh the metadata finder after pip replaces a distribution in-process.
    metadata.MetadataPathFinder.invalidate_caches()
    after = packages()
    differences = {name: [before.get(name), after.get(name)]
                   for name in before.keys() | after.keys() if before.get(name) != after.get(name)}
    if differences != {'cuda-pathfinder': ['1.6.0', '1.8.1']}:
        raise RuntimeError(f'Unexpected package changes: {differences}')
    if native_stats() != native_before:
        raise RuntimeError('Native library files changed during the Python helper update')
    run('pip-check', pip + ['check'], allowed=1)
    expected = ('torch 2.14.0+cu126 has requirement nvidia-cudnn-cu12==9.10.2.21; '
                'platform_system == "Linux", but you have nvidia-cudnn-cu12 9.25.1.1.')
    if (OUT / 'pip-check.log').read_text().strip() != expected:
        raise RuntimeError('Package check differs from the single declared cuDNN exception')
    run('freeze', pip + ['freeze', '--all'])
    (OUT / 'requirements.lock').write_bytes((OUT / 'freeze.log').read_bytes())
    run('inspect', pip + ['inspect'])
    # Repeat the same import-only contract, with the new helper version checked.
    old_commands = json.loads((INITIAL / 'commands.json').read_text())
    command = next(row['argv'] for row in old_commands if row['name'] == 'import-only')
    command = command[:-1] + [command[-1] + '\nassert m.version("cuda-pathfinder") == "1.8.1"\n']
    run('import-only', command)
    result = {'status': 'final_import_and_package_checks_passed',
        'package_changes': differences, 'native_library_stat_records_unchanged': len(native_before),
        'gpu_api_or_model_calls': 0, 'declared_pin_exceptions': 1,
        'initial_requirements_sha256': hashlib.sha256((INITIAL / 'requirements.lock').read_bytes()).hexdigest(),
        'final_requirements_sha256': hashlib.sha256((OUT / 'requirements.lock').read_bytes()).hexdigest(),
        'model_checks_note': 'Scheduled head and runtime measurements preceding this finalization used cuda-pathfinder1.6.0. The only subsequent change was this Python helper; no native libraries changed.'}
    (OUT / 'complete.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
