"""Create an isolated latest-compatible package environment, without GPU work."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

BASE = Path('/tmp/fast-audiovae-recovery-20260909')
ORIGINAL = Path('/workspace/fast-audiovae-convnext-20260908-r1/.train-venv')
VENV = BASE / 'venv214'
OUT = BASE / 'venv214-evidence'
OVERRIDE = '9.25.1.1'
DIRECT = ['numpy', 'tensorboard', 'pyarrow', 'huggingface-hub', 'soundfile',
          'soxr', 'pydantic', 'pytest', 'onnx==1.22.0', 'onnxruntime==1.29.0',
          'packaging', 'PyYAML', 'tqdm', 'pillow']


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def main():
    if Path(sys.prefix).resolve() != ORIGINAL.resolve():
        raise RuntimeError('Run with the original training interpreter')
    if VENV.exists():
        raise RuntimeError('Refusing to overwrite an existing venv214')
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for name in ('PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH', 'LD_PRELOAD',
                 'CUDA_HOME', 'CUDA_PATH', 'VIRTUAL_ENV'):
        env.pop(name, None)
    env.update(PYTHONNOUSERSITE='1', CUDA_VISIBLE_DEVICES='',
               PIP_CONFIG_FILE='/dev/null', PIP_NO_CACHE_DIR='1',
               PIP_DISABLE_PIP_VERSION_CHECK='1')
    commands = []

    def run(name, args, *, check=True):
        start = time.time()
        print(name, flush=True)
        with (OUT / (name + '.log')).open('w') as log:
            result = subprocess.run(args, env=env, cwd=BASE, stdout=log, stderr=subprocess.STDOUT)
        commands.append({'name': name, 'argv': args, 'exit_code': result.returncode,
                         'wall_seconds': time.time() - start})
        save('commands.json', commands)
        if check and result.returncode:
            print((OUT / (name + '.log')).read_text()[-6000:], flush=True)
            raise RuntimeError(f'{name} exited {result.returncode}')
        return result.returncode

    old_site = next((ORIGINAL / 'lib').glob('python*/site-packages'))
    def original_metadata():
        return {str(p.relative_to(old_site)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in old_site.glob('*.dist-info/METADATA')}
    old_metadata = original_metadata()
    save('original-environment-before.json', old_metadata)
    # These are only package download, unpacked-wheel and build caches. The
    # preceding read-only inventory found no symlinks to them from either the
    # original venv or existing isolated recovery packages. Installed hardlinks
    # remain valid when the cache's link is removed.
    targets = [Path('/root/.cache/pip/http-v2'),
               Path('/root/.cache/uv/archive-v0'),
               Path('/root/.cache/uv/sdists-v9'),
               Path('/root/.cache/uv/builds-v0')]
    cleanup = {'free_bytes_before': shutil.disk_usage(BASE).free, 'removed': []}
    for target in targets:
        if target.exists():
            if target.is_symlink():
                raise RuntimeError(f'Refusing a symlinked cache: {target}')
            shutil.rmtree(target)
            cleanup['removed'].append(str(target))
    cleanup['free_bytes_after'] = shutil.disk_usage(BASE).free
    save('download-cache-cleanup.json', cleanup)
    if cleanup['free_bytes_after'] < 9 * 1024**3:
        raise RuntimeError('Need at least9GiB free before the new installation')
    run('create', [sys.executable, '-m', 'venv', str(VENV)])
    python = str(VENV / 'bin/python')
    pip = [python, '-m', 'pip']
    run('packaging', pip + ['install', '--upgrade', '--only-binary=:all:',
        '--index-url', 'https://pypi.org/simple', '--report', str(OUT / 'packaging-report.json'),
        'pip', 'setuptools', 'wheel'])
    run('torch-official', pip + ['install', '--only-binary=:all:',
        '--index-url', 'https://download.pytorch.org/whl/cu126',
        '--report', str(OUT / 'torch-official-report.json'), 'torch==2.14.0+cu126'])
    # Freeze the chosen torch build while resolving the newest compatible
    # application dependencies together. This avoids selecting a CUDA13 wheel.
    constraint = OUT / 'official-torch-constraint.txt'
    constraint.write_text('torch==2.14.0+cu126\n')
    run('application-dependencies', pip + ['install', '--upgrade', '--upgrade-strategy', 'eager',
        '--only-binary=:all:', '--index-url', 'https://pypi.org/simple',
        '--constraint', str(constraint), '--report', str(OUT / 'application-report.json'), *DIRECT])
    run('official-pip-check', pip + ['check'])
    # A known, explicit upstream exact-pin exception. No metadata is rewritten.
    run('cudnn-qualified-override', pip + ['install', '--upgrade', '--no-deps',
        '--only-binary=:all:', '--index-url', 'https://pypi.org/simple',
        '--report', str(OUT / 'cudnn-override-report.json'), f'nvidia-cudnn-cu12=={OVERRIDE}'])
    check_status = run('final-pip-check', pip + ['check'], check=False)
    lines = [line for line in (OUT / 'final-pip-check.log').read_text().splitlines() if line.strip()]
    expected = ('torch 2.14.0+cu126 has requirement nvidia-cudnn-cu12==9.10.2.21; '
                'platform_system == "Linux", but you have nvidia-cudnn-cu12 9.25.1.1.')
    if check_status != 1 or lines != [expected]:
        raise RuntimeError(f'Unexpected final package-validation result: {lines}')
    save('declared-override.json', {'package': 'nvidia-cudnn-cu12', 'installed': OVERRIDE,
        'upstream_requirement': '==9.10.2.21', 'consumer': 'torch==2.14.0+cu126',
        'reason': '9.25.1 fixes the independently reproduced encoder-head numerical failure',
        'pip_check_exit': check_status, 'pip_check_messages': lines,
        'metadata_modified': False, 'full_new_environment_gpu_qualification': 'pending'})
    run('freeze', pip + ['freeze', '--all'])
    shutil.copyfile(OUT / 'freeze.log', OUT / 'requirements.lock')
    run('inspect', pip + ['inspect'])
    run('import-only', [python, '-c', r'''
import json,sys,importlib.metadata as m
from pathlib import Path
import torch,numpy,onnx,onnxruntime,pyarrow,huggingface_hub,soundfile,soxr,pydantic,pytest
from torch.utils.tensorboard import SummaryWriter
assert torch.__version__ == '2.14.0+cu126'
assert torch.version.cuda == '12.6'
assert m.version('nvidia-cudnn-cu12') == '9.25.1.1'
assert m.version('nvidia-nccl-cu12') == '2.29.3'
assert not torch.cuda._initialized  # Read a Python flag only; no CUDA API calls.
mapped=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
    if '/nvidia/' in line or '/torch/lib/' in line})
assert all('/venv214/' in path for path in mapped),mapped
print(json.dumps({'python':sys.version,'executable':sys.executable,'torch':torch.__version__,
 'torch_compiled_cuda':torch.version.cuda,'cuda_initialized':False,
 'sys_path':sys.path,'mapped_libraries':mapped,
 'packages':sorted([{'name':d.metadata['Name'],'version':d.version} for d in m.distributions()],key=lambda x:x['name'].lower())},indent=2))
'''])
    if original_metadata() != old_metadata:
        raise RuntimeError('Original environment metadata changed during isolation setup')
    save('complete.json', {'status': 'imports_passed_gpu_qualification_pending',
        'venv': str(VENV), 'original_environment_metadata_unchanged': True,
        'declared_pin_exceptions': 1, 'gpu_model_calls': 0,
        'free_bytes_after': shutil.disk_usage(BASE).free,
        'requirements_sha256': hashlib.sha256((OUT / 'requirements.lock').read_bytes()).hexdigest()})
    print((OUT / 'complete.json').read_text(), flush=True)


if __name__ == '__main__':
    main()
