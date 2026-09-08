"""Prepare one further exact row fusion on the measured inline/VNNI candidate."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

EXPERIMENT = Path('/var/tmp/fast-audiovae-intel-precision/iteration3')
RESULTS = Path('/dev/shm/fast-audiovae-intel-iteration3')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    base = RESULTS/'inline-candidate-r3'
    build = RESULTS/'build-inline-r3/build.json'
    source = RESULTS/'inline-prepared-r3'
    prepared = RESULTS/'fused-inline-prepared-r1'
    output = RESULTS/'fused-inline-candidate-r1'
    if (output.exists() or prepared.exists() or
            sha(build) != 'baaf618251d85f7649d5f1e63dc58d96f362c0afade9dedc72bff4a7828364c2'):
        raise ValueError('Fresh outputs and pinned parent build required')
    inputs = json.loads(build.read_text())['input_sha256']
    for path, digest in inputs.items():
        if sha(Path(path)) != digest:
            raise ValueError('Parent build input changed: '+path)
    before = {str(p.relative_to(base)): sha(p) for p in base.rglob('*') if p.is_file()}
    subprocess.run([sys.executable, str(EXPERIMENT/'sleef-fused-row/apply_to_copy.py'),
                    '--inline-source', str(source), '--source-manifest-sha256', sha(source/'source-provenance.json'),
                    '--output-dir', str(prepared)], check=True)
    shutil.copytree(base, output)
    manifest = json.loads((prepared/'source-provenance.json').read_text())
    for name, digest in manifest['outputs'].items():
        if sha(prepared/name) != digest:
            raise ValueError('Prepared source hash mismatch')
        shutil.copy2(prepared/name, output/'pipeline'/name)
    if before != {str(p.relative_to(base)): sha(p) for p in base.rglob('*') if p.is_file()}:
        raise ValueError('Parent source changed')
    record = {'base_file_sha256': before, 'base': str(base), 'output': str(output),
              'parent_build_sha256': sha(build), 'source_manifest': manifest,
              'source_manifest_sha256': sha(prepared/'source-provenance.json'),
              'script_sha256': sha(Path(__file__).resolve()),
              'changes': 'Only AVX512 DW/post-Snake register fusion in the row helper'}
    (output/'fused-variant.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'status': 'prepared', 'candidate': str(output)}))


if __name__ == '__main__':
    main()
