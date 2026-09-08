"""Layer exact sine inlining onto the already measured row-fusion/VNNI hybrid."""
import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', required=True, choices=('r2', 'r3'))
    args = parser.parse_args()
    base = EXPERIMENT/'hybrid-candidate-r1'
    output = RESULTS/('inline-candidate-'+args.revision)
    prepared = RESULTS/('inline-prepared-'+args.revision)
    generated = RESULTS/'sleef-inline-generation-r1/generation.json'
    provenance = json.loads(generated.read_text())
    header = Path(provenance['header'])
    row = base/'pipeline/row_nonlinear.h'
    if (output.exists() or prepared.exists() or provenance['status'] != 'generated'
            or sha(header) != provenance['header_sha256']):
        raise ValueError('Fresh output and verified generated header required')
    before = {str(p.relative_to(base)): sha(p) for p in base.rglob('*') if p.is_file()}
    if sha(base/'core/precision.cpp') != 'f3548d164cadeb3597d5f051f471d322788c262f4efa3951109c3bc534a7a90d':
        raise ValueError('Wrong measured hybrid core')
    subprocess.run([sys.executable, str(EXPERIMENT/'sleef-inline/apply_to_copy.py'),
                    '--row-source', str(row), '--row-sha256', sha(row),
                    '--generated-header', str(header),
                    '--generated-header-sha256', provenance['header_sha256'],
                    '--sleef-source-archive', provenance['source_archive'],
                    '--output-dir', str(prepared)], check=True)
    shutil.copytree(base, output)
    for name in ('row_nonlinear.h', 'sleef_inline_avx512.h', 'sleefinline_avx512f.h', 'check_sine.cpp'):
        shutil.copy2(prepared/name, output/'pipeline'/name)
    if before != {str(p.relative_to(base)): sha(p) for p in base.rglob('*') if p.is_file()}:
        raise ValueError('Base inputs changed')
    record = {'base_file_sha256': before, 'base': str(base), 'output': str(output),
              'generation_manifest_sha256': sha(generated),
              'prepared_manifest_sha256': sha(prepared/'source-provenance.json'),
              'script_sha256': sha(Path(__file__).resolve()),
              'changes': 'Only the 16-lane sine call and its generated-header wrapper',
              'core_sha256': sha(output/'core/precision.cpp')}
    (output/'inline-variant.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'status': 'prepared', 'candidate': str(output)}))


if __name__ == '__main__':
    main()
