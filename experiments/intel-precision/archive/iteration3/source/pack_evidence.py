"""Archive completed CPU experiment evidence without model weights or user audio."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path('/dev/shm/fast-audiovae-intel-iteration3')
DESTINATION = ROOT/'evidence-final-r1.tar.gz'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if DESTINATION.exists():
        raise ValueError('Fresh archive required')
    files = set(ROOT.glob('*checks*.json'))
    files.update(ROOT.glob('screen-*/results.json'))
    files.update(ROOT.glob('build-*/build.json'))
    files.update(ROOT.glob('graph-*/candidate.json'))
    files.update(ROOT.glob('diagnostic-*/*.json'))
    files.update(ROOT.glob('diagnostic-*/*.npz'))
    files.update(ROOT.glob('*disassembly.txt'))
    files.update(ROOT.glob('*prepare-*.log'))
    for name in ('inline-candidate-r3', 'inline-prepared-r3',
                 'fused-inline-candidate-r1', 'fused-inline-prepared-r1'):
        directory = ROOT/name
        if directory.exists():
            files.update(p for p in directory.rglob('*') if p.is_file()
                         and p.suffix in ('.json', '.h', '.cpp', '.c', '.py', '.md', '.sh'))
    generation = ROOT/'sleef-inline-generation-r1'
    files.update(p for p in (generation/'generation.json', generation/'generation.log',
                            generation/'sleef-3.9.0/LICENSE.txt') if p.exists())
    for path in files:
        if path.name == 'results.json':
            report = json.loads(path.read_text())
            if report.get('status') not in ('complete', 'failed'):
                raise ValueError('Do not snapshot a running result: '+str(path))
    manifest = {'gpu_used': False, 'models_or_audio_included': False,
                'files': {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)},
                'script_sha256': sha(Path(__file__).resolve())}
    manifest_path = ROOT/'evidence-final-r1-manifest.json'
    if manifest_path.exists():
        raise ValueError('Fresh manifest required')
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    with tarfile.open(DESTINATION, 'w:gz') as archive:
        for path in sorted(files | {manifest_path}):
            archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    if any(sha(ROOT/name) != digest for name, digest in manifest['files'].items()):
        raise ValueError('Evidence changed during packing')
    print(json.dumps({'archive': str(DESTINATION), 'sha256': sha(DESTINATION),
                      'bytes': DESTINATION.stat().st_size, 'files': len(files)}))


if __name__ == '__main__':
    main()
