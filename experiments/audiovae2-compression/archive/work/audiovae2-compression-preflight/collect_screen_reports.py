"""Collect only small reports from the completed settings comparison."""
from pathlib import Path
import hashlib
import json
import tarfile

root = Path('/workspace/fast-audiovae-compression-20260910-v1')
screen = root/'settings-screen-v1'
assert (screen/'completed.json').is_file(), 'The four-run screen is not complete'
files = sorted(p for p in screen.rglob('*') if p.suffix in ('.json', '.jsonl'))
files += [root/'pilot/screen.log', root/'pilot/export-check.json', root/'pilot/streaming-check.json',
          root/'code-manifest-singleton.json', root/'screen-singleton-pid.json']
archive = root/'settings-screen-reports-v1.tar.gz'
assert not archive.exists()
with tarfile.open(archive, 'w:gz') as handle:
    for path in files:
        handle.add(path, arcname=str(path.relative_to(root)))
print(json.dumps({'files': len(files), 'bytes': archive.stat().st_size,
                  'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}))
