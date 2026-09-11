"""Collect completed continuation reports, excluding weights and audio."""
from pathlib import Path
import hashlib
import json
import tarfile

root = Path('/workspace/fast-audiovae-compression-20260910-v1')
directory = root/'current-lowrate-continue1000-v1'
assert (directory/'completed.json').exists()
paths = sorted(p for p in directory.iterdir() if p.suffix in ('.json','.jsonl'))
paths += [root/'continuation-quiet-addendum-v1.json', root/'continuation-singleton-pid.json',
          root/'pilot/current-lowrate-continue1000-v1.log']
out = root/'continuation-reports-v1.tar.gz'
assert not out.exists()
with tarfile.open(out,'w:gz') as handle:
    for path in paths: handle.add(path,arcname=str(path.relative_to(root)))
print(json.dumps({'bytes':out.stat().st_size,'files':len(paths),
                  'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
