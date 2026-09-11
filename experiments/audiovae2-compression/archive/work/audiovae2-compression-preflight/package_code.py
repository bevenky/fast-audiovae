"""Package only this pilot and the unchanged helper modules it imports."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parents[2]
source=root/'work/fast-audiovae/experiments/audiovae2-compression'
shared=root/'work/fast-audiovae/experiments/convnext/audiovae_student'
files={p.name:p for p in source.glob('*.py')}
for path in (source/'data').glob('*'):
    if path.is_file() and path.suffix in ('.npz', '.json', '.md'):
        files['data/'+path.name]=path
for name in ('__init__.py','teacher.py','quiet_audio.py','reconstruction_v2.py','losses_distillation.py'):
    files['audiovae_student/'+name]=shared/name
destination=root/'work/audiovae2-compression-preflight/pilot-code.tar.gz'
manifest={name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in sorted(files.items())}
with tarfile.open(destination,'w:gz') as archive:
    for name,p in sorted(files.items()):archive.add(p,arcname=name)
receipt={'files':manifest,'archive_sha256':hashlib.sha256(destination.read_bytes()).hexdigest()}
(destination.parent/'code-manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'archive':str(destination),'file_count':len(files),'archive_sha256':receipt['archive_sha256']}))
