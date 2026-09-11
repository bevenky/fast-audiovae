"""Copy completed small result files; preserve fresh checkpoints on the volume."""
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


base=Path('/tmp/fast-audiovae-recovery-20260909')
src=base/'corrected-update-400-v1'
status=json.loads((src/'status.json').read_text())
if status['stage']!='complete':raise ValueError('Experiment not complete')
files={str(p.relative_to(src)):json.loads(p.read_text()) for p in sorted(src.rglob('*.json'))}
target=Path('/workspace/fast-audiovae-convnext-20260909-r9/remediation/corrected-update-400-reference-v1')
models=[src/name/'final.pt' for name in ('current_rate','quarter_rate')]
needed=sum(p.stat().st_size for p in models)+sum(p.stat().st_size for p in src.rglob('*.json'))
if target.exists():raise FileExistsError('Refusing to overwrite preserved results')
if shutil.disk_usage(target.parent).free < needed + 512*1024**2:
    raise ValueError('Insufficient volume space with512MiB reserve; temporary outputs preserved')
target.mkdir()
preservation=[]
for p in models:
    dest=target/p.relative_to(src)
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(p,dest)
    actual=sha(dest)
    complete=files[str(p.parent.relative_to(src))+'/complete.json']
    if actual!=sha(p) or actual!=complete['checkpoint_sha256']:
        raise ValueError('Checkpoint copy checksum differs')
    preservation.append({'path':str(dest),'sha256':actual,'bytes':dest.stat().st_size})
for name,value in files.items():
    p=target/name;p.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(src/name,p)
for p in sorted(src.rglob('*.jsonl')):
    dest=target/p.relative_to(src);dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(p,dest)
for p in (base/'corrected-training-resolution.json',base/'corrected-update-400-v1.log'):
    shutil.copyfile(p,target/p.name)
bundle={'files':files,'checkpoint_preservation':preservation,'durable_result_directory':str(target),
    'source_result_directory':str(src),'promoted':False}
out=base/'corrected-update-400-results.json'
out.write_text(json.dumps(bundle,sort_keys=True)+'\n')
(target/'checkpoint-preservation.json').write_text(json.dumps(preservation,indent=2)+'\n')
print(json.dumps({'bundle':str(out),'bytes':out.stat().st_size,'preserved':preservation}))
