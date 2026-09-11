"""Verify and archive the completed bounded pilot without promoting it."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

root=Path('/tmp/fast-audiovae-recovery-20260909')
run=root/'teacher-l1-v1'
dest=Path('/workspace/fast-audiovae-convnext-20260909-r9/retained-candidates/teacher-l1-20260910')
completion=json.loads((run/'complete.json').read_text())
preserved=json.loads((run/'preservation.json').read_text())
assert completion['complete'] and not completion['promoted']
assert all(preserved[k] for k in ['original_engine_preserved','base_checkpoint_preserved','retained_candidate_preserved','canonical_inputs_preserved'])
assert not dest.exists()
def sha(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
files=[p for directory in [run,root/'l1-code-v1'] for p in directory.rglob('*') if p.is_file() and not any(x in p.parts for x in ['__pycache__','.pytest_cache'])]
files.append(root/'teacher-l1-v1.log')
notes=root/'l1-notes-v1'
if notes.exists():files.extend(p for p in notes.rglob('*') if p.is_file())
manifest={str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)}
archive=root/'teacher-l1-20260910.tar.gz'
with tarfile.open(archive,'x:gz',compresslevel=3) as tar:
 for p in sorted(files):tar.add(p,arcname=str(p.relative_to(root)),recursive=False)
with tarfile.open(archive,'r:gz') as tar:
 assert {x.name for x in tar.getmembers()}==set(manifest)
 for member in tar:
  with tar.extractfile(member) as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
  assert actual==manifest[member.name]['sha256'],member.name
assert shutil.disk_usage(dest.parent).free>archive.stat().st_size+32*2**20,'Insufficient durable free space; scratch archive retained'
dest.mkdir()
target=dest/archive.name
shutil.copyfile(archive,target)
assert sha(target)==sha(archive)
receipt={'archive':str(target),'archive_bytes':target.stat().st_size,'archive_sha256':sha(target),'manifest':manifest,'verified_member_count':len(manifest),'all_members_verified':True,'durable_copy_verified':True,'retained_candidate_preserved':True,'original_checkpoint_preserved':True,'promoted':False,'training_data_included':False}
(dest/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='manifest'}))
