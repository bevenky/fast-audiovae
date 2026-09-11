"""Archive completed pilot artifacts and verify every stored file's bytes."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path('/tmp/fast-audiovae-recovery-20260909')
DEST = Path('/workspace/fast-audiovae-convnext-20260909-r9/retained-candidates/teacher-refinements-20260910')
COMPLETE = ('target-v2', 'wn-v2', 'short-v2', 'pooled-v1', 'late-v3', 'aux-v3')
ATTEMPTS = ('target-v1', 'late-v2')

def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

if DEST.exists():
    raise FileExistsError('Existing archive destination is preserved')
retained = DEST.parent / 'joint-spectral-step8890-256-20260910'
expected = {
    'parent-step8890.pt': 'f7a91a2f3bef6fdec2a37643dc661a97b18a24b9e25b822df49c1c4017ddc948',
    'joint_spectral-heads.pt': 'b612baf57c5a3ae113ed67ea1a48f6f9cb2d740001aa5b2b5dc3a7a5ba9097e2',
}
for name, digest in expected.items():
    if sha(retained / name) != digest:
        raise RuntimeError('Retained original bytes changed: ' + name)
statuses = {}
files = []
for label in COMPLETE + ATTEMPTS:
    directory = ROOT / ('teacher-refinements-' + label)
    completion = directory / 'complete.json'
    statuses[label] = json.loads(completion.read_text()) if completion.exists() else {'complete': False}
    if label in COMPLETE and not statuses[label].get('complete'):
        raise RuntimeError('Pilot is not complete: ' + label)
    preservation = json.loads((directory / 'preservation.json').read_text())
    if not all(preservation.get(k) for k in ('base_checkpoint_preserved', 'retained_candidate_preserved',
                                             'original_engine_preserved', 'canonical_inputs_preserved')):
        raise RuntimeError('Missing preservation check: ' + label)
    files.extend(p for p in directory.rglob('*') if p.is_file())
    files.append(ROOT / (directory.name + '.log'))
files.extend(p for p in (ROOT / 'refinement-code-v2').rglob('*')
             if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts)
notes = ROOT / 'teacher-refinement-notes'
if notes.exists():
    files.extend(p for p in notes.rglob('*') if p.is_file())
manifest = {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(files)}
archive = ROOT / 'teacher-refinements-20260910.tar.gz'
with tarfile.open(archive, 'x:gz', compresslevel=3) as handle:
    for p in sorted(files):
        handle.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)
with tarfile.open(archive, 'r:gz') as handle:
    if {m.name for m in handle.getmembers()} != set(manifest):
        raise RuntimeError('Archive membership differs from manifest')
    for member in handle:
        stream = handle.extractfile(member)
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != manifest[member.name]['sha256']:
            raise RuntimeError('Archive content mismatch: ' + member.name)
size, digest = archive.stat().st_size, sha(archive)
if shutil.disk_usage(DEST.parent).free < size + 32 * 2**20:
    raise RuntimeError('Insufficient durable storage; verified temporary archive is preserved')
DEST.mkdir()
target = DEST / archive.name
shutil.copyfile(archive, target)
if sha(target) != digest:
    raise RuntimeError('Durable archive copy differs')
receipt = {'archive': str(target), 'archive_bytes': size, 'archive_sha256': digest,
           'verified_member_count': len(manifest), 'manifest': manifest,
           'completed_pilots': list(COMPLETE), 'pretraining_stopped_attempts': list(ATTEMPTS),
           'original_parent_sha256': 'f7a91a2f3bef6fdec2a37643dc661a97b18a24b9e25b822df49c1c4017ddc948',
           'retained_candidate_sha256': 'b612baf57c5a3ae113ed67ea1a48f6f9cb2d740001aa5b2b5dc3a7a5ba9097e2',
           'all_members_verified': True, 'durable_copy_verified': True, 'promoted': False,
           'training_data_included': False}
(DEST / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({k: v for k, v in receipt.items() if k != 'manifest'}))
