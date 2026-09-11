"""Preserve the rejected screen and install the tested loss-only fix."""
from pathlib import Path
import hashlib
import json
import shutil
import tarfile

root = Path('/workspace/fast-audiovae-compression-20260910-v1')
old = json.loads((root/'code-manifest-singleton.json').read_text())
new = json.loads((root/'code-manifest-settings-screen-deterministic-v2.json').read_text())
changed = {name for name in old['files'] | new['files']
           if old['files'].get(name) != new['files'].get(name)}
assert changed == {'author_mel.py', 'test_author_mel.py'}, changed
archive = root/'settings-screen-code-deterministic-v2.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == new['archive_sha256']
prior = json.loads((root/'screen-singleton-pid.json').read_text())
process = Path('/proc')/str(prior['pid'])/'cmdline'
assert not process.exists() or b'settings_screen.py' not in process.read_bytes()
assert list((root/'settings-screen-v1').iterdir()) == [root/'settings-screen-v1/screen-identity.json']
for source, target in [
    ('settings-screen-v1', 'settings-screen-native-padding-rejected-v1'),
    ('pilot/screen.log', 'pilot/screen-native-padding-rejected.log'),
    ('screen-singleton-pid.json', 'screen-native-padding-rejected-pid.json'),
]:
    assert not (root/target).exists()
    (root/source).rename(root/target)
shutil.copyfile(root/'code-manifest-singleton.json', root/'code-manifest-before-deterministic-v2.json')
with tarfile.open(archive) as handle:
    handle.extractall(root/'code', filter='data')
for name, expected in new['files'].items():
    assert hashlib.sha256((root/'code'/name).read_bytes()).hexdigest() == expected, name
shutil.copyfile(root/'code-manifest-settings-screen-deterministic-v2.json', root/'code-manifest-singleton.json')
print(json.dumps({'installed': True, 'changed_files': sorted(changed), 'retained_training_updates': 0}))
