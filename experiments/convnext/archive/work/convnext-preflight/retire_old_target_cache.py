"""Remove only reproducible targets from the stopped, previous training cache."""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import time

root = Path('/workspace/fast-audiovae-convnext-20260908-r1/target-cache-500h')
assert root.is_dir() and not root.is_symlink()
targets = root / 'targets'
assert targets.is_dir() and not targets.is_symlink()
with (root / '.writer.lock').open('rb') as lock:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    report = root / 'retired-targets.json'
    if report.exists():
        raise RuntimeError('Already retired; inspect original report')
    # SourceCorpus recovers missing target files from its existing index on open.
    # Keep that index and all identity/qualification evidence unchanged.
    files = list(targets.iterdir())
    if any(p.is_symlink() or not p.is_file() or not re.fullmatch(r'[0-9a-f]{64}\.pt', p.name) for p in files):
        raise RuntimeError('Unexpected cache contents; nothing removed')
    database = sqlite3.connect((root / 'index.sqlite3').as_uri() + '?mode=ro', uri=True)
    entries = {k: json.loads(v) for k, v in database.execute('SELECT lookup_key,info FROM entries')}
    database.close()
    if any(p.stem not in entries or p.stat().st_size != entries[p.stem]['bytes'] for p in files):
        raise RuntimeError('Target inventory differs from index; nothing removed')
    inventory = [{'name': p.name, 'bytes': p.stat().st_size, 'index': entries[p.stem]} for p in files]
    metadata = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.iterdir() if p.is_file() and p.name != '.writer.lock'}
    result = {'state': 'retiring', 'time_unix': time.time(), 'target_inventory': inventory,
        'metadata_sha256': metadata, 'retired_bytes': sum(x['bytes'] for x in inventory),
        'policy': 'only generated old teacher targets; source audio, checkpoints, index and qualification retained'}
    report.write_text(json.dumps(result, indent=2))
    for path in files:
        path.unlink()
    result['state'] = 'retired'
    result['free_bytes_after'] = shutil.disk_usage(root).free
    report.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ('target_inventory', 'metadata_sha256')}))
