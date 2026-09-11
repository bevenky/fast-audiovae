import fcntl, json, re, shutil, sqlite3
from pathlib import Path
root = Path('/workspace/fast-audiovae-convnext-20260909-r6/teacher-cache-train')
output = Path('/workspace/fast-audiovae-convnext-20260909-r9/reclaimed-r6-target-cache')
output.mkdir(exist_ok=False)
with (root / '.writer.lock').open('a+b') as lock:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    paths = sorted((root / 'targets').glob('*.pt'))
    if any(p.is_symlink() or not re.fullmatch(r'[a-f0-9]{64}\.pt', p.name) for p in paths):
        raise ValueError('Unexpected target path')
    inventory = [{'name': p.name, 'bytes': p.stat().st_size} for p in paths]
    shutil.copy2(root / 'identity.json', output / 'identity.json')
    db = sqlite3.connect(root / 'index.sqlite3')
    backup = sqlite3.connect(output / 'index.sqlite3')
    db.backup(backup)
    backup.close()
    for p in paths:
        p.unlink()
    db.execute('DELETE FROM entries')
    db.commit()
    db.close()
    report = {'scope': 'Reproducible frozen teacher targets only; original audio and checkpoints preserved',
              'cache': str(root), 'targets': inventory, 'reclaimed_bytes': sum(r['bytes'] for r in inventory)}
    (output / 'inventory.json').write_text(json.dumps(report))
    print(json.dumps({k: v for k, v in report.items() if k != 'targets'}))
